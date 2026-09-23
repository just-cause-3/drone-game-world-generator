"""Stage 3: cars and pedestrians from the validated video to Unity trajectories."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Iterable

import cv2
import numpy as np

from .common import log
from .stage1_ground import (
    DEFAULT_FOREGROUND_MODEL,
    _normalized_class_name,
    _tile_origins,
)
from .track_agents import resolve_device
from setup_drone_model import EXPECTED_CLASSES, MODEL_SHA256


STAGE1_SCHEMA = "stage1-ground-v1"
STAGE2_SCHEMA = "stage2-calibration-v1"
STAGE3_SCHEMA = "stage3-tracks-v1"
OUTPUT_MARKER = ".stage3-tracks-output"
CANONICAL_CLASSES = {
    "car": "car",
    "pedestrian": "person",
    "people": "person",
    "person": "person",
}


@dataclass(frozen=True)
class Stage3Options:
    detector_model: str = DEFAULT_FOREGROUND_MODEL
    detector_imgsz: int = 1280
    detector_low_confidence: float = 0.08
    detector_high_confidence: float = 0.25
    detector_iou: float = 0.45
    detector_max_detections: int = 3000
    detector_full_frame: bool = True
    detector_tile_size: int = 640
    detector_tile_overlap: float = 0.25
    detector_tta: bool = True
    min_confirmed_observations: int = 5
    max_lost_frames: int = 3
    max_interpolation_frames: int = 3
    car_link_diagonal_fraction_per_frame: float = 0.015
    person_link_diagonal_fraction_per_frame: float = 0.006
    minimum_link_distance_px: float = 6.0
    maximum_size_ratio: float = 2.2
    appearance_max_distance: float = 0.72
    appearance_cost_weight: float = 0.40
    process_noise: float = 1.0
    measurement_noise: float = 2.0
    boundary_tolerance_fraction: float = 0.002
    preview_line_thickness: int = 2

    def validate(self) -> None:
        if self.detector_imgsz < 320:
            raise ValueError("detector_imgsz must be at least 320")
        if not 0.0 < self.detector_low_confidence < 1.0:
            raise ValueError("detector_low_confidence must be in (0, 1)")
        if not self.detector_low_confidence < self.detector_high_confidence < 1.0:
            raise ValueError(
                "detector_high_confidence must exceed the low threshold and be below 1")
        if not 0.0 < self.detector_iou < 1.0:
            raise ValueError("detector_iou must be in (0, 1)")
        if self.detector_max_detections < 1:
            raise ValueError("detector_max_detections must be positive")
        if self.detector_tile_size != 0 and self.detector_tile_size < 128:
            raise ValueError("detector_tile_size must be zero or at least 128")
        if not 0.0 <= self.detector_tile_overlap < 0.9:
            raise ValueError("detector_tile_overlap must be in [0, 0.9)")
        if self.min_confirmed_observations < 1:
            raise ValueError("min_confirmed_observations must be positive")
        if self.max_lost_frames < 0 or self.max_interpolation_frames < 0:
            raise ValueError("gap limits cannot be negative")
        if self.max_interpolation_frames > self.max_lost_frames:
            raise ValueError("interpolation cannot exceed the tracker lost-frame limit")
        if self.minimum_link_distance_px <= 0.0:
            raise ValueError("minimum_link_distance_px must be positive")
        if self.maximum_size_ratio < 1.0:
            raise ValueError("maximum_size_ratio must be at least one")
        if not 0.0 < self.appearance_max_distance <= 1.0:
            raise ValueError("appearance_max_distance must be in (0, 1]")
        if self.appearance_cost_weight < 0.0:
            raise ValueError("appearance_cost_weight cannot be negative")
        if self.process_noise <= 0.0 or self.measurement_noise <= 0.0:
            raise ValueError("filter noise values must be positive")
        if not 0.0 <= self.boundary_tolerance_fraction <= 0.5:
            raise ValueError("boundary_tolerance_fraction must be in [0, 0.5]")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exception:
        raise ValueError(f"invalid JSON in {path}: {exception}") from exception
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _bundle_path(root: Path, relative, label: str) -> Path:
    if not isinstance(relative, str) or not relative.strip():
        raise ValueError(f"missing {label} output path")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exception:
        raise ValueError(f"{label} path escapes its bundle") from exception
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _video_info(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"could not open Stage 1 CFR video: {path}")
    try:
        result = {
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        }
    finally:
        capture.release()
    if any(result[key] <= 0 for key in result):
        raise ValueError("Stage 1 CFR video has invalid metadata")
    return result
