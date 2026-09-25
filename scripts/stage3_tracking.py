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


def _load_inputs(stage1_bundle, stage2_bundle) -> dict:
    stage1_root = Path(stage1_bundle).resolve()
    stage1_report_path = stage1_root / "report.json"
    if not stage1_report_path.is_file():
        raise FileNotFoundError(f"Stage 1 report not found: {stage1_report_path}")
    stage1 = _read_json(stage1_report_path)
    if stage1.get("schema_version") != STAGE1_SCHEMA or stage1.get("stage") != 1:
        raise ValueError("the selected directory is not a Stage 1 bundle")
    if stage1.get("pass") is not True:
        raise ValueError("Stage 3 refuses a failed Stage 1 bundle")
    try:
        outputs = stage1["outputs"]
        declared = {
            "width": int(stage1["input"]["width"]),
            "height": int(stage1["input"]["height"]),
            "frame_count": int(stage1["input"]["frame_count"]),
            "fps": float(stage1["input"]["fps"]),
        }
    except (KeyError, TypeError, ValueError) as exception:
        raise ValueError("Stage 1 report is missing timeline metadata") from exception
    if any(declared[key] <= 0 for key in declared):
        raise ValueError("Stage 1 timeline metadata is invalid")

    video_path = _bundle_path(stage1_root, outputs.get("video_cfr"), "CFR video")
    transforms_path = _bundle_path(
        stage1_root, outputs.get("frame_transforms"), "frame transforms")
    ground_path = _bundle_path(stage1_root, outputs.get("ground"), "ground image")
    actual_video = _video_info(video_path)
    for key in ("width", "height", "frame_count"):
        if actual_video[key] != declared[key]:
            raise ValueError(
                f"CFR video {key}={actual_video[key]} does not match "
                f"Stage 1 report {declared[key]}")
    if abs(actual_video["fps"] - declared["fps"]) > max(1e-3, declared["fps"] * 1e-4):
        raise ValueError("CFR video FPS does not match the Stage 1 report")

    transforms = np.load(transforms_path, allow_pickle=False)
    expected_shape = (declared["frame_count"], 3, 3)
    if transforms.shape != expected_shape:
        raise ValueError(
            f"frame transforms have shape {transforms.shape}; expected {expected_shape}")
    if not np.all(np.isfinite(transforms)):
        raise ValueError("frame transforms contain non-finite values")
    determinants = np.linalg.det(transforms)
    if np.any(np.abs(determinants) < 1e-10):
        raise ValueError("frame transforms contain a singular matrix")

    stage2_root = Path(stage2_bundle).resolve()
    calibration_path = stage2_root / "world_calibration.json"
    if not calibration_path.is_file():
        raise FileNotFoundError(f"Stage 2 calibration not found: {calibration_path}")
    stage2 = _read_json(calibration_path)
    if stage2.get("schema_version") != STAGE2_SCHEMA or stage2.get("stage") != 2:
        raise ValueError("the selected directory is not a Stage 2 calibration bundle")
    if stage2.get("pass") is not True:
        raise ValueError("Stage 3 refuses a failed Stage 2 calibration")
    try:
        source2 = stage2["source_stage1"]
        matrix = np.asarray(stage2["reference_pixel_to_world_xz"], dtype=np.float64)
        confidence = str(stage2["confidence"])
        coordinate_system = stage2["coordinate_system"]
    except (KeyError, TypeError, ValueError) as exception:
        raise ValueError("Stage 2 calibration is missing required data") from exception
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("Stage 2 pixel-to-world transform is not a finite 3x3 matrix")
    if abs(float(np.linalg.det(matrix))) < 1e-12:
        raise ValueError("Stage 2 pixel-to-world transform is singular")
    if source2.get("report_sha256") != _sha256(stage1_report_path):
        raise ValueError("Stage 2 calibration is stale: Stage 1 report hash changed")
    if source2.get("ground_sha256") != _sha256(ground_path):
        raise ValueError("Stage 2 calibration is stale: Stage 1 ground hash changed")
    if source2.get("source_video_sha256") != stage1["input"].get("sha256"):
        raise ValueError("Stage 2 calibration was created for a different source video")
    if (int(source2.get("pixel_width", -1)) != declared["width"] or
            int(source2.get("pixel_height", -1)) != declared["height"]):
        raise ValueError("Stage 2 calibration dimensions do not match Stage 1")

    ground = cv2.imread(str(ground_path), cv2.IMREAD_COLOR)
    if ground is None or ground.shape[:2] != (declared["height"], declared["width"]):
        raise ValueError("Stage 1 ground image is unreadable or dimensionally inconsistent")
    return {
        "stage1_root": stage1_root,
        "stage1": stage1,
        "stage1_report_path": stage1_report_path,
        "stage2_root": stage2_root,
        "stage2": stage2,
        "calibration_path": calibration_path,
        "video_path": video_path,
        "video_sha256": _sha256(video_path),
        "transforms_path": transforms_path,
        "transforms_sha256": _sha256(transforms_path),
        "transforms": transforms,
        "ground_path": ground_path,
        "ground": ground,
        "pixel_to_world": matrix,
        "scale_confidence": confidence,
        "metric_scale_is_assumed": bool(
            coordinate_system.get("metric_scale_is_assumed", False)),
        "ground_y": float(coordinate_system.get("ground_y_m", 0.0)),
        **declared,
    }


def _project_point(homography: np.ndarray, u: float, v: float) -> tuple[float, float]:
    projected = homography @ np.array([u, v, 1.0], dtype=np.float64)
    denominator = float(projected[2])
    if not math.isfinite(denominator) or abs(denominator) < 1e-10:
        raise ValueError("actor point mapped to invalid projective coordinates")
    x = float(projected[0] / denominator)
    y = float(projected[1] / denominator)
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("actor point mapped to non-finite coordinates")
    return x, y


def _transform_box(homography: np.ndarray, box: Iterable[float]) -> list[float]:
    x1, y1, x2, y2 = (float(value) for value in box)
    points = np.array(
        [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], dtype=np.float64)
    mapped = cv2.perspectiveTransform(points, homography)[0]
    if not np.all(np.isfinite(mapped)):
        raise ValueError("actor box mapped to non-finite reference coordinates")
    return [
        float(np.min(mapped[:, 0])), float(np.min(mapped[:, 1])),
        float(np.max(mapped[:, 0])), float(np.max(mapped[:, 1])),
    ]


def _iou(first: Iterable[float], second: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = width * height
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def _appearance_descriptor(frame: np.ndarray, box: Iterable[float]) -> np.ndarray | None:
    """Return a compact HSV histogram for conservative short-gap association."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (float(value) for value in box)
    # Use the central region to reduce background included by an axis-aligned box.
    inset_x = 0.12 * (x2 - x1)
    inset_y = 0.12 * (y2 - y1)
    left = max(0, min(width, int(math.floor(x1 + inset_x))))
    top = max(0, min(height, int(math.floor(y1 + inset_y))))
    right = max(0, min(width, int(math.ceil(x2 - inset_x))))
    bottom = max(0, min(height, int(math.ceil(y2 - inset_y))))
    if right - left < 2 or bottom - top < 2:
        return None
    crop = frame[top:bottom, left:right]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1, 2], None, [12, 4, 4],
                             [0, 180, 0, 256, 0, 256]).reshape(-1)
    total = float(histogram.sum())
    if total <= 0.0 or not np.all(np.isfinite(histogram)):
        return None
    return (histogram / total).astype(np.float32)


def _appearance_distance(first: np.ndarray | None,
                         second: np.ndarray | None) -> float | None:
    if first is None or second is None or first.shape != second.shape:
        return None
    coefficient = float(np.sum(np.sqrt(np.maximum(first, 0.0) *
                                       np.maximum(second, 0.0))))
    return math.sqrt(max(0.0, 1.0 - min(1.0, coefficient)))


def _canonical_nms(candidates: list[dict], threshold: float) -> list[dict]:
    merged = []
    for kind in ("car", "person"):
        group = [candidate for candidate in candidates if candidate["kind"] == kind]
        group.sort(key=lambda item: (-item["confidence"], item["bbox_xyxy"]))
        while group:
            best = group.pop(0)
            sources = {best["inference_pass"]}
            remaining = []
            for candidate in group:
                if _iou(best["bbox_xyxy"], candidate["bbox_xyxy"]) > threshold:
                    sources.add(candidate["inference_pass"])
                else:
                    remaining.append(candidate)
            best = dict(best)
            best["inference_passes"] = sorted(sources)
            best.pop("inference_pass", None)
            merged.append(best)
            group = remaining
    merged.sort(key=lambda item: (item["kind"], -item["confidence"], item["bbox_xyxy"]))
    return merged
