"""Stage 2: bind a passing Stage 1 ground image to metric Unity coordinates.

The calibration is deliberately explicit. It never estimates scale from detected
vehicles: the caller supplies ground-control points whose image pixels and real-world
X/Z coordinates are both known.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile

import cv2
import numpy as np


STAGE1_SCHEMA = "stage1-ground-v1"
CONTROL_POINTS_SCHEMA = "stage2-control-points-v1"
CALIBRATION_SCHEMA = "stage2-calibration-v1"
OUTPUT_MARKER = ".stage2-calibration-output"


@dataclass(frozen=True)
class Stage2Options:
    min_points: int = 4
    recommended_points: int = 6
    min_image_hull_area_pct: float = 8.0
    min_world_hull_area_m2: float = 1.0
    max_reprojection_rmse_m: float = 0.25
    max_reprojection_error_m: float = 0.75
    max_local_anisotropy: float = 3.0
    validation_grid_size: int = 9

    def validate(self) -> None:
        if self.min_points < 4:
            raise ValueError("Stage 2 requires at least four control points")
        if self.recommended_points < self.min_points:
            raise ValueError("recommended_points cannot be below min_points")
        if not 0.0 < self.min_image_hull_area_pct <= 100.0:
            raise ValueError("min_image_hull_area_pct must be in (0, 100]")
        if self.min_world_hull_area_m2 <= 0.0:
            raise ValueError("min_world_hull_area_m2 must be positive")
        if self.max_reprojection_rmse_m <= 0.0:
            raise ValueError("max_reprojection_rmse_m must be positive")
        if self.max_reprojection_error_m < self.max_reprojection_rmse_m:
            raise ValueError("max_reprojection_error_m must be at least the RMSE limit")
        if self.max_local_anisotropy < 1.0:
            raise ValueError("max_local_anisotropy must be at least one")
        if self.validation_grid_size < 3:
            raise ValueError("validation_grid_size must be at least three")


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


def _finite_number(value, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exception:
        raise ValueError(f"{label} must be a finite number") from exception
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number
