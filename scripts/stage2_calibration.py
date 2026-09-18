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


def _load_stage1_bundle(bundle_directory) -> tuple[Path, dict, Path]:
    root = Path(bundle_directory).resolve()
    report_path = root / "report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"Stage 1 report not found: {report_path}")
    report = _read_json(report_path)
    if report.get("schema_version") != STAGE1_SCHEMA or report.get("stage") != 1:
        raise ValueError("the selected directory is not a Stage 1 ground bundle")
    if report.get("pass") is not True:
        raise ValueError("Stage 2 refuses a Stage 1 bundle that failed its quality gate")
    try:
        width = int(report["input"]["width"])
        height = int(report["input"]["height"])
        ground_relative = report["outputs"]["ground"]
    except (KeyError, TypeError, ValueError) as exception:
        raise ValueError("Stage 1 report is missing input dimensions or ground output") from exception
    if width <= 0 or height <= 0:
        raise ValueError("Stage 1 report contains invalid image dimensions")
    ground_path = (root / ground_relative).resolve()
    try:
        ground_path.relative_to(root)
    except ValueError as exception:
        raise ValueError("Stage 1 ground path escapes its bundle")
    if not ground_path.is_file():
        raise FileNotFoundError(f"Stage 1 ground image not found: {ground_path}")
    ground = cv2.imread(str(ground_path), cv2.IMREAD_UNCHANGED)
    if ground is None:
        raise ValueError(f"Stage 1 ground image is unreadable: {ground_path}")
    if ground.shape[1] != width or ground.shape[0] != height:
        raise ValueError(
            f"Stage 1 ground is {ground.shape[1]}x{ground.shape[0]}, "
            f"but report declares {width}x{height}")
    return root, report, ground_path


def _load_control_points(path, width: int, height: int, minimum: int):
    control_path = Path(path).resolve()
    data = _read_json(control_path)
    if data.get("schema_version") != CONTROL_POINTS_SCHEMA:
        raise ValueError(
            f"unsupported control-point schema {data.get('schema_version')!r}; "
            f"expected {CONTROL_POINTS_SCHEMA!r}")
    if data.get("units") != "meters":
        raise ValueError("Stage 2 control-point units must be 'meters'")
    raw_points = data.get("points")
    if not isinstance(raw_points, list) or len(raw_points) < minimum:
        raise ValueError(f"Stage 2 requires at least {minimum} control points")

    points = []
    identifiers = set()
    for index, raw in enumerate(raw_points):
        if not isinstance(raw, dict):
            raise ValueError(f"control point {index} must be an object")
        identifier = str(raw.get("id", "")).strip()
        if not identifier:
            raise ValueError(f"control point {index} has no id")
        if identifier in identifiers:
            raise ValueError(f"duplicate control-point id: {identifier}")
        identifiers.add(identifier)
        try:
            image = raw["image"]
            world = raw["world"]
            u = _finite_number(image["u"], f"{identifier}.image.u")
            v = _finite_number(image["v"], f"{identifier}.image.v")
            x = _finite_number(world["x"], f"{identifier}.world.x")
            z = _finite_number(world["z"], f"{identifier}.world.z")
        except (KeyError, TypeError) as exception:
            raise ValueError(
                f"control point {identifier!r} must define image u/v and world x/z") \
                from exception
        if not 0.0 <= u <= width - 1 or not 0.0 <= v <= height - 1:
            raise ValueError(
                f"control point {identifier!r} is outside the {width}x{height} image")
        points.append({
            "id": identifier,
            "image": {"u": u, "v": v},
            "world": {"x": x, "z": z},
        })

    image_points = np.array(
        [[point["image"]["u"], point["image"]["v"]] for point in points],
        dtype=np.float64)
    world_points = np.array(
        [[point["world"]["x"], point["world"]["z"]] for point in points],
        dtype=np.float64)
    if len(np.unique(np.round(image_points, 6), axis=0)) != len(points):
        raise ValueError("control points contain duplicate image coordinates")
    if len(np.unique(np.round(world_points, 6), axis=0)) != len(points):
        raise ValueError("control points contain duplicate world coordinates")

    metadata = {
        "calibration_mode": str(
            data.get("calibration_mode", "measured_control_points")).strip(),
        "assumed_world_width_m": (
            _finite_number(data["assumed_world_width_m"],
                           "assumed_world_width_m")
            if data.get("assumed_world_width_m") is not None else None),
        "origin_description": str(data.get("origin_description", "")).strip(),
        "x_axis_description": str(data.get("x_axis_description", "right/east")).strip(),
        "z_axis_description": str(data.get("z_axis_description", "forward/north")).strip(),
        "ground_y_m": _finite_number(data.get("ground_y_m", 0.0), "ground_y_m"),
    }
    if metadata["calibration_mode"] not in {
            "measured_control_points", "approximate_image_extent"}:
        raise ValueError(
            f"unsupported calibration_mode: {metadata['calibration_mode']!r}")
    if (metadata["calibration_mode"] == "approximate_image_extent" and
            (metadata["assumed_world_width_m"] is None or
             metadata["assumed_world_width_m"] <= 0.0)):
        raise ValueError(
            "approximate_image_extent requires a positive assumed_world_width_m")
    return control_path, points, image_points, world_points, metadata


def _convex_hull_area(points: np.ndarray) -> float:
    hull = cv2.convexHull(points.astype(np.float32))
    return float(abs(cv2.contourArea(hull)))


def _project(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    projected = cv2.perspectiveTransform(
        points.astype(np.float64).reshape(-1, 1, 2), homography)
    return projected.reshape(-1, 2)


def _jacobian(homography: np.ndarray, u: float, v: float) -> np.ndarray:
    h = homography
    denominator = h[2, 0] * u + h[2, 1] * v + h[2, 2]
    x_numerator = h[0, 0] * u + h[0, 1] * v + h[0, 2]
    z_numerator = h[1, 0] * u + h[1, 1] * v + h[1, 2]
    denominator_sq = denominator * denominator
    return np.array([
        [
            (h[0, 0] * denominator - x_numerator * h[2, 0]) / denominator_sq,
            (h[0, 1] * denominator - x_numerator * h[2, 1]) / denominator_sq,
        ],
        [
            (h[1, 0] * denominator - z_numerator * h[2, 0]) / denominator_sq,
            (h[1, 1] * denominator - z_numerator * h[2, 1]) / denominator_sq,
        ],
    ], dtype=np.float64)
