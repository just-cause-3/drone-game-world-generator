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

from .common import publish_directory


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
        for name, value in asdict(self).items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
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


def _transform_quality(homography: np.ndarray, width: int, height: int,
                       grid_size: int) -> dict:
    us = np.linspace(0.0, width - 1.0, grid_size)
    vs = np.linspace(0.0, height - 1.0, grid_size)
    denominators = []
    scales = []
    anisotropies = []
    handedness = []
    for v in vs:
        for u in us:
            denominator = float(
                homography[2, 0] * u +
                homography[2, 1] * v +
                homography[2, 2])
            denominators.append(denominator)
            if abs(denominator) < 1e-10:
                continue
            jacobian = _jacobian(homography, float(u), float(v))
            singular_values = np.linalg.svd(jacobian, compute_uv=False)
            if singular_values[-1] <= 1e-12:
                anisotropy = math.inf
            else:
                anisotropy = float(singular_values[0] / singular_values[-1])
            scales.append(float(math.sqrt(abs(np.linalg.det(jacobian)))))
            anisotropies.append(anisotropy)
            handedness.append(float(np.linalg.det(jacobian)))

    finite = (
        all(math.isfinite(value) for value in denominators + scales + anisotropies) and
        bool(scales))
    same_denominator_sign = (
        min(denominators) > 1e-10 or max(denominators) < -1e-10)
    same_handedness = (
        min(handedness, default=0.0) > 0.0 or
        max(handedness, default=0.0) < 0.0)
    return {
        "finite_across_image": finite,
        "projective_denominator_has_constant_sign": same_denominator_sign,
        "jacobian_handedness_has_constant_sign": same_handedness,
        "handedness": (
            "preserved" if float(np.median(handedness)) > 0.0 else "reflected"),
        "meters_per_pixel_min": round(min(scales, default=0.0), 9),
        "meters_per_pixel_median": round(float(np.median(scales)) if scales else 0.0, 9),
        "meters_per_pixel_max": round(max(scales, default=0.0), 9),
        "max_local_anisotropy": round(max(anisotropies, default=math.inf), 6),
    }


def _draw_overlay(ground_path: Path, points: list[dict],
                  errors: np.ndarray, destination: Path) -> None:
    image = cv2.imread(str(ground_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not render calibration overlay from {ground_path}")
    radius = max(5, round(min(image.shape[:2]) / 120))
    for point, error in zip(points, errors):
        u = round(point["image"]["u"])
        v = round(point["image"]["v"])
        label = (
            f"{point['id']} ({point['world']['x']:.2f},"
            f" {point['world']['z']:.2f}) e={float(error):.3f}m")
        cv2.circle(image, (u, v), radius + 3, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(image, (u, v), radius, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(
            image, label, (u + radius + 5, max(18, v - radius - 3)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(
            image, label, (u + radius + 5, max(18, v - radius - 3)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(destination), image):
        raise IOError(f"could not write calibration overlay: {destination}")


def _publish_directory(temporary: Path, output: Path) -> None:
    if output.exists():
        marker = output / OUTPUT_MARKER
        if not output.is_dir():
            raise ValueError(
                f"refusing to replace unrelated output directory: {output}")
        has_contents = any(output.iterdir())
        if has_contents and not marker.is_file():
            raise ValueError(
                f"refusing to replace unrelated output directory: {output}")
    publish_directory(temporary, output)


def run_stage2(stage1_bundle, control_points_path, output_directory="stage2_out",
               options: Stage2Options | None = None) -> dict:
    options = options or Stage2Options()
    options.validate()
    stage1_root, stage1_report, ground_path = _load_stage1_bundle(stage1_bundle)
    width = int(stage1_report["input"]["width"])
    height = int(stage1_report["input"]["height"])
    (control_path, points, image_points,
     world_points, metadata) = _load_control_points(
        control_points_path, width, height, options.min_points)

    image_hull_area_pct = (
        100.0 * _convex_hull_area(image_points) / float(width * height))
    world_hull_area_m2 = _convex_hull_area(world_points)
    homography, _ = cv2.findHomography(image_points, world_points, method=0)
    if homography is None or not np.all(np.isfinite(homography)):
        raise ValueError("control points do not define a finite pixel-to-world transform")
    if abs(float(homography[2, 2])) <= 1e-12:
        raise ValueError("calibration homography cannot be normalized")
    homography = homography / homography[2, 2]

    projected = _project(homography, image_points)
    error_vectors = projected - world_points
    errors = np.linalg.norm(error_vectors, axis=1)
    rmse = float(math.sqrt(np.mean(np.square(errors))))
    max_error = float(np.max(errors))
    quality = _transform_quality(
        homography, width, height, options.validation_grid_size)

    corners_pixels = np.array([
        [0.0, 0.0],
        [width - 1.0, 0.0],
        [width - 1.0, height - 1.0],
        [0.0, height - 1.0],
    ], dtype=np.float64)
    corners_world = _project(homography, corners_pixels)

    failure_reasons = []
    if image_hull_area_pct < options.min_image_hull_area_pct:
        failure_reasons.append(
            f"control points cover {image_hull_area_pct:.2f}% of the image; "
            f"minimum is {options.min_image_hull_area_pct:.2f}%")
    if world_hull_area_m2 < options.min_world_hull_area_m2:
        failure_reasons.append(
            f"control points cover only {world_hull_area_m2:.3f} square metres")
    if rmse > options.max_reprojection_rmse_m:
        failure_reasons.append(
            f"reprojection RMSE {rmse:.3f} m exceeds "
            f"{options.max_reprojection_rmse_m:.3f} m")
    if max_error > options.max_reprojection_error_m:
        failure_reasons.append(
            f"maximum control-point error {max_error:.3f} m exceeds "
            f"{options.max_reprojection_error_m:.3f} m")
    if not quality["finite_across_image"]:
        failure_reasons.append("pixel-to-world transform is not finite across the image")
    if not quality["projective_denominator_has_constant_sign"]:
        failure_reasons.append("pixel-to-world transform crosses infinity inside the image")
    if not quality["jacobian_handedness_has_constant_sign"]:
        failure_reasons.append("pixel-to-world transform folds over inside the image")
    if quality["max_local_anisotropy"] > options.max_local_anisotropy:
        failure_reasons.append(
            f"local scale anisotropy {quality['max_local_anisotropy']:.3f} exceeds "
            f"{options.max_local_anisotropy:.3f}")

    approximate = metadata["calibration_mode"] == "approximate_image_extent"
    confidence = (
        "approximate_assumed_scale" if approximate
        else "minimum_unverified" if len(points) == 4
        else "measured" if len(points) < options.recommended_points
        else "well_constrained")
    warnings = []
    if approximate:
        warnings.append(
            "World scale is an operator-selected approximation, not a physical "
            "measurement. Distances, object sizes, and speeds are not metric evidence.")
    elif len(points) == 4:
        warnings.append(
            "Four points solve a homography exactly, so reprojection error is not an "
            "independent accuracy check; add at least two more surveyed points.")
    elif len(points) < options.recommended_points:
        warnings.append(
            f"Use at least {options.recommended_points} points for stronger validation.")

    point_results = []
    for point, projected_point, error in zip(points, projected, errors):
        item = dict(point)
        item["projected_world"] = {
            "x": round(float(projected_point[0]), 9),
            "z": round(float(projected_point[1]), 9),
        }
        item["error_m"] = round(float(error), 9)
        point_results.append(item)

    calibration = {
        "schema_version": CALIBRATION_SCHEMA,
        "stage": 2,
        "pass": not failure_reasons,
        "failure_reasons": failure_reasons,
        "warnings": warnings,
        "confidence": confidence,
        "source_stage1": {
            "bundle_path": str(stage1_root),
            "report_sha256": _sha256(stage1_root / "report.json"),
            "ground_sha256": _sha256(ground_path),
            "source_video_sha256": stage1_report["input"].get("sha256"),
            "reference_video_frame": stage1_report["reference"].get("video_frame"),
            "pixel_width": width,
            "pixel_height": height,
        },
        "coordinate_system": {
            "input_space": "stage1_reference_image_pixels",
            "input_origin": "top_left",
            "input_u_axis": "right",
            "input_v_axis": "down",
            "output_space": "unity_world_xz",
            "units": "meters",
            "calibration_mode": metadata["calibration_mode"],
            "metric_scale_is_assumed": approximate,
            "assumed_world_width_m": metadata["assumed_world_width_m"],
            "ground_y_m": metadata["ground_y_m"],
            "origin_description": metadata["origin_description"],
            "x_axis_description": metadata["x_axis_description"],
            "z_axis_description": metadata["z_axis_description"],
        },
        "control_points": point_results,
        "quality": {
            "point_count": len(points),
            "independent_residual_degrees_of_freedom": max(0, 2 * len(points) - 8),
            "image_convex_hull_area_pct": round(image_hull_area_pct, 6),
            "world_convex_hull_area_m2": round(world_hull_area_m2, 6),
            "reprojection_rmse_m": round(rmse, 9),
            "reprojection_max_error_m": round(max_error, 9),
            **quality,
        },
        "reference_pixel_to_world_xz": [
            [round(float(value), 12) for value in row] for row in homography
        ],
        "reference_pixel_to_world_xz_flat": [
            round(float(value), 12) for value in homography.reshape(-1)
        ],
        "image_corners_world_xz": [
            {
                "u": float(pixel[0]),
                "v": float(pixel[1]),
                "x": round(float(world[0]), 9),
                "z": round(float(world[1]), 9),
            }
            for pixel, world in zip(corners_pixels, corners_world)
        ],
        "world_bounds": {
            "min_x": round(float(np.min(corners_world[:, 0])), 9),
            "max_x": round(float(np.max(corners_world[:, 0])), 9),
            "min_z": round(float(np.min(corners_world[:, 1])), 9),
            "max_z": round(float(np.max(corners_world[:, 1])), 9),
        },
        "options": asdict(options),
        "outputs": {
            "calibration": "world_calibration.json",
            "overlay": "calibration_overlay.png",
            "control_points": "control_points_used.json",
        },
    }

    output = Path(output_directory).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{output.name}-", dir=str(output.parent)))
    try:
        (temporary / OUTPUT_MARKER).write_text(
            CALIBRATION_SCHEMA + "\n", encoding="utf-8")
        (temporary / "world_calibration.json").write_text(
            json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
        normalized_input = {
            "schema_version": CONTROL_POINTS_SCHEMA,
            "units": "meters",
            **metadata,
            "points": points,
            "source_path": str(control_path),
        }
        (temporary / "control_points_used.json").write_text(
            json.dumps(normalized_input, indent=2) + "\n", encoding="utf-8")
        _draw_overlay(
            ground_path, points, errors, temporary / "calibration_overlay.png")
        _publish_directory(temporary, output)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return calibration
