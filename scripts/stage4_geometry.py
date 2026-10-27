"""Stage 4B/C: convert an accepted semantic map into provisional 3D geometry data.

The generator is intentionally fail-closed.  It will not vectorize a Stage 4A
proposal until an operator acceptance record matches the exact report and label
image hashes.  Horizontal coordinates come from Stage 2.  With stationary nadir
footage, building heights remain explicit assumptions rather than measurements.
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

from .common import log, publish_directory
from .stage4_semantics import CANONICAL_CLASSES, CANONICAL_COLORS_BGR, CANONICAL_IDS


STAGE4A_SCHEMA = "stage4-semantics-v1"
ACCEPTANCE_SCHEMA = "stage4-semantic-acceptance-v1"
GEOMETRY_SCHEMA = "stage4-geometry-v1"
OUTPUT_MARKER = ".stage4-geometry-output"


@dataclass(frozen=True)
class Stage4GeometryOptions:
    minimum_building_area_m2: float = 6.0
    minimum_surface_area_m2: float = 2.0
    simplify_tolerance_m: float = 0.25
    maximum_polygon_vertices: int = 96
    maximum_buildings: int = 500
    height_policy: str = "fixed"
    default_building_height_m: float = 8.0
    floor_height_m: float = 3.2
    maximum_assumed_height_m: float = 24.0
    generate_vegetation_proxies: bool = False
    tree_spacing_m: float = 4.0
    minimum_tree_radius_m: float = 0.8
    maximum_tree_radius_m: float = 4.0
    maximum_trees: int = 120

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.minimum_building_area_m2 <= 0.0:
            raise ValueError("minimum_building_area_m2 must be positive")
        if self.minimum_surface_area_m2 <= 0.0:
            raise ValueError("minimum_surface_area_m2 must be positive")
        if self.simplify_tolerance_m <= 0.0:
            raise ValueError("simplify_tolerance_m must be positive")
        if self.maximum_polygon_vertices < 3:
            raise ValueError("maximum_polygon_vertices must be at least three")
        if self.maximum_buildings < 1:
            raise ValueError("maximum_buildings must be positive")
        if self.height_policy not in {"fixed", "footprint_prior"}:
            raise ValueError("height_policy must be 'fixed' or 'footprint_prior'")
        if self.default_building_height_m <= 0.0:
            raise ValueError("default_building_height_m must be positive")
        if self.floor_height_m <= 0.0:
            raise ValueError("floor_height_m must be positive")
        if self.maximum_assumed_height_m < self.floor_height_m:
            raise ValueError("maximum_assumed_height_m must be at least one floor")
        if self.tree_spacing_m <= 0.0:
            raise ValueError("tree_spacing_m must be positive")
        if self.minimum_tree_radius_m <= 0.0:
            raise ValueError("minimum_tree_radius_m must be positive")
        if self.maximum_tree_radius_m < self.minimum_tree_radius_m:
            raise ValueError(
                "maximum_tree_radius_m must be at least minimum_tree_radius_m")
        if self.maximum_trees < 0:
            raise ValueError("maximum_trees cannot be negative")


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


def _resolve_artifact(root: Path, relative_path, label: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise ValueError(f"{label} path is missing")
    candidate = Path(relative_path)
    if candidate.is_absolute():
        raise ValueError(f"{label} path must be relative to its bundle")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exception:
        raise ValueError(f"{label} path escapes its bundle") from exception
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is missing: {resolved}")
    return resolved


def _required_dict(value, label: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label} is missing or invalid")
    return value


def _load_inputs(stage4_bundle) -> dict:
    stage4_root = Path(stage4_bundle).resolve()
    report_path = stage4_root / "report.json"
    acceptance_path = stage4_root / "semantic_acceptance.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"Stage 4A report not found: {report_path}")
    if not acceptance_path.is_file():
        raise FileNotFoundError(
            "Stage 4B is blocked until Unity writes semantic_acceptance.json")

    report = _read_json(report_path)
    if (report.get("schema_version") != STAGE4A_SCHEMA or
            report.get("stage") != 4 or
            report.get("substage") != "semantic_proposal"):
        raise ValueError("the selected directory is not a Stage 4A semantic bundle")
    if report.get("pass") is not True:
        raise ValueError("Stage 4B refuses a failed Stage 4A proposal")
    if report.get("requires_human_review") is not True:
        raise ValueError("Stage 4A report does not preserve the human-review gate")

    outputs = _required_dict(report.get("outputs"), "Stage 4A outputs")
    hashes = _required_dict(report.get("artifact_hashes"), "Stage 4A hashes")
    labels_path = _resolve_artifact(
        stage4_root, outputs.get("semantic_labels"), "semantic labels")
    labels_hash = _sha256(labels_path)
    if labels_hash.lower() != str(hashes.get("semantic_labels_sha256", "")).lower():
        raise ValueError("Stage 4A semantic label hash does not match its report")

    acceptance = _read_json(acceptance_path)
    if (acceptance.get("schema_version") != ACCEPTANCE_SCHEMA or
            acceptance.get("stage") != 4):
        raise ValueError("semantic acceptance record has the wrong schema")
    if acceptance.get("accepted") is not True:
        raise ValueError("Stage 4A semantic acceptance was revoked")
    report_hash = _sha256(report_path)
    if str(acceptance.get("stage4_report_sha256", "")).lower() != report_hash:
        raise ValueError("semantic acceptance is stale: Stage 4A report changed")
    if str(acceptance.get("semantic_labels_sha256", "")).lower() != labels_hash:
        raise ValueError("semantic acceptance is stale: label image changed")

    source1 = _required_dict(report.get("source_stage1"), "Stage 4A Stage 1 source")
    source2 = _required_dict(report.get("source_stage2"), "Stage 4A Stage 2 source")
    stage1_root = Path(str(source1.get("bundle_path", ""))).resolve()
    stage2_root = Path(str(source2.get("bundle_path", ""))).resolve()
    stage1_report_path = stage1_root / "report.json"
    calibration_path = stage2_root / "world_calibration.json"
    if not stage1_report_path.is_file() or not calibration_path.is_file():
        raise FileNotFoundError("an upstream Stage 1 or Stage 2 bundle is missing")
    if _sha256(stage1_report_path).lower() != str(
            source1.get("report_sha256", "")).lower():
        raise ValueError("Stage 1 report changed after Stage 4A")
    if _sha256(calibration_path).lower() != str(
            source2.get("calibration_sha256", "")).lower():
        raise ValueError("Stage 2 calibration changed after Stage 4A")

    stage1 = _read_json(stage1_report_path)
    calibration = _read_json(calibration_path)
    if (stage1.get("schema_version") != "stage1-ground-v1" or
            stage1.get("pass") is not True):
        raise ValueError("Stage 4B requires a passing Stage 1 bundle")
    if (calibration.get("schema_version") != "stage2-calibration-v1" or
            calibration.get("pass") is not True):
        raise ValueError("Stage 4B requires a passing Stage 2 calibration")
    try:
        width = int(source1["pixel_width"])
        height = int(source1["pixel_height"])
        homography = np.asarray(
            calibration["reference_pixel_to_world_xz"], dtype=np.float64)
        ground_relative = stage1["outputs"]["ground"]
    except (KeyError, TypeError, ValueError) as exception:
        raise ValueError("upstream geometry provenance is incomplete") from exception
    if width < 2 or height < 2 or homography.shape != (3, 3):
        raise ValueError("upstream dimensions or homography are invalid")
    if not np.all(np.isfinite(homography)):
        raise ValueError("Stage 2 homography contains non-finite values")

    labels = cv2.imread(str(labels_path), cv2.IMREAD_UNCHANGED)
    if labels is None or labels.ndim != 2 or labels.shape != (height, width):
        raise ValueError("semantic labels are unreadable or dimensionally inconsistent")
    if int(labels.min()) < 0 or int(labels.max()) >= len(CANONICAL_CLASSES):
        raise ValueError("semantic labels contain an unsupported class ID")

    ground_path = _resolve_artifact(stage1_root, ground_relative, "Stage 1 ground")
    if _sha256(ground_path).lower() != str(source1.get("ground_sha256", "")).lower():
        raise ValueError("Stage 1 ground changed after Stage 4A")
    ground = cv2.imread(str(ground_path), cv2.IMREAD_COLOR)
    if ground is None or ground.shape[:2] != (height, width):
        raise ValueError("Stage 1 ground is unreadable or dimensionally inconsistent")

    coordinate = _required_dict(
        calibration.get("coordinate_system"), "Stage 2 coordinate system")
    return {
        "stage4_root": stage4_root,
        "stage4_report": report,
        "stage4_report_path": report_path,
        "acceptance_path": acceptance_path,
        "labels_path": labels_path,
        "labels": labels.astype(np.uint8, copy=False),
        "ground": ground,
        "stage1_root": stage1_root,
        "stage1_report_path": stage1_report_path,
        "stage2_root": stage2_root,
        "calibration_path": calibration_path,
        "calibration": calibration,
        "homography": homography,
        "width": width,
        "height": height,
        "ground_y": float(coordinate.get("ground_y_m", 0.0)),
        "metric_scale_is_assumed": bool(
            coordinate.get("metric_scale_is_assumed", False)),
        "scale_confidence": str(calibration.get("confidence", "unknown")),
        "report_hash": report_hash,
        "labels_hash": labels_hash,
        "acceptance_hash": _sha256(acceptance_path),
    }


def _project_points(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    source = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(source, homography).reshape(-1, 2)
    if not np.all(np.isfinite(projected)):
        raise ValueError("a semantic boundary maps to non-finite world coordinates")
    return projected


def _median_metres_per_pixel(homography: np.ndarray, width: int, height: int) -> float:
    samples = []
    for v in np.linspace(0.0, height - 2.0, 5):
        for u in np.linspace(0.0, width - 2.0, 5):
            world = _project_points(homography, np.array([
                [u, v], [u + 1.0, v], [u, v + 1.0]], dtype=np.float64))
            samples.extend((
                float(np.linalg.norm(world[1] - world[0])),
                float(np.linalg.norm(world[2] - world[0])),
            ))
    finite = [value for value in samples if math.isfinite(value) and value > 1e-9]
    if not finite:
        raise ValueError("Stage 2 mapping has no usable local scale")
    return float(np.median(finite))


def _signed_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _point_in_triangle(point, a, b, c, epsilon=1e-9) -> bool:
    def cross(p, q, r):
        return ((q[0] - p[0]) * (r[1] - p[1]) -
                (q[1] - p[1]) * (r[0] - p[0]))
    c1 = cross(a, b, point)
    c2 = cross(b, c, point)
    c3 = cross(c, a, point)
    return ((c1 >= -epsilon and c2 >= -epsilon and c3 >= -epsilon) or
            (c1 <= epsilon and c2 <= epsilon and c3 <= epsilon))


def _triangulate_polygon(points: np.ndarray) -> list[int]:
    """Ear-clip one simple polygon and return flattened vertex indices."""
    count = len(points)
    if count < 3 or abs(_signed_area(points)) < 1e-8:
        raise ValueError("polygon has fewer than three usable vertices")
    orientation = 1.0 if _signed_area(points) > 0.0 else -1.0
    remaining = list(range(count))
    triangles: list[int] = []
    guard = 0
    while len(remaining) > 3:
        ear_found = False
        for offset, current in enumerate(remaining):
            previous = remaining[offset - 1]
            following = remaining[(offset + 1) % len(remaining)]
            a, b, c = points[previous], points[current], points[following]
            first = b - a
            second = c - b
            cross = float(first[0] * second[1] - first[1] * second[0])
            if orientation * cross <= 1e-10:
                continue
            if any(
                    _point_in_triangle(points[index], a, b, c)
                    for index in remaining
                    if index not in {previous, current, following}):
                continue
            triangles.extend((previous, current, following))
            del remaining[offset]
            ear_found = True
            break
        guard += 1
        if not ear_found or guard > count * count:
            raise ValueError("polygon triangulation failed; boundary may be self-intersecting")
    triangles.extend(remaining)
    return triangles


def _simplify_contour(contour: np.ndarray, tolerance_px: float,
                      maximum_vertices: int) -> np.ndarray:
    epsilon = max(0.5, tolerance_px)
    simplified = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    for _ in range(12):
        if len(simplified) <= maximum_vertices:
            break
        epsilon *= 1.4
        simplified = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    if len(simplified) < 3 or len(simplified) > maximum_vertices:
        raise ValueError(
            f"could not simplify polygon to 3..{maximum_vertices} vertices")
    return simplified.astype(np.float64)


def _assumed_height(area_m2: float, options: Stage4GeometryOptions) -> tuple[float, str]:
    if options.height_policy == "fixed":
        height = options.default_building_height_m
        source = "operator_configured_fixed_assumption"
    else:
        floors = max(1, int(round(math.sqrt(max(area_m2, 1.0)) / 7.0)))
        height = floors * options.floor_height_m
        source = "footprint_area_prior_assumption"
    return min(height, options.maximum_assumed_height_m), source
