"""Stage 4A: propose and validate static semantic labels for operator review."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Protocol

import cv2
import numpy as np

from .common import log
from .stage1_ground import _tile_origins
from .track_agents import resolve_device
from setup_stage4_model import (
    AERIAL_CLASSES,
    AERIAL_DIRECTORY,
    AERIAL_FILES,
    AERIAL_REVISION,
    EXPECTED_CLASSES,
    MODEL_SHA256,
    MODEL_SIZE,
    verify_aerial_model,
    verify_legacy_model,
)


STAGE1_SCHEMA = "stage1-ground-v1"
STAGE2_SCHEMA = "stage2-calibration-v1"
STAGE4_SCHEMA = "stage4-semantics-v1"
OUTPUT_MARKER = ".stage4-semantics-output"
DEFAULT_SEMANTIC_BACKEND = "hybrid-openearthmap-road"
DEFAULT_SEMANTIC_MODEL = str(AERIAL_DIRECTORY).replace("\\", "/")
LEGACY_SEMANTIC_MODEL = "models/yolo26x-sem.pt"

CANONICAL_CLASSES = (
    "unknown", "road", "pavement", "building", "vegetation", "terrain",
    "water", "other_static",
)
CANONICAL_IDS = {name: index for index, name in enumerate(CANONICAL_CLASSES)}
CANONICAL_COLORS_BGR = (
    (180, 0, 180),    # unknown
    (70, 70, 70),     # road
    (210, 180, 140),  # pavement
    (40, 80, 220),    # building
    (50, 170, 50),    # vegetation
    (60, 130, 170),   # terrain
    (220, 140, 30),   # water
    (30, 200, 220),   # other static
)
BOOTSTRAP_CLASS_TO_CANONICAL = {
    "road": "road",
    "sidewalk": "pavement",
    "building": "building",
    "wall": "other_static",
    "fence": "other_static",
    "pole": "other_static",
    "traffic light": "other_static",
    "traffic sign": "other_static",
    "vegetation": "vegetation",
    "terrain": "terrain",
    # Sky and dynamic classes deliberately remain unknown.
}
OPENEARTHMAP_CLASS_TO_CANONICAL = {
    "bareland": "terrain",
    "rangeland": "terrain",
    "developed_space": "pavement",
    "road": "road",
    "tree": "vegetation",
    "water": "water",
    "agriculture": "terrain",
    "building": "building",
}


@dataclass(frozen=True)
class Stage4Options:
    semantic_backend: str = DEFAULT_SEMANTIC_BACKEND
    semantic_model: str = DEFAULT_SEMANTIC_MODEL
    road_specialist_model: str = LEGACY_SEMANTIC_MODEL
    inference_imgsz: int = 1280
    tile_size: int = 640
    tile_overlap: float = 0.25
    horizontal_flip_tta: bool = True
    minimum_component_area_px: int = 24
    maximum_unknown_valid_pct: float = 35.0
    building_area_warning_pct: float = 55.0
    other_static_warning_pct: float = 20.0
    fragmentation_warning_components_per_megapixel: float = 400.0
    overlay_alpha: float = 0.48

    def validate(self) -> None:
        if self.semantic_backend not in (
                "hybrid-openearthmap-road", "openearthmap-mask2former",
                "legacy-cityscapes-yolo"):
            raise ValueError(
                "semantic_backend must be hybrid-openearthmap-road, "
                "openearthmap-mask2former, or legacy-cityscapes-yolo")
        if self.inference_imgsz < 320:
            raise ValueError("Stage 4 inference_imgsz must be at least 320")
        if self.tile_size != 0 and self.tile_size < 128:
            raise ValueError("Stage 4 tile_size must be zero or at least 128")
        if not 0.0 <= self.tile_overlap < 0.9:
            raise ValueError("Stage 4 tile_overlap must be in [0, 0.9)")
        if self.minimum_component_area_px < 0:
            raise ValueError("minimum_component_area_px cannot be negative")
        if not 0.0 <= self.maximum_unknown_valid_pct <= 100.0:
            raise ValueError("maximum_unknown_valid_pct must be in [0, 100]")
        if not 0.0 <= self.building_area_warning_pct <= 100.0:
            raise ValueError("building_area_warning_pct must be in [0, 100]")
        if not 0.0 <= self.other_static_warning_pct <= 100.0:
            raise ValueError("other_static_warning_pct must be in [0, 100]")
        if self.fragmentation_warning_components_per_megapixel <= 0.0:
            raise ValueError("fragmentation warning threshold must be positive")
        if not 0.0 <= self.overlay_alpha <= 1.0:
            raise ValueError("overlay_alpha must be in [0, 1]")


class SemanticProvider(Protocol):
    @property
    def metadata(self) -> dict: ...

    def predict(self, image: np.ndarray) -> np.ndarray: ...


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


def _read_bundle_image(path: Path, flags: int, expected_shape: tuple[int, int],
                       label: str) -> np.ndarray:
    """Decode a small pipeline image robustly after long GPU-heavy stages.

    Reading bytes first avoids intermittent Windows/OpenCV filename decoder failures.
    Retrying after collection also handles temporary allocation pressure left by an
    earlier detector without accepting a wrong-sized or corrupt artifact.
    """
    last_shape = None
    last_error = None
    for _ in range(3):
        try:
            encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
            image = cv2.imdecode(encoded, flags) if encoded.size else None
        except (OSError, cv2.error) as exception:
            image = None
            last_error = str(exception)
        if image is not None:
            last_shape = image.shape[:2]
            if last_shape == expected_shape:
                return image
        gc.collect()
    detail = (
        f"decoded shape was {last_shape}" if last_shape is not None
        else f"decoder returned no image{': ' + last_error if last_error else ''}")
    raise ValueError(
        f"Stage 1 {label} is unreadable or dimensionally inconsistent; "
        f"expected {expected_shape[1]}x{expected_shape[0]}, {detail}")


def _load_inputs(stage1_bundle, stage2_bundle) -> dict:
    stage1_root = Path(stage1_bundle).resolve()
    stage1_report_path = stage1_root / "report.json"
    if not stage1_report_path.is_file():
        raise FileNotFoundError(f"Stage 1 report not found: {stage1_report_path}")
    stage1 = _read_json(stage1_report_path)
    if stage1.get("schema_version") != STAGE1_SCHEMA or stage1.get("stage") != 1:
        raise ValueError("the selected directory is not a Stage 1 ground bundle")
    if stage1.get("pass") is not True:
        raise ValueError("Stage 4 refuses a failed Stage 1 bundle")
    try:
        width = int(stage1["input"]["width"])
        height = int(stage1["input"]["height"])
        outputs1 = stage1["outputs"]
    except (KeyError, TypeError, ValueError) as exception:
        raise ValueError("Stage 1 report is missing dimensions or outputs") from exception
    if width <= 0 or height <= 0:
        raise ValueError("Stage 1 dimensions are invalid")
    ground_path = _bundle_path(stage1_root, outputs1.get("ground"), "ground image")
    valid_path = _bundle_path(stage1_root, outputs1.get("valid_mask"), "valid mask")
    expected_shape = (height, width)
    ground = _read_bundle_image(
        ground_path, cv2.IMREAD_COLOR, expected_shape, "ground")
    valid = _read_bundle_image(
        valid_path, cv2.IMREAD_GRAYSCALE, expected_shape, "valid mask")
    valid = valid > 0
    if not np.any(valid):
        raise ValueError("Stage 1 valid mask contains no supported pixels")

    stage2_root = Path(stage2_bundle).resolve()
    calibration_path = stage2_root / "world_calibration.json"
    if not calibration_path.is_file():
        raise FileNotFoundError(f"Stage 2 calibration not found: {calibration_path}")
    stage2 = _read_json(calibration_path)
    if stage2.get("schema_version") != STAGE2_SCHEMA or stage2.get("stage") != 2:
        raise ValueError("the selected directory is not a Stage 2 calibration bundle")
    if stage2.get("pass") is not True:
        raise ValueError("Stage 4 refuses a failed Stage 2 calibration")
    try:
        source2 = stage2["source_stage1"]
        matrix = np.asarray(stage2["reference_pixel_to_world_xz"], dtype=np.float64)
        scale_confidence = str(stage2["confidence"])
        metric_scale_is_assumed = bool(
            stage2["coordinate_system"].get("metric_scale_is_assumed", False))
    except (KeyError, TypeError, ValueError) as exception:
        raise ValueError("Stage 2 calibration is missing required data") from exception
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("Stage 2 homography is not a finite 3x3 matrix")
    if abs(float(np.linalg.det(matrix))) < 1e-12:
        raise ValueError("Stage 2 homography is singular")
    if source2.get("report_sha256") != _sha256(stage1_report_path):
        raise ValueError("Stage 2 calibration is stale: Stage 1 report hash changed")
    if source2.get("ground_sha256") != _sha256(ground_path):
        raise ValueError("Stage 2 calibration is stale: Stage 1 ground hash changed")
    if source2.get("source_video_sha256") != stage1["input"].get("sha256"):
        raise ValueError("Stage 2 calibration belongs to another source video")
    if (int(source2.get("pixel_width", -1)) != width or
            int(source2.get("pixel_height", -1)) != height):
        raise ValueError("Stage 2 dimensions do not match Stage 1")

    return {
        "stage1_root": stage1_root,
        "stage1": stage1,
        "stage1_report_path": stage1_report_path,
        "ground_path": ground_path,
        "valid_path": valid_path,
        "ground": ground,
        "valid": valid,
        "stage2_root": stage2_root,
        "stage2": stage2,
        "calibration_path": calibration_path,
        "pixel_to_world": matrix,
        "width": width,
        "height": height,
        "scale_confidence": scale_confidence,
        "metric_scale_is_assumed": metric_scale_is_assumed,
    }
