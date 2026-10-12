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


class ExternalLabelProvider:
    def __init__(self, path):
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"external semantic label image not found: {self.path}")
        self._metadata = {
            "implementation": "external-canonical-labels-v1",
            "source": str(self.path),
            "source_sha256": _sha256(self.path),
            "domain": "operator_supplied_or_corrected",
            "automatic": False,
        }

    @property
    def metadata(self) -> dict:
        return dict(self._metadata)

    def predict(self, image: np.ndarray) -> np.ndarray:
        labels = cv2.imread(str(self.path), cv2.IMREAD_UNCHANGED)
        if labels is None:
            raise ValueError(f"external semantic label image is unreadable: {self.path}")
        if labels.ndim != 2:
            raise ValueError("external semantic labels must be a one-channel PNG")
        if labels.shape != image.shape[:2]:
            raise ValueError(
                f"external labels are {labels.shape[1]}x{labels.shape[0]}; expected "
                f"{image.shape[1]}x{image.shape[0]}")
        return labels.astype(np.uint8, copy=False)


class OpenEarthMapMask2FormerProvider:
    """Dense aerial land-cover proposal from a pinned OpenEarthMap checkpoint."""

    def __init__(self, options: Stage4Options):
        import torch
        from transformers import (
            AutoImageProcessor,
            Mask2FormerForUniversalSegmentation,
        )

        self.options = options
        self.path = Path(options.semantic_model).resolve()
        verify_aerial_model(self.path)
        self.torch = torch
        self.processor = AutoImageProcessor.from_pretrained(
            str(self.path), local_files_only=True, use_fast=False)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            str(self.path), local_files_only=True)
        if self.model.config.num_labels != len(AERIAL_CLASSES):
            raise ValueError(
                "OpenEarthMap checkpoint does not expose the pinned eight-class head")
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.pass_count = 0
        weights = self.path / "model.safetensors"
        self._metadata = {
            "implementation": "openearthmap-mask2former-swin-consensus-v1",
            "model": str(self.path),
            "model_revision": AERIAL_REVISION,
            "model_size": weights.stat().st_size,
            "model_sha256": _sha256(weights),
            "expected_model_sha256": AERIAL_FILES["model.safetensors"][1],
            "device": str(self.device),
            "domain": "openearthmap_aerial_satellite_0.25_0.5m_gsd",
            "automatic": True,
            "raw_classes": list(AERIAL_CLASSES),
            "class_to_canonical": dict(OPENEARTHMAP_CLASS_TO_CANONICAL),
            "reported_validation_miou": 0.5202420374155045,
        }

    @property
    def metadata(self) -> dict:
        result = dict(self._metadata)
        result["inference_passes"] = self.pass_count
        return result

    def _predict_one(self, image: np.ndarray) -> np.ndarray:
        from PIL import Image

        height, width = image.shape[:2]
        side = max(height, width)
        top = (side - height) // 2
        bottom = side - height - top
        left = (side - width) // 2
        right = side - width - left
        square = cv2.copyMakeBorder(
            image, top, bottom, left, right, cv2.BORDER_REFLECT_101)
        rgb = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
        tensors = self.processor(
            images=Image.fromarray(rgb), return_tensors="pt")
        tensors = {name: value.to(self.device)
                   for name, value in tensors.items()}
        with self.torch.inference_mode():
            output = self.model(**tensors)
        raw = self.processor.post_process_semantic_segmentation(
            output, target_sizes=[(side, side)])[0]
        raw = raw.detach().cpu().numpy()[top:top + height, left:left + width]
        if raw.shape != (height, width):
            raise ValueError(
                f"OpenEarthMap model returned invalid mask shape {raw.shape}")
        mapped = np.zeros((height, width), dtype=np.uint8)
        for raw_id in np.unique(raw):
            raw_index = int(raw_id)
            if raw_index < 0 or raw_index >= len(AERIAL_CLASSES):
                raise ValueError(
                    f"OpenEarthMap model returned unknown class ID {raw_index}")
            raw_name = AERIAL_CLASSES[raw_index]
            mapped[raw == raw_index] = CANONICAL_IDS[
                OPENEARTHMAP_CLASS_TO_CANONICAL[raw_name]]
        self.pass_count += 1
        return mapped

    @staticmethod
    def _blend_weights(height: int, width: int) -> np.ndarray:
        y = np.maximum(np.hanning(max(3, height))[:height], 0.15)
        x = np.maximum(np.hanning(max(3, width))[:width], 0.15)
        return np.outer(y, x).astype(np.float32)

    def predict(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        scores = np.zeros(
            (height, width, len(CANONICAL_CLASSES)), dtype=np.float32)

        def add(labels: np.ndarray, y0: int, x0: int,
                weight: np.ndarray | float) -> None:
            region = scores[y0:y0 + labels.shape[0], x0:x0 + labels.shape[1]]
            for class_id in range(len(CANONICAL_CLASSES)):
                region[:, :, class_id] += weight * (labels == class_id)

        # The padded whole-image pass supplies context; overlapping square crops retain
        # the aerial model's training geometry and dominate local decisions.
        add(self._predict_one(image), 0, 0, 0.35)
        if self.options.horizontal_flip_tta:
            labels = self._predict_one(
                np.ascontiguousarray(image[:, ::-1]))[:, ::-1]
            add(labels, 0, 0, 0.35)

        tile = self.options.tile_size
        if tile > 0 and (tile < width or tile < height):
            for y0 in _tile_origins(height, tile, self.options.tile_overlap):
                for x0 in _tile_origins(width, tile, self.options.tile_overlap):
                    crop = image[y0:min(height, y0 + tile),
                                 x0:min(width, x0 + tile)]
                    weight = self._blend_weights(*crop.shape[:2])
                    add(self._predict_one(crop), y0, x0, weight)
                    if self.options.horizontal_flip_tta:
                        labels = self._predict_one(
                            np.ascontiguousarray(crop[:, ::-1]))[:, ::-1]
                        add(labels, y0, x0, weight)
        return np.argmax(scores, axis=2).astype(np.uint8)


class YoloSemanticProvider:
    def __init__(self, options: Stage4Options):
        from ultralytics import YOLO

        self.options = options
        self.path = Path(options.semantic_model).resolve()
        verify_legacy_model(self.path)
        self.model = YOLO(str(self.path))
        raw_names = self.model.names
        names = tuple(raw_names.values()) if isinstance(raw_names, dict) else tuple(raw_names)
        if names != EXPECTED_CLASSES or self.model.task != "semantic":
            raise ValueError("Stage 4 semantic checkpoint has an unexpected task/class map")
        self.names = {int(key): str(value) for key, value in raw_names.items()}
        self.device = resolve_device("auto")
        self.pass_count = 0
        self._metadata = {
            "implementation": "ultralytics-yolo26x-semantic-consensus-v1",
            "model": str(self.path),
            "model_size": self.path.stat().st_size,
            "model_sha256": _sha256(self.path),
            "expected_model_sha256": MODEL_SHA256,
            "expected_model_size": MODEL_SIZE,
            "device": str(self.device),
            "domain": "cityscapes_street_view_bootstrap",
            "automatic": True,
            "class_to_canonical": dict(BOOTSTRAP_CLASS_TO_CANONICAL),
        }

    @property
    def metadata(self) -> dict:
        result = dict(self._metadata)
        result["inference_passes"] = self.pass_count
        return result

    def _predict_one(self, image: np.ndarray) -> np.ndarray:
        result = self.model.predict(
            image,
            imgsz=self.options.inference_imgsz,
            device=self.device,
            verbose=False,
        )[0]
        semantic = getattr(result, "semantic_mask", None)
        if semantic is None or getattr(semantic, "data", None) is None:
            raise ValueError("semantic model returned no dense semantic mask")
        raw = semantic.data.detach().cpu().numpy()
        if raw.ndim != 2:
            raise ValueError(f"semantic model returned invalid mask shape {raw.shape}")
        if raw.shape != image.shape[:2]:
            raw = cv2.resize(
                raw.astype(np.uint8), (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST)
        mapped = np.zeros(raw.shape, dtype=np.uint8)
        for raw_id in np.unique(raw):
            name = self.names.get(int(raw_id), "")
            canonical = BOOTSTRAP_CLASS_TO_CANONICAL.get(name, "unknown")
            mapped[raw == raw_id] = CANONICAL_IDS[canonical]
        self.pass_count += 1
        return mapped

    @staticmethod
    def _blend_weights(height: int, width: int) -> np.ndarray:
        # A nonzero raised-cosine weight reduces tile seams without leaving edge holes.
        y = np.maximum(np.hanning(max(3, height))[:height], 0.15)
        x = np.maximum(np.hanning(max(3, width))[:width], 0.15)
        return np.outer(y, x).astype(np.float32)

    def predict(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        scores = np.zeros(
            (height, width, len(CANONICAL_CLASSES)), dtype=np.float32)

        def add(labels: np.ndarray, y0: int, x0: int, weight: np.ndarray) -> None:
            region = scores[y0:y0 + labels.shape[0], x0:x0 + labels.shape[1]]
            for class_id in range(len(CANONICAL_CLASSES)):
                region[:, :, class_id] += weight * (labels == class_id)

        full = self._predict_one(image)
        add(full, 0, 0, np.ones((height, width), dtype=np.float32))
        if self.options.horizontal_flip_tta:
            flipped = self._predict_one(np.ascontiguousarray(image[:, ::-1]))[:, ::-1]
            add(flipped, 0, 0, np.ones((height, width), dtype=np.float32))

        tile = self.options.tile_size
        if tile > 0 and (tile < width or tile < height):
            for y0 in _tile_origins(height, tile, self.options.tile_overlap):
                for x0 in _tile_origins(width, tile, self.options.tile_overlap):
                    crop = image[y0:min(height, y0 + tile), x0:min(width, x0 + tile)]
                    labels = self._predict_one(crop)
                    weight = self._blend_weights(*labels.shape)
                    add(labels, y0, x0, weight)
                    if self.options.horizontal_flip_tta:
                        flipped = self._predict_one(
                            np.ascontiguousarray(crop[:, ::-1]))[:, ::-1]
                        add(flipped, y0, x0, weight)
        return np.argmax(scores, axis=2).astype(np.uint8)


class HybridOpenEarthMapRoadProvider:
    """Aerial semantics with a narrowly scoped road-mask specialist.

    The Cityscapes model is never permitted to propose buildings or other static
    classes. It can only promote an OpenEarthMap pixel to road, which addresses the
    observed developed-space/road ambiguity without reintroducing its false buildings.
    """

    def __init__(self, options: Stage4Options):
        aerial_options = Stage4Options(
            **{**asdict(options),
               "semantic_backend": "openearthmap-mask2former"})
        road_options = Stage4Options(
            **{**asdict(options),
               "semantic_backend": "legacy-cityscapes-yolo",
               "semantic_model": options.road_specialist_model})
        self.aerial = OpenEarthMapMask2FormerProvider(aerial_options)
        self.road = YoloSemanticProvider(road_options)

    @property
    def metadata(self) -> dict:
        aerial = self.aerial.metadata
        road = self.road.metadata
        return {
            "implementation": "hybrid-openearthmap-mask2former-yolo-road-v1",
            "model": aerial["model"],
            "model_size": aerial["model_size"],
            "model_sha256": aerial["model_sha256"],
            "model_revision": aerial["model_revision"],
            "device": aerial["device"],
            "domain": "openearthmap_aerial_plus_cityscapes_road_only",
            "automatic": True,
            "class_to_canonical": aerial["class_to_canonical"],
            "aerial_provider": aerial,
            "road_specialist": {
                "implementation": road["implementation"],
                "model": road["model"],
                "model_sha256": road["model_sha256"],
                "domain": road["domain"],
                "allowed_output": "road_only",
                "inference_passes": road["inference_passes"],
            },
            "fusion_policy": (
                "OpenEarthMap owns every class; pinned Cityscapes output may only "
                "promote pixels to canonical road"),
            "inference_passes": (
                aerial["inference_passes"] + road["inference_passes"]),
        }

    def predict(self, image: np.ndarray) -> np.ndarray:
        labels = self.aerial.predict(image)
        road_labels = self.road.predict(image)
        return _fuse_aerial_with_road_specialist(labels, road_labels)


def _fuse_aerial_with_road_specialist(
        aerial_labels: np.ndarray, specialist_labels: np.ndarray) -> np.ndarray:
    if aerial_labels.shape != specialist_labels.shape:
        raise ValueError("semantic fusion inputs must have identical dimensions")
    fused = aerial_labels.copy()
    fused[specialist_labels == CANONICAL_IDS["road"]] = CANONICAL_IDS["road"]
    return fused


def _validate_labels(labels: np.ndarray, height: int, width: int) -> np.ndarray:
    if not isinstance(labels, np.ndarray) or labels.ndim != 2:
        raise ValueError("semantic provider must return a two-dimensional label array")
    if labels.shape != (height, width):
        raise ValueError(
            f"semantic provider returned {labels.shape}; expected {(height, width)}")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("semantic labels must use an integer dtype")
    minimum = int(np.min(labels))
    maximum = int(np.max(labels))
    if minimum < 0 or maximum >= len(CANONICAL_CLASSES):
        raise ValueError(
            f"semantic labels contain IDs outside 0..{len(CANONICAL_CLASSES) - 1}")
    return labels.astype(np.uint8, copy=True)


def _clean_components(labels: np.ndarray, minimum_area: int) -> tuple[np.ndarray, dict]:
    cleaned = labels.copy()
    removed_components = 0
    removed_pixels = 0
    if minimum_area > 0:
        for class_id in range(1, len(CANONICAL_CLASSES)):
            binary = (cleaned == class_id).astype(np.uint8)
            count, connected, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
            for component in range(1, count):
                area = int(stats[component, cv2.CC_STAT_AREA])
                if area < minimum_area:
                    cleaned[connected == component] = CANONICAL_IDS["unknown"]
                    removed_components += 1
                    removed_pixels += area
    return cleaned, {
        "removed_components": removed_components,
        "removed_pixels": removed_pixels,
        "minimum_component_area_px": minimum_area,
    }


def _component_audit(labels: np.ndarray) -> dict:
    classes = []
    for class_id, class_name in enumerate(CANONICAL_CLASSES[1:], start=1):
        binary = (labels == class_id).astype(np.uint8)
        count, _, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
        components = []
        for component in range(1, count):
            x, y, width, height, area = (
                int(value) for value in stats[component])
            components.append({
                "id": component,
                "area_px": area,
                "bounds": {"x": x, "y": y, "width": width, "height": height},
                "centroid": {
                    "u": round(float(centroids[component, 0]), 4),
                    "v": round(float(centroids[component, 1]), 4),
                },
                "touches_image_border": bool(
                    x == 0 or y == 0 or x + width == labels.shape[1] or
                    y + height == labels.shape[0]),
            })
        classes.append({
            "id": class_id,
            "name": class_name,
            "component_count": len(components),
            "components": components,
        })
    return {"schema_version": "stage4-components-v1", "classes": classes}


def _write_png(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise OSError(f"could not write image: {path}")


def _draw_overlay(ground: np.ndarray, labels: np.ndarray, valid: np.ndarray,
                  alpha: float) -> np.ndarray:
    color = np.zeros_like(ground)
    for class_id, bgr in enumerate(CANONICAL_COLORS_BGR):
        color[labels == class_id] = bgr
    overlay = ground.copy()
    blend_pixels = valid & (labels != CANONICAL_IDS["unknown"])
    blended = cv2.addWeighted(ground, 1.0 - alpha, color, alpha, 0.0)
    overlay[blend_pixels] = blended[blend_pixels]
    overlay[valid & (labels == CANONICAL_IDS["unknown"])] = cv2.addWeighted(
        ground, 0.7, color, 0.3, 0.0)[valid & (labels == CANONICAL_IDS["unknown"])]

    legend_height = 18 + 24 * len(CANONICAL_CLASSES)
    legend_width = 220
    cv2.rectangle(overlay, (8, 8), (8 + legend_width, 8 + legend_height),
                  (15, 15, 15), -1)
    cv2.putText(overlay, "STAGE 4A - REVIEW REQUIRED", (18, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
    for index, (name, bgr) in enumerate(zip(CANONICAL_CLASSES, CANONICAL_COLORS_BGR)):
        y = 50 + index * 24
        cv2.rectangle(overlay, (18, y - 12), (34, y + 4), bgr, -1)
        cv2.putText(overlay, f"{index}: {name}", (42, y + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255),
                    1, cv2.LINE_AA)
    return overlay


def _validate_output_target(output: Path) -> None:
    if output.exists() and not output.is_dir():
        raise ValueError(f"Stage 4 output target is not a directory: {output}")
    if output.exists() and any(output.iterdir()) and not (output / OUTPUT_MARKER).is_file():
        raise ValueError(
            f"refusing to replace unrelated non-empty directory: {output}")


def _publish_directory(temporary: Path, output: Path) -> None:
    backup = output.with_name(f".{output.name}-previous")
    if backup.exists():
        shutil.rmtree(backup)
    if output.exists():
        output.replace(backup)
    try:
        temporary.replace(output)
    except Exception:
        if backup.exists() and not output.exists():
            backup.replace(output)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def run_stage4_semantics(stage1_bundle, stage2_bundle,
                         output_directory="stage4_out",
                         options: Stage4Options | None = None,
                         provider: SemanticProvider | None = None,
                         external_labels=None) -> dict:
    options = options or Stage4Options()
    options.validate()
    inputs = _load_inputs(stage1_bundle, stage2_bundle)
    output = Path(output_directory).resolve()
    _validate_output_target(output)
    if provider is not None and external_labels is not None:
        raise ValueError("provide either a semantic provider or external labels, not both")
    if provider is None:
        if external_labels:
            provider = ExternalLabelProvider(external_labels)
        elif options.semantic_backend == "hybrid-openearthmap-road":
            provider = HybridOpenEarthMapRoadProvider(options)
        elif options.semantic_backend == "openearthmap-mask2former":
            provider = OpenEarthMapMask2FormerProvider(options)
        else:
            provider = YoloSemanticProvider(options)

    log("stage4", f"semantic provider={provider.metadata.get('implementation', 'custom')}")
    raw_labels = _validate_labels(
        provider.predict(inputs["ground"]), inputs["height"], inputs["width"])
    predicted_outside_valid = int(np.count_nonzero(raw_labels[~inputs["valid"]]))
    raw_labels[~inputs["valid"]] = CANONICAL_IDS["unknown"]
    labels, cleanup = _clean_components(
        raw_labels, options.minimum_component_area_px)
    labels[~inputs["valid"]] = CANONICAL_IDS["unknown"]

    valid_pixels = int(np.count_nonzero(inputs["valid"]))
    total_pixels = inputs["width"] * inputs["height"]
    coverage = []
    for class_id, class_name in enumerate(CANONICAL_CLASSES):
        count = int(np.count_nonzero((labels == class_id) & inputs["valid"]))
        coverage.append({
            "id": class_id,
            "name": class_name,
            "pixels": count,
            "valid_area_pct": round(100.0 * count / valid_pixels, 6),
        })
    coverage_by_name = {item["name"]: item["valid_area_pct"] for item in coverage}
    components = _component_audit(labels)
    component_count = sum(item["component_count"] for item in components["classes"])
    components_per_megapixel = component_count / max(1e-9, total_pixels / 1_000_000.0)

    failures = []
    warnings = []
    if coverage_by_name["unknown"] > options.maximum_unknown_valid_pct:
        failures.append(
            f"unknown semantic area {coverage_by_name['unknown']:.2f}% exceeds "
            f"{options.maximum_unknown_valid_pct:.2f}%")
    if coverage_by_name["building"] > options.building_area_warning_pct:
        warnings.append(
            f"Building proposal covers {coverage_by_name['building']:.2f}% of valid "
            "pixels; this is suspicious for top-down footage and must be reviewed.")
    if coverage_by_name["other_static"] > options.other_static_warning_pct:
        warnings.append(
            f"Other-static proposal covers {coverage_by_name['other_static']:.2f}% of "
            "valid pixels and may contain semantic confusion.")
    if components_per_megapixel > options.fragmentation_warning_components_per_megapixel:
        warnings.append(
            f"Semantic map is fragmented ({components_per_megapixel:.1f} components per "
            "megapixel); correct or smooth it before vectorization.")
    if coverage_by_name["building"] == 0.0:
        warnings.append("No building pixels were proposed; this is valid only when none are visible.")
    if coverage_by_name["vegetation"] == 0.0:
        warnings.append("No vegetation pixels were proposed; review if vegetation is visible.")
    road_components = next(
        item for item in components["classes"] if item["name"] == "road")
    if road_components["component_count"] and not any(
            item["touches_image_border"] for item in road_components["components"]):
        warnings.append("No proposed road component reaches the image boundary.")
    metadata = provider.metadata
    if metadata.get("domain") == "cityscapes_street_view_bootstrap":
        warnings.append(
            "The bootstrap semantic model is trained on street-view Cityscapes, not "
            "aerial footage. Treat every label as provisional and inspect the overlay.")
    if "openearthmap" in str(metadata.get("domain", "")):
        warnings.append(
            "The proposal model is aerial-domain but remains sensitive to ground "
            "sampling distance, geography, shadows, and synthetic/stylized footage. "
            "Inspect every label before acceptance.")
    if inputs["metric_scale_is_assumed"]:
        warnings.append(
            "Stage 2 scale is assumed; later horizontal dimensions and heights are not "
            "physical measurements.")
    if predicted_outside_valid:
        warnings.append(
            f"Forced {predicted_outside_valid} provider-labelled unsupported pixels to unknown.")

    temporary = Path(tempfile.mkdtemp(
        prefix=f".{output.name}-", dir=output.parent))
    try:
        (temporary / OUTPUT_MARKER).write_text(STAGE4_SCHEMA, encoding="utf-8")
        labels_path = temporary / "semantic_labels.png"
        overlay_path = temporary / "semantic_overlay.png"
        components_path = temporary / "components.json"
        _write_png(labels_path, labels)
        _write_png(
            overlay_path,
            _draw_overlay(inputs["ground"], labels, inputs["valid"], options.overlay_alpha))
        components_path.write_text(
            json.dumps(components, indent=2), encoding="utf-8")
        mask_outputs = []
        for class_id, class_name in enumerate(CANONICAL_CLASSES[1:], start=1):
            filename = f"mask_{class_name}.png"
            _write_png(temporary / filename, ((labels == class_id) * 255).astype(np.uint8))
            mask_outputs.append({"name": class_name, "path": filename})

        artifact_hashes = {
            "semantic_labels_sha256": _sha256(labels_path),
            "semantic_overlay_sha256": _sha256(overlay_path),
            "components_sha256": _sha256(components_path),
        }
        report = {
            "schema_version": STAGE4_SCHEMA,
            "stage": 4,
            "substage": "semantic_proposal",
            "pass": not failures,
            "requires_human_review": True,
            "ready_for_geometry": False,
            "failure_reasons": failures,
            "warnings": warnings,
            "source_stage1": {
                "bundle_path": str(inputs["stage1_root"]),
                "report_sha256": _sha256(inputs["stage1_report_path"]),
                "ground_sha256": _sha256(inputs["ground_path"]),
                "valid_mask_sha256": _sha256(inputs["valid_path"]),
                "source_video_sha256": inputs["stage1"]["input"].get("sha256"),
                "pixel_width": inputs["width"],
                "pixel_height": inputs["height"],
            },
            "source_stage2": {
                "bundle_path": str(inputs["stage2_root"]),
                "calibration_sha256": _sha256(inputs["calibration_path"]),
                "confidence": inputs["scale_confidence"],
                "metric_scale_is_assumed": inputs["metric_scale_is_assumed"],
            },
            "coordinates": {
                "space": "stage1_reference_image_pixels",
                "origin": "top_left",
                "u_axis": "right",
                "v_axis": "down",
                "future_world_mapping": "stage2_reference_pixel_to_world_xz",
            },
            "canonical_classes": [
                {"id": index, "name": name}
                for index, name in enumerate(CANONICAL_CLASSES)
            ],
            "semantic_provider": metadata,
            "quality": {
                "width": inputs["width"],
                "height": inputs["height"],
                "valid_pixels": valid_pixels,
                "valid_area_pct": round(100.0 * valid_pixels / total_pixels, 6),
                "class_coverage": coverage,
                "known_valid_area_pct": round(100.0 - coverage_by_name["unknown"], 6),
                "component_count": component_count,
                "components_per_megapixel": round(components_per_megapixel, 6),
                "cleanup": cleanup,
                "provider_labels_outside_valid_mask": predicted_outside_valid,
            },
            "options": asdict(options),
            "limitations": [
                "A structural pass does not establish semantic accuracy.",
                "Stationary near-nadir monocular footage does not provide measured height.",
                "Roof appearance is not guaranteed to equal the ground-level footprint.",
                "An operator acceptance record is required before polygonization.",
            ],
            "outputs": {
                "report": "report.json",
                "semantic_labels": "semantic_labels.png",
                "semantic_overlay": "semantic_overlay.png",
                "components": "components.json",
                "class_masks": mask_outputs,
                "acceptance": "semantic_acceptance.json",
            },
            "artifact_hashes": artifact_hashes,
        }
        (temporary / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
        _publish_directory(temporary, output)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    log(
        "stage4",
        f"{'PASS' if report['pass'] else 'FAIL'}: known="
        f"{report['quality']['known_valid_area_pct']:.2f}%; "
        f"components={component_count}; review={output / 'semantic_overlay.png'}")
    return report
