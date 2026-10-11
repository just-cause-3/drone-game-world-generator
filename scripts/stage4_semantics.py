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
