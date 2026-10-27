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
