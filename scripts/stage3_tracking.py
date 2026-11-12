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

from .common import log, publish_directory
from .progress import PhaseProgress
from .detection_merge import suppress_tile_fragments
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
    detector_car_confidence: float = 0.20
    detector_person_confidence: float = 0.60
    detector_iou: float = 0.45
    detector_max_detections: int = 3000
    detector_full_frame: bool = True
    detector_tile_size: int = 640
    detector_tile_overlap: float = 0.25
    detector_tta: bool = True
    detector_frame_stride: int = 1
    min_confirmed_observations: int = 5
    person_minimum_motion_px: float = 4.0
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
    stationary_heading_speed_threshold: float = 0.5
    boundary_tolerance_fraction: float = 0.002
    preview_line_thickness: int = 2

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.detector_imgsz < 320:
            raise ValueError("detector_imgsz must be at least 320")
        if not 0.0 < self.detector_low_confidence < 1.0:
            raise ValueError("detector_low_confidence must be in (0, 1)")
        if not self.detector_low_confidence < self.detector_high_confidence < 1.0:
            raise ValueError(
                "detector_high_confidence must exceed the low threshold and be below 1")
        if not self.detector_low_confidence <= self.detector_car_confidence < 1.0:
            raise ValueError(
                "detector_car_confidence must be at least the low threshold and below 1")
        if not self.detector_low_confidence <= self.detector_person_confidence < 1.0:
            raise ValueError(
                "detector_person_confidence must be at least the low threshold and below 1")
        if not 0.0 < self.detector_iou < 1.0:
            raise ValueError("detector_iou must be in (0, 1)")
        if self.detector_max_detections < 1:
            raise ValueError("detector_max_detections must be positive")
        if self.detector_tile_size != 0 and self.detector_tile_size < 128:
            raise ValueError("detector_tile_size must be zero or at least 128")
        if not 0.0 <= self.detector_tile_overlap < 0.9:
            raise ValueError("detector_tile_overlap must be in [0, 0.9)")
        if self.detector_frame_stride < 1:
            raise ValueError("detector_frame_stride must be at least one")
        if self.min_confirmed_observations < 1:
            raise ValueError("min_confirmed_observations must be positive")
        if self.person_minimum_motion_px < 0.0:
            raise ValueError("person_minimum_motion_px cannot be negative")
        if self.max_lost_frames < 0 or self.max_interpolation_frames < 0:
            raise ValueError("gap limits cannot be negative")
        if self.max_interpolation_frames > self.max_lost_frames:
            raise ValueError("interpolation cannot exceed the tracker lost-frame limit")
        if self.detector_frame_stride > self.max_interpolation_frames + 1:
            raise ValueError("detector stride skips more frames than interpolation allows")
        if (self.car_link_diagonal_fraction_per_frame <= 0.0 or
                self.person_link_diagonal_fraction_per_frame <= 0.0):
            raise ValueError("association motion limits must be positive")
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
        if self.stationary_heading_speed_threshold < 0.0:
            raise ValueError("stationary_heading_speed_threshold cannot be negative")
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
    if any(not math.isfinite(value) or value <= 0 for value in result.values()):
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
    if any(not math.isfinite(value) or value <= 0 for value in declared.values()):
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
    ground_y = float(coordinate_system.get("ground_y_m", 0.0))
    if not math.isfinite(ground_y):
        raise ValueError("Stage 2 ground elevation must be finite")

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
        "ground_y": ground_y,
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


class YoloActorDetector:
    def __init__(self, options: Stage3Options):
        from ultralytics import YOLO

        self.options = options
        self.device = resolve_device("auto")
        requested = Path(str(options.detector_model))
        repository_model = Path(__file__).resolve().parents[1] / requested
        model_path = (repository_model if not requested.is_absolute() and
                      repository_model.exists() else requested)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"drone detector is missing: {model_path}. Run "
                "'.venv\\Scripts\\python.exe setup_drone_model.py' once.")
        self.model_path = model_path.resolve()
        model_sha256 = _sha256(self.model_path)
        self.model_sha256 = model_sha256
        if (str(options.detector_model) == DEFAULT_FOREGROUND_MODEL and
                model_sha256 != MODEL_SHA256):
            raise ValueError(
                "default drone detector hash does not match the pinned checkpoint; "
                "run '.venv\\Scripts\\python.exe setup_drone_model.py --verify-only'")
        self.model = YOLO(str(self.model_path))
        raw_names = self.model.names
        self.model_names = {
            int(key): str(value) for key, value in
            (raw_names.items() if isinstance(raw_names, dict) else enumerate(raw_names))
        }
        if (str(options.detector_model) == DEFAULT_FOREGROUND_MODEL and
                tuple(self.model_names.values()) != EXPECTED_CLASSES):
            raise ValueError(
                f"default drone detector class map is not pinned: {self.model_names}")
        self.class_kinds = {}
        for class_id, name in self.model_names.items():
            normalized = _normalized_class_name(name)
            if normalized in CANONICAL_CLASSES:
                self.class_kinds[class_id] = CANONICAL_CLASSES[normalized]
        if set(self.class_kinds.values()) != {"car", "person"}:
            raise ValueError(
                "detector must contain car and pedestrian/person classes; "
                f"model classes={self.model_names}")
        log(
            "stage3",
            f"detector={self.model_path} device={self.device}; full_frame="
            f"{options.detector_full_frame}, tile={options.detector_tile_size}, "
            f"tta={options.detector_tta}")

    @property
    def metadata(self) -> dict:
        return {
            "model": str(self.model_path),
            "model_sha256": self.model_sha256,
            "pinned_default": str(self.options.detector_model) == DEFAULT_FOREGROUND_MODEL,
            "device": str(self.device),
            "model_classes": self.model_names,
            "canonical_classes": {str(key): value for key, value in self.class_kinds.items()},
            "low_confidence": self.options.detector_low_confidence,
            "high_confidence": self.options.detector_high_confidence,
            "class_confidence": {
                "car": self.options.detector_car_confidence,
                "person": self.options.detector_person_confidence,
            },
            "iou": self.options.detector_iou,
            "imgsz": self.options.detector_imgsz,
            "full_frame": self.options.detector_full_frame,
            "tile_size": self.options.detector_tile_size,
            "tile_overlap": self.options.detector_tile_overlap,
            "test_time_augmentation": self.options.detector_tta,
            "retains_low_confidence_for_association": True,
            "tile_fragment_suppression": "internal-edge-containment-v1",
        }

    def _predict(self, frame: np.ndarray, x_offset: int, y_offset: int,
                 pass_name: str) -> list[dict]:
        result = self.model.predict(
            frame,
            imgsz=self.options.detector_imgsz,
            conf=self.options.detector_low_confidence,
            iou=self.options.detector_iou,
            max_det=self.options.detector_max_detections,
            classes=sorted(self.class_kinds),
            device=self.device,
            augment=self.options.detector_tta,
            agnostic_nms=False,
            end2end=False,
            verbose=False,
        )[0]
        if result.boxes is None:
            return []
        detections = []
        for box, class_value, score in zip(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.cls.cpu().numpy(),
                result.boxes.conf.cpu().numpy()):
            class_id = int(class_value)
            kind = self.class_kinds.get(class_id)
            if kind is None:
                continue
            confidence = float(score)
            # Class thresholds govern track creation, not candidate retention.
            # Weak detections can continue an existing identity but cannot spawn one.
            if confidence < self.options.detector_low_confidence:
                continue
            x1, y1, x2, y2 = (float(value) for value in box)
            detections.append({
                "kind": kind,
                "detector_class_id": class_id,
                "detector_class": self.model_names[class_id],
                "confidence": confidence,
                "bbox_xyxy": [
                    x1 + x_offset, y1 + y_offset,
                    x2 + x_offset, y2 + y_offset,
                ],
                "inference_pass": pass_name,
            })
        return detections

    def detect(self, frame: np.ndarray, frame_index: int) -> list[dict]:
        del frame_index
        height, width = frame.shape[:2]
        candidates = []
        if self.options.detector_full_frame:
            candidates.extend(self._predict(frame, 0, 0, "full"))
        tile_size = self.options.detector_tile_size
        tiled = tile_size > 0 and (width > tile_size or height > tile_size)
        if tiled:
            tile_width = min(tile_size, width)
            tile_height = min(tile_size, height)
            for y0 in _tile_origins(
                    height, tile_height, self.options.detector_tile_overlap):
                for x0 in _tile_origins(
                        width, tile_width, self.options.detector_tile_overlap):
                    crop = frame[y0:y0 + tile_height, x0:x0 + tile_width]
                    candidates.extend(self._predict(
                        crop, x0, y0, f"tile:{x0},{y0}"))
        if not self.options.detector_full_frame and not tiled:
            candidates.extend(self._predict(frame, 0, 0, "single"))
        candidates = suppress_tile_fragments(candidates, frame.shape, tile_size)
        return _canonical_nms(candidates, self.options.detector_iou)


class CachedActorDetector:
    """Replay audited YOLO detections after strict upstream provenance checks."""
    def __init__(self, bundle_directory):
        self.bundle = Path(bundle_directory).resolve()
        report_path = self.bundle / "report.json"
        detections_path = self.bundle / "detections.jsonl"
        if not report_path.is_file() or not detections_path.is_file():
            raise FileNotFoundError(
                "cached Stage 3 bundle must contain report.json and detections.jsonl")
        self.report = _read_json(report_path)
        if (self.report.get("schema_version") != STAGE3_SCHEMA or
                self.report.get("stage") != 3):
            raise ValueError("cached detections do not come from a Stage 3 bundle")
        expected_hash = (self.report.get("artifact_hashes") or {}).get("detections_sha256")
        if expected_hash and _sha256(detections_path) != expected_hash:
            raise ValueError("cached detections hash differs from the Stage 3 report")
        metadata = self.report.get("detector")
        if not isinstance(metadata, dict):
            raise ValueError("cached Stage 3 report has no detector metadata")
        self._metadata = dict(metadata)
        self._metadata["reused_detections"] = True
        self._metadata["cache_source"] = str(self.bundle)
        self._metadata["tile_fragment_suppression"] = "internal-edge-containment-v1"
        self.by_frame: dict[int, list[dict]] = {}
        previous_frame = -1
        with detections_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    frame = int(item["frame"])
                    box = [float(value) for value in item["bbox_xyxy"]]
                    confidence = float(item["confidence"])
                    kind = str(item["kind"])
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exception:
                    raise ValueError(
                        f"invalid cached detection on line {line_number}") from exception
                if frame < 0 or frame < previous_frame:
                    raise ValueError("cached detections are not frame-monotonic")
                if kind not in {"car", "person"} or len(box) != 4 or not all(
                        math.isfinite(value) for value in box):
                    raise ValueError(f"invalid cached detection on line {line_number}")
                if not 0.0 <= confidence <= 1.0:
                    raise ValueError(f"invalid cached confidence on line {line_number}")
                previous_frame = frame
                self.by_frame.setdefault(frame, []).append({
                    "kind": kind,
                    "detector_class_id": int(item.get("detector_class_id", -1)),
                    "detector_class": str(item.get("detector_class", kind)),
                    "confidence": confidence,
                    "bbox_xyxy": box,
                    "inference_passes": list(item.get("inference_passes", [])),
                })

    @property
    def metadata(self) -> dict:
        return self._metadata

    def validate_inputs(self, inputs: dict) -> None:
        source1 = self.report.get("source_stage1") or {}
        source2 = self.report.get("source_stage2") or {}
        timeline = self.report.get("timeline") or {}
        expected = {
            "report_sha256": _sha256(inputs["stage1_report_path"]),
            "cfr_video_sha256": inputs["video_sha256"],
            "frame_transforms_sha256": inputs["transforms_sha256"],
        }
        for key, value in expected.items():
            if source1.get(key) != value:
                raise ValueError(f"cached detections are stale: Stage 1 {key} changed")
        if source2.get("calibration_sha256") != _sha256(inputs["calibration_path"]):
            raise ValueError("cached detections are stale: Stage 2 calibration changed")
        if (int(timeline.get("frame_count", -1)) != inputs["frame_count"] or
                abs(float(timeline.get("fps", -1.0)) - inputs["fps"]) > 1e-3):
            raise ValueError("cached detections use a different CFR timeline")
        if self.by_frame and max(self.by_frame) >= inputs["frame_count"]:
            raise ValueError("cached detections contain an out-of-range frame")

    def validate_options(self, options: Stage3Options) -> None:
        stride = (self.report.get("options") or {}).get("detector_frame_stride", 1)
        if stride != options.detector_frame_stride:
            raise ValueError(
                "cached detector cadence differs from requested stride; "
                "reuse the original stride or run fresh inference")

    def detect(self, frame: np.ndarray, frame_index: int) -> list[dict]:
        candidates = [dict(item) for item in self.by_frame.get(frame_index, [])]
        return suppress_tile_fragments(
            candidates, frame.shape, int(self._metadata.get("tile_size", 0)))


class _Track:
    def __init__(self, identifier: int, detection: dict):
        self.identifier = identifier
        self.kind = detection["kind"]
        self.observations = [detection]
        self.last_frame = detection["frame"]
        self.last_position = np.asarray(detection["reference_anchor"], dtype=np.float64)
        self.velocity = np.zeros(2, dtype=np.float64)
        self.last_box = detection["reference_bbox_xyxy"]
        descriptor = detection.get("_appearance")
        self.appearance = None if descriptor is None else descriptor.copy()

    def predict(self, frame_index: int) -> np.ndarray:
        return self.last_position + self.velocity * (frame_index - self.last_frame)

    def append(self, detection: dict) -> None:
        position = np.asarray(detection["reference_anchor"], dtype=np.float64)
        dt = detection["frame"] - self.last_frame
        if dt > 0:
            measured = (position - self.last_position) / dt
            self.velocity = measured if len(self.observations) == 1 else (
                0.65 * self.velocity + 0.35 * measured)
        self.observations.append(detection)
        self.last_frame = detection["frame"]
        self.last_position = position
        self.last_box = detection["reference_bbox_xyxy"]
        descriptor = detection.get("_appearance")
        if descriptor is not None:
            self.appearance = (descriptor.copy() if self.appearance is None else
                               (0.8 * self.appearance + 0.2 * descriptor).astype(np.float32))


def _assignment(cost: np.ndarray) -> list[tuple[int, int]]:
    if cost.size == 0:
        return []
    try:
        from scipy.optimize import linear_sum_assignment
        rows, columns = linear_sum_assignment(cost)
        return [(int(row), int(column)) for row, column in zip(rows, columns)]
    except ImportError:
        candidates = sorted(
            (float(cost[row, column]), row, column)
            for row in range(cost.shape[0]) for column in range(cost.shape[1]))
        used_rows = set()
        used_columns = set()
        result = []
        for _, row, column in candidates:
            if row not in used_rows and column not in used_columns:
                result.append((row, column))
                used_rows.add(row)
                used_columns.add(column)
        return result


class StabilizedTracker:
    def __init__(self, width: int, height: int, fps: float, options: Stage3Options):
        self.width = width
        self.height = height
        self.fps = fps
        self.options = options
        self.diagonal = math.hypot(width, height)
        self.next_identifier = 1
        self.active: list[_Track] = []
        self.finished: list[_Track] = []

    def _gate(self, kind: str, dt: int) -> float:
        fraction = (
            self.options.car_link_diagonal_fraction_per_frame
            if kind == "car"
            else self.options.person_link_diagonal_fraction_per_frame)
        return max(
            self.options.minimum_link_distance_px,
            self.diagonal * fraction * max(1, dt))

    @staticmethod
    def _box_area(box: Iterable[float]) -> float:
        x1, y1, x2, y2 = box
        return max(1e-6, (x2 - x1) * (y2 - y1))

    def _associate(self, tracks: list[_Track], detections: list[dict],
                   frame_index: int) -> tuple[list[tuple[int, int]], set[int], set[int]]:
        if not tracks or not detections:
            return [], set(range(len(tracks))), set(range(len(detections)))
        invalid = 1e6
        cost = np.full((len(tracks), len(detections)), invalid, dtype=np.float64)
        for track_index, track in enumerate(tracks):
            dt = frame_index - track.last_frame
            gate = self._gate(track.kind, dt)
            predicted = track.predict(frame_index)
            old_area = self._box_area(track.last_box)
            for detection_index, detection in enumerate(detections):
                if detection["kind"] != track.kind:
                    continue
                position = np.asarray(detection["reference_anchor"], dtype=np.float64)
                distance = float(np.linalg.norm(position - predicted))
                new_area = self._box_area(detection["reference_bbox_xyxy"])
                size_ratio = max(old_area, new_area) / min(old_area, new_area)
                if distance > gate or size_ratio > self.options.maximum_size_ratio:
                    continue
                appearance_distance = _appearance_distance(
                    track.appearance, detection.get("_appearance"))
                if (appearance_distance is not None and
                        appearance_distance > self.options.appearance_max_distance):
                    continue
                overlap = _iou(track.last_box, detection["reference_bbox_xyxy"])
                size_penalty = abs(math.log(new_area / old_area))
                cost[track_index, detection_index] = (
                    distance / gate + 0.20 * (1.0 - overlap) +
                    0.10 * size_penalty + self.options.appearance_cost_weight *
                    (appearance_distance if appearance_distance is not None else 0.0))
        matches = []
        for track_index, detection_index in _assignment(cost):
            if cost[track_index, detection_index] < invalid:
                matches.append((track_index, detection_index))
        matched_tracks = {item[0] for item in matches}
        matched_detections = {item[1] for item in matches}
        return (
            matches,
            set(range(len(tracks))) - matched_tracks,
            set(range(len(detections))) - matched_detections,
        )

    def update(self, frame_index: int, detections: list[dict]) -> None:
        survivors = []
        for track in self.active:
            if frame_index - track.last_frame - 1 > self.options.max_lost_frames:
                self.finished.append(track)
            else:
                survivors.append(track)
        self.active = survivors

        high = [detection for detection in detections
                if detection["confidence"] >= _track_start_threshold(detection["kind"], self.options)]
        low = [detection for detection in detections
               if self.options.detector_low_confidence <= detection["confidence"] <
               _track_start_threshold(detection["kind"], self.options)]
        matches, unmatched_track_indices, unmatched_high_indices = self._associate(
            self.active, high, frame_index)
        for track_index, detection_index in matches:
            detection = high[detection_index]
            detection["track_id"] = self.active[track_index].identifier
            self.active[track_index].append(detection)

        unmatched_tracks = [self.active[index] for index in sorted(unmatched_track_indices)]
        low_matches, _, _ = self._associate(unmatched_tracks, low, frame_index)
        for local_track_index, detection_index in low_matches:
            track = unmatched_tracks[local_track_index]
            detection = low[detection_index]
            detection["track_id"] = track.identifier
            track.append(detection)

        for detection_index in sorted(unmatched_high_indices):
            detection = high[detection_index]
            detection["track_id"] = self.next_identifier
            track = _Track(self.next_identifier, detection)
            self.next_identifier += 1
            self.active.append(track)

    def all_tracks(self) -> list[_Track]:
        return sorted(self.finished + self.active, key=lambda track: track.identifier)


def _track_start_threshold(kind: str, options: Stage3Options) -> float:
    class_threshold = (options.detector_car_confidence if kind == "car"
                       else options.detector_person_confidence)
    return max(options.detector_high_confidence, class_threshold)


def _rts_smooth(observations: list[dict],
                process_noise: float, measurement_noise: float) -> np.ndarray:
    # Detector uncertainty is in reference pixels, not metres. Model motion per
    # frame so an arbitrary Stage 2 scale cannot change the shape of a trajectory.
    measurements = np.asarray(
        [item["reference_anchor"] for item in observations], dtype=np.float64)
    frames = np.asarray([item["frame"] for item in observations], dtype=np.int64)
    count = len(observations)
    if count < 3:
        return measurements.copy()
    states = np.zeros((count, 4), dtype=np.float64)
    covariances = np.zeros((count, 4, 4), dtype=np.float64)
    predicted_states = np.zeros_like(states)
    predicted_covariances = np.zeros_like(covariances)
    state = np.array([measurements[0, 0], measurements[0, 1], 0.0, 0.0])
    covariance = np.diag([measurement_noise, measurement_noise, 10.0, 10.0])
    observation_matrix = np.array([[1.0, 0.0, 0.0, 0.0],
                                   [0.0, 1.0, 0.0, 0.0]])
    observation_noise = np.eye(2) * measurement_noise
    transitions = [np.eye(4)]
    for index in range(count):
        if index > 0:
            dt = float(frames[index] - frames[index - 1])
            transition = np.array([
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ])
            transitions.append(transition)
            q = process_noise
            process = q * np.array([
                [dt ** 4 / 4, 0.0, dt ** 3 / 2, 0.0],
                [0.0, dt ** 4 / 4, 0.0, dt ** 3 / 2],
                [dt ** 3 / 2, 0.0, dt ** 2, 0.0],
                [0.0, dt ** 3 / 2, 0.0, dt ** 2],
            ])
            state = transition @ state
            covariance = transition @ covariance @ transition.T + process
        predicted_states[index] = state
        predicted_covariances[index] = covariance
        confidence = max(0.05, float(observations[index]["confidence"]))
        adaptive_noise = observation_noise / confidence
        innovation = measurements[index] - observation_matrix @ state
        innovation_covariance = (
            observation_matrix @ covariance @ observation_matrix.T + adaptive_noise)
        gain = covariance @ observation_matrix.T @ np.linalg.inv(innovation_covariance)
        state = state + gain @ innovation
        covariance = (np.eye(4) - gain @ observation_matrix) @ covariance
        states[index] = state
        covariances[index] = covariance

    smoothed = states.copy()
    smooth_covariances = covariances.copy()
    for index in range(count - 2, -1, -1):
        transition = transitions[index + 1]
        predicted_covariance = predicted_covariances[index + 1]
        gain = (covariances[index] @ transition.T @
                np.linalg.pinv(predicted_covariance))
        smoothed[index] = states[index] + gain @ (
            smoothed[index + 1] - predicted_states[index + 1])
        smooth_covariances[index] = covariances[index] + gain @ (
            smooth_covariances[index + 1] - predicted_covariance) @ gain.T
    return smoothed[:, :2]


def _playback_track(track: _Track, fps: float, options: Stage3Options,
                    pixel_to_world: np.ndarray,
                    image_size: tuple[int, int] | None = None) -> dict:
    observations = sorted(track.observations, key=lambda item: item["frame"])
    # Never smooth through an occlusion the export policy refuses to interpolate.
    boundaries = [0] + [index for index in range(1, len(observations))
                       if observations[index]["frame"] - observations[index - 1]["frame"] - 1 >
                       options.max_interpolation_frames] + [len(observations)]
    smooth_positions = np.concatenate([
        _rts_smooth(observations[start:end], options.process_noise, options.measurement_noise)
        for start, end in zip(boundaries, boundaries[1:])])
    if image_size is not None:
        smooth_positions = np.clip(smooth_positions, [0.0, 0.0],
                                   [image_size[0] - 1.0, image_size[1] - 1.0])
    samples = []
    for index, observation in enumerate(observations):
        start_frame = int(observation["frame"])
        start_position = smooth_positions[index]
        world_position = _project_point(pixel_to_world, *start_position)
        samples.append({
            "frame": start_frame,
            "t": round(start_frame / fps, 6),
            "x": world_position[0],
            "z": world_position[1],
            "observed": True,
            "confidence": float(observation["confidence"]),
        })
        if index + 1 >= len(observations):
            continue
        end_frame = int(observations[index + 1]["frame"])
        gap = end_frame - start_frame - 1
        if gap <= 0 or gap > options.max_interpolation_frames:
            continue
        end_position = smooth_positions[index + 1]
        for frame in range(start_frame + 1, end_frame):
            alpha = (frame - start_frame) / (end_frame - start_frame)
            position = (1.0 - alpha) * start_position + alpha * end_position
            world_position = _project_point(pixel_to_world, *position)
            samples.append({
                "frame": frame,
                "t": round(frame / fps, 6),
                "x": world_position[0],
                "z": world_position[1],
                "observed": False,
                "confidence": 0.0,
            })
    samples.sort(key=lambda item: item["frame"])
    observation_frames = [item["frame"] for item in observations]
    measured_pixels = np.asarray([item["reference_anchor"] for item in observations])
    sample_frames = [sample["frame"] for sample in samples]
    measured_at_samples = np.column_stack([
        np.interp(sample_frames, observation_frames, measured_pixels[:, axis])
        for axis in range(2)])
    measured_world = np.asarray([
        _project_point(pixel_to_world, *point) for point in measured_at_samples])
    for index, sample in enumerate(samples):
        previous = samples[max(0, index - 1)]
        following = samples[min(len(samples) - 1, index + 1)]
        if sample["frame"] - previous["frame"] > 1:
            previous = sample
        if following["frame"] - sample["frame"] > 1:
            following = sample
        dt = (following["frame"] - previous["frame"]) / fps
        dx = following["x"] - previous["x"]
        dz = following["z"] - previous["z"]
        speed = math.hypot(dx, dz) / dt if dt > 0.0 else 0.0
        before_index = index if previous is sample else max(0, index - 1)
        after_index = index if following is sample else min(len(samples) - 1, index + 1)
        measured_delta = measured_world[after_index] - measured_world[before_index]
        measured_speed = float(np.linalg.norm(measured_delta)) / dt if dt > 0.0 else 0.0
        # A smoother can overshoot and settle after a stop. Do not interpret
        # that numerical rebound as a vehicle reversing its facing direction.
        if (speed > options.stationary_heading_speed_threshold and
                measured_speed > options.stationary_heading_speed_threshold and
                dx * measured_delta[0] + dz * measured_delta[1] > 0.0):
            heading = math.degrees(math.atan2(dx, dz))
        else:
            heading = None
        sample["x"] = round(sample["x"], 6)
        sample["z"] = round(sample["z"], 6)
        sample["heading_deg"] = round(heading, 4) if heading is not None else None
        sample["speed_units_per_s"] = round(speed, 6)
        sample["confidence"] = round(sample["confidence"], 4)
    # Hold a moving actor's last direction at a stop. For an initially stopped
    # actor, use its first measured direction; a never-moving car has only an
    # ambiguous box-axis prior, which must still pass through Stage 2 calibration.
    for start, end in zip(boundaries, boundaries[1:]):
        segment = [sample for sample in samples
                   if observations[start]["frame"] <= sample["frame"] <= observations[end - 1]["frame"]]
        heading = next((sample["heading_deg"] for sample in segment
                        if sample["heading_deg"] is not None), None)
        if heading is None:
            heading = 0.0
            if track.kind == "car":
                observation = observations[start]
                x1, y1, x2, y2 = observation["reference_bbox_xyxy"]
                u, v = observation["reference_anchor"]
                axis = (1.0, 0.0) if x2 - x1 >= y2 - y1 else (0.0, -1.0)
                origin = _project_point(pixel_to_world, u, v)
                tip = _project_point(pixel_to_world, u + axis[0], v + axis[1])
                heading = round(math.degrees(math.atan2(tip[0] - origin[0], tip[1] - origin[1])), 4)
        for sample in segment:
            if sample["heading_deg"] is None:
                sample["heading_deg"] = heading
            else:
                heading = sample["heading_deg"]
    confidences = [float(item["confidence"]) for item in observations]
    return {
        "id": track.identifier,
        "kind": track.kind,
        "first_frame": int(observations[0]["frame"]),
        "last_frame": int(observations[-1]["frame"]),
        "observed_frames": len(observations),
        "mean_confidence": round(float(np.mean(confidences)), 4),
        "samples": samples,
    }


def _color(identifier: int, kind: str) -> tuple[int, int, int]:
    if kind == "car":
        base = np.array([255, 155, 45], dtype=np.int32)
    else:
        base = np.array([45, 220, 255], dtype=np.int32)
    shift = np.array([(identifier * 53) % 80, (identifier * 29) % 60, 0])
    return tuple(int(value) for value in np.clip(base - shift, 40, 255))


def _annotate(frame: np.ndarray, detections: list[dict], frame_index: int,
              fps: float, thickness: int) -> np.ndarray:
    result = frame.copy()
    for detection in detections:
        x1, y1, x2, y2 = (int(round(value)) for value in detection["bbox_xyxy"])
        identifier = detection.get("track_id")
        color = _color(int(identifier or 0), detection["kind"])
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        identity = f"#{identifier}" if identifier is not None else "unassigned"
        label = f"{detection['kind']} {identity} {detection['confidence']:.2f}"
        cv2.putText(result, label, (x1, max(15, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(result, label, (x1, max(15, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    cv2.putText(
        result, f"frame={frame_index} t={frame_index / fps:.3f}s",
        (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(
        result, f"frame={frame_index} t={frame_index / fps:.3f}s",
        (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return result


def _draw_track_map(ground: np.ndarray, tracks: list[_Track], destination: Path) -> None:
    image = ground.copy()
    for track in tracks:
        points = np.asarray(
            [item["reference_anchor"] for item in track.observations], dtype=np.float64)
        if len(points) == 0:
            continue
        points_i = np.round(points).astype(np.int32).reshape(-1, 1, 2)
        color = _color(track.identifier, track.kind)
        if len(points) > 1:
            cv2.polylines(image, [points_i], False, color, 2, cv2.LINE_AA)
        start = tuple(points_i[0, 0])
        end = tuple(points_i[-1, 0])
        cv2.circle(image, start, 5, color, -1, cv2.LINE_AA)
        cv2.drawMarker(image, end, color, cv2.MARKER_TILTED_CROSS, 10, 2)
        cv2.putText(image, f"{track.kind} #{track.identifier}",
                    (int(start[0]) + 6, int(start[1]) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, f"{track.kind} #{track.identifier}",
                    (int(start[0]) + 6, int(start[1]) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    if not cv2.imwrite(str(destination), image):
        raise IOError(f"could not write track map: {destination}")


def _validate_output_target(output: Path) -> None:
    if output.exists():
        if not output.is_dir():
            raise ValueError(f"refusing to replace unrelated output path: {output}")
        marker = output / OUTPUT_MARKER
        if any(output.iterdir()) and not marker.is_file():
            raise ValueError(f"refusing to replace unrelated output directory: {output}")


def _publish_directory(temporary: Path, output: Path) -> None:
    _validate_output_target(output)
    publish_directory(temporary, output)


def run_stage3(stage1_bundle, stage2_bundle, output_directory="stage3_out",
               options: Stage3Options | None = None, detector=None) -> dict:
    options = options or Stage3Options()
    options.validate()
    output = Path(output_directory).resolve()
    _validate_output_target(output)
    inputs = _load_inputs(stage1_bundle, stage2_bundle)
    with PhaseProgress("stage3", "loading actor detector"):
        detector = detector or YoloActorDetector(options)
    if not hasattr(detector, "detect") or not hasattr(detector, "metadata"):
        raise TypeError("detector must expose detect(frame, frame_index) and metadata")
    validate_inputs = getattr(detector, "validate_inputs", None)
    if validate_inputs is not None:
        validate_inputs(inputs)
    validate_options = getattr(detector, "validate_options", None)
    if validate_options is not None:
        validate_options(options)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{output.name}-", dir=str(output.parent)))
    capture = None
    writer = None
    try:
        (temporary / OUTPUT_MARKER).write_text(STAGE3_SCHEMA + "\n", encoding="utf-8")
        capture = cv2.VideoCapture(str(inputs["video_path"]))
        if not capture.isOpened():
            raise ValueError("could not reopen Stage 1 CFR video")
        preview_path = temporary / "tracking_preview.mp4"
        writer = cv2.VideoWriter(
            str(preview_path), cv2.VideoWriter_fourcc(*"mp4v"), inputs["fps"],
            (inputs["width"], inputs["height"]))
        if not writer.isOpened():
            raise RuntimeError("OpenCV could not create tracking_preview.mp4")

        tracker = StabilizedTracker(
            inputs["width"], inputs["height"], inputs["fps"], options)
        detections_path = temporary / "detections.jsonl"
        counts_path = temporary / "frame_counts.csv"
        detection_counts = Counter()
        high_count = 0
        total_count = 0
        assigned_count = 0
        processed = 0
        inference_frames = 0
        out_of_bounds_count = 0
        out_of_bounds_detections = []
        with PhaseProgress("stage3", "tracking frames", inputs["frame_count"]) as progress, \
                detections_path.open("w", encoding="utf-8", newline="\n") as detection_file, \
                counts_path.open("w", encoding="utf-8", newline="") as counts_file:
            counts_writer = csv.writer(counts_file)
            counts_writer.writerow([
                "frame", "time_s", "car_detections", "person_detections",
                "assigned_detections", "active_tracks",
            ])
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if processed >= inputs["frame_count"]:
                    raise ValueError("CFR decoder produced more frames than Stage 1 declares")
                # Accuracy mode uses stride 1. Balanced fast mode uses stride 2,
                # but still advances the tracker and writes the exact CFR clock on
                # skipped frames. Always infer the final frame so actor lifetimes
                # reach the end of the source clip.
                run_inference = (
                    processed % options.detector_frame_stride == 0 or
                    processed == inputs["frame_count"] - 1)
                raw_detections = detector.detect(frame, processed) if run_inference else []
                if run_inference:
                    inference_frames += 1
                if not isinstance(raw_detections, list):
                    raise TypeError("detector.detect must return a list")
                mapped = []
                frame_transform = inputs["transforms"][processed]
                for raw in raw_detections:
                    kind = str(raw.get("kind", ""))
                    if kind not in {"car", "person"}:
                        raise ValueError(f"detector emitted unsupported actor kind: {kind!r}")
                    score = float(raw["confidence"])
                    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                        raise ValueError("detector emitted an invalid confidence")
                    if score < options.detector_low_confidence:
                        continue
                    box = [float(value) for value in raw["bbox_xyxy"]]
                    if len(box) != 4 or not all(math.isfinite(value) for value in box):
                        raise ValueError("detector emitted an invalid bounding box")
                    x1, y1, x2, y2 = box
                    if x2 <= x1 or y2 <= y1:
                        continue
                    source_u = 0.5 * (x1 + x2)
                    source_v = 0.5 * (y1 + y2)
                    reference_u, reference_v = _project_point(
                        frame_transform, source_u, source_v)
                    reference_box = _transform_box(frame_transform, box)
                    tolerance_x = inputs["width"] * options.boundary_tolerance_fraction
                    tolerance_y = inputs["height"] * options.boundary_tolerance_fraction
                    if not (-tolerance_x <= reference_u <= inputs["width"] - 1 + tolerance_x and
                            -tolerance_y <= reference_v <= inputs["height"] - 1 + tolerance_y):
                        out_of_bounds_count += 1
                        if len(out_of_bounds_detections) < 50:
                            out_of_bounds_detections.append({
                                "frame": processed,
                                "kind": kind,
                                "reference_anchor": [reference_u, reference_v],
                            })
                        continue
                    # Keep the actor centre on the finite reconstructed ground. The
                    # small tolerance absorbs sub-pixel registration noise; clamping
                    # prevents it from becoming an off-plane Unity position.
                    reference_u = min(max(reference_u, 0.0), inputs["width"] - 1.0)
                    reference_v = min(max(reference_v, 0.0), inputs["height"] - 1.0)
                    world_x, world_z = _project_point(
                        inputs["pixel_to_world"], reference_u, reference_v)
                    item = {
                        "frame": processed,
                        "t": round(processed / inputs["fps"], 6),
                        "kind": kind,
                        "detector_class_id": int(raw.get("detector_class_id", -1)),
                        "detector_class": str(raw.get("detector_class", kind)),
                        "confidence": score,
                        "bbox_xyxy": box,
                        "source_anchor": [source_u, source_v],
                        "reference_bbox_xyxy": reference_box,
                        "reference_anchor": [reference_u, reference_v],
                        "world_anchor": [world_x, world_z],
                        "inference_passes": list(raw.get("inference_passes", [])),
                        "_appearance": _appearance_descriptor(frame, box),
                    }
                    mapped.append(item)
                    detection_counts[kind] += 1
                    total_count += 1
                    if score >= _track_start_threshold(kind, options):
                        high_count += 1

                tracker.update(processed, mapped)
                assigned = sum("track_id" in detection for detection in mapped)
                assigned_count += assigned
                for detection in mapped:
                    serialized = dict(detection)
                    serialized.pop("_appearance", None)
                    for key in ("bbox_xyxy", "source_anchor", "reference_bbox_xyxy",
                                "reference_anchor", "world_anchor"):
                        serialized[key] = [round(float(value), 6)
                                           for value in serialized[key]]
                    serialized["confidence"] = round(serialized["confidence"], 6)
                    detection_file.write(json.dumps(serialized, separators=(",", ":")) + "\n")
                counts_writer.writerow([
                    processed,
                    f"{processed / inputs['fps']:.6f}",
                    sum(item["kind"] == "car" for item in mapped),
                    sum(item["kind"] == "person" for item in mapped),
                    assigned,
                    len(tracker.active),
                ])
                writer.write(_annotate(
                    frame, mapped, processed, inputs["fps"],
                    options.preview_line_thickness))
                processed += 1
                progress.update(processed,
                                f"detections={total_count}; tracks_started={tracker.next_identifier - 1}; "
                                f"inference_frames={inference_frames}")
        capture.release()
        capture = None
        writer.release()
        writer = None
        if processed != inputs["frame_count"]:
            raise ValueError(
                f"CFR decoder produced {processed} frames; Stage 1 declares "
                f"{inputs['frame_count']}")

        log("stage3", "Finalizing tracks, smoothing trajectories and exporting artifacts; frame ETA excluded this work.")
        all_tracks = tracker.all_tracks()
        observation_confirmed = [
            track for track in all_tracks
            if len(track.observations) >= options.min_confirmed_observations]
        static_person_tracks = []
        confirmed = []
        for track in observation_confirmed:
            points = np.asarray(
                [item["reference_anchor"] for item in track.observations],
                dtype=np.float64)
            maximum_motion = float(np.max(np.linalg.norm(points - points[0], axis=1)))
            if (track.kind == "person" and
                    maximum_motion < options.person_minimum_motion_px):
                static_person_tracks.append(track)
            else:
                confirmed.append(track)
        playback_tracks = [
            _playback_track(track, inputs["fps"], options, inputs["pixel_to_world"],
                            (inputs["width"], inputs["height"]))
            for track in confirmed
        ]
        tracks_payload = {
            "schema_version": STAGE3_SCHEMA,
            "fps": inputs["fps"],
            "frame_count": inputs["frame_count"],
            "duration_s": round(inputs["frame_count"] / inputs["fps"], 6),
            "ground_y": inputs["ground_y"],
            "scale_confidence": inputs["scale_confidence"],
            "tracks": playback_tracks,
        }
        (temporary / "tracks.json").write_text(
            json.dumps(tracks_payload, indent=2) + "\n", encoding="utf-8")
        _draw_track_map(inputs["ground"], confirmed, temporary / "track_map.png")

        preview_info = _video_info(preview_path)
        failures = []
        if preview_info["width"] != inputs["width"] or preview_info["height"] != inputs["height"]:
            failures.append("tracking preview dimensions do not match the CFR video")
        if preview_info["frame_count"] != inputs["frame_count"]:
            failures.append("tracking preview frame count does not match the CFR video")
        if abs(preview_info["fps"] - inputs["fps"]) > max(1e-3, inputs["fps"] * 1e-4):
            failures.append("tracking preview FPS does not match the CFR video")
        confirmed_observations = sum(len(track.observations) for track in confirmed)
        interpolated_samples = sum(
            sum(not sample["observed"] for sample in track["samples"])
            for track in playback_tracks)
        max_speeds = {
            kind: round(max(
                (sample["speed_units_per_s"] for track in playback_tracks
                 if track["kind"] == kind for sample in track["samples"]),
                default=0.0), 6)
            for kind in ("car", "person")
        }
        warnings = []
        if (detector.metadata.get("reused_detections") and
                not detector.metadata.get("retains_low_confidence_for_association")):
            warnings.append(
                "This detection cache predates low-confidence retention. Retracking "
                "cannot recover candidates removed by the old detector; run fresh "
                "inference to use the improved occlusion recovery.")
        if not playback_tracks:
            warnings.append(
                "No confirmed cars or people were found. This may be correct for an "
                "empty clip; inspect the preview before accepting it.")
        if inputs["metric_scale_is_assumed"]:
            warnings.append(
                "Stage 2 scale is assumed. Positions are valid in the chosen Unity "
                "coordinate system, but speeds and dimensions are not physical measurements.")
        if out_of_bounds_count:
            warnings.append(
                f"Excluded {out_of_bounds_count} detection(s) outside the reconstructed "
                "Stage 1 ground; actors enter and leave playback at the validated "
                "world boundary.")
        unconfirmed = len(all_tracks) - len(observation_confirmed)
        if unconfirmed:
            warnings.append(
                f"Discarded {unconfirmed} track fragment(s) with fewer than "
                f"{options.min_confirmed_observations} observations.")
        if static_person_tracks:
            warnings.append(
                f"Discarded {len(static_person_tracks)} stationary person candidate(s) "
                f"with less than {options.person_minimum_motion_px:g} reference-pixel "
                "motion; dynamic playback does not treat persistent static details as people.")

        report = {
            "schema_version": STAGE3_SCHEMA,
            "stage": 3,
            "pass": not failures,
            "failure_reasons": failures,
            "warnings": warnings,
            "source_stage1": {
                "bundle_path": str(inputs["stage1_root"]),
                "report_sha256": _sha256(inputs["stage1_report_path"]),
                "cfr_video_sha256": inputs["video_sha256"],
                "frame_transforms_sha256": inputs["transforms_sha256"],
                "source_video_sha256": inputs["stage1"]["input"].get("sha256"),
            },
            "source_stage2": {
                "bundle_path": str(inputs["stage2_root"]),
                "calibration_sha256": _sha256(inputs["calibration_path"]),
                "confidence": inputs["scale_confidence"],
                "metric_scale_is_assumed": inputs["metric_scale_is_assumed"],
            },
            "timeline": {
                "fps": inputs["fps"],
                "frame_count": inputs["frame_count"],
                "duration_s": round(inputs["frame_count"] / inputs["fps"], 6),
                "clock": "t = CFR frame index / fps",
            },
            "detector": detector.metadata,
            "tracker": {
                "implementation": "stabilized-reference-plane-association-v2",
                "association_space": "stage1_reference_image_pixels",
                "high_low_confidence_association": True,
                "offline_smoother": "constant_velocity_RTS",
                "smoothing_space": "stage1_reference_image_pixels",
                "smoothing_time_unit": "CFR_frame",
                "forward_backward_reconciliation": False,
            },
            "quality": {
                "processed_frames": processed,
                "detector_inference_frames": inference_frames,
                "detector_skipped_frames": processed - inference_frames,
                "detections": total_count,
                "detections_by_kind": dict(sorted(detection_counts.items())),
                "high_confidence_detections": high_count,
                "assigned_detections": assigned_count,
                "assigned_detection_pct": round(
                    100.0 * assigned_count / total_count, 3) if total_count else 100.0,
                "started_tracks": len(all_tracks),
                "confirmed_tracks": len(confirmed),
                "confirmed_tracks_by_kind": dict(sorted(Counter(
                    track.kind for track in confirmed).items())),
                "confirmed_observations": confirmed_observations,
                "interpolated_samples": interpolated_samples,
                "discarded_short_tracks": unconfirmed,
                "discarded_stationary_person_tracks": len(static_person_tracks),
                "max_speed_units_per_s_by_kind": max_speeds,
                "out_of_bounds_detections": out_of_bounds_count,
                "out_of_bounds_detections_sample": out_of_bounds_detections,
            },
            "coordinates": {
                "source_detection_space": "CFR frame pixels, top-left origin",
                "association_space": "Stage 1 reference-image pixels, top-left origin",
                "output_space": "Unity world X/Z",
                "ground_y": inputs["ground_y"],
                "scale_confidence": inputs["scale_confidence"],
            },
            "options": asdict(options),
            "limitations": [
                "The initial tracker is geometry-aware but has no project-specific ReID model.",
                "Axis-aligned detector boxes do not directly reveal a stopped actor's heading.",
                "Detector and tracker accuracy must be measured on held-out footage from the actual camera.",
            ],
            "outputs": {
                "report": "report.json",
                "detections": "detections.jsonl",
                "tracks": "tracks.json",
                "tracking_preview": "tracking_preview.mp4",
                "track_map": "track_map.png",
                "frame_counts": "frame_counts.csv",
            },
            "artifact_hashes": {
                name + "_sha256": _sha256(temporary / filename)
                for name, filename in {
                    "detections": "detections.jsonl", "tracks": "tracks.json",
                    "tracking_preview": "tracking_preview.mp4", "track_map": "track_map.png",
                    "frame_counts": "frame_counts.csv",
                }.items()
            },
        }
        (temporary / "report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")
        _publish_directory(temporary, output)
        log(
            "stage3",
            f"{'PASS' if report['pass'] else 'FAIL'}: {total_count} detections, "
            f"{len(confirmed)} confirmed tracks -> {output}")
        return report
    except Exception:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
