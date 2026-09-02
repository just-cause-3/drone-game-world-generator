"""Stage 1 ground extraction for stationary, nadir/top-down drone footage.

This intentionally does not perform object detection, world scaling, 3D inference, or
Unity deployment.  It registers a bounded amount of camera drift directly to one
reference frame, builds a temporal-median static background, and emits diagnostics that
prove whether the input satisfies this stage's assumptions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from collections import Counter
from contextlib import ExitStack
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import uuid
import warnings

import cv2
import numpy as np

from .common import log


SCHEMA_VERSION = "stage1-ground-v1"
INPUT_CONTRACT = "single_cut_free_stationary_top_down_video_with_minimal_hover_jitter_and_bounded_zoom"
ASSUMPTIONS = [
    "The operator has confirmed that the camera looks approximately straight down.",
    "The visible static scene is approximately planar.",
    "Dynamic actors are visible as background in enough frames or occupy only small repairable gaps.",
]
DEFAULT_FOREGROUND_MODEL = "models/visdrone-yolo26x.pt"
VISDRONE_FOREGROUND_CLASSES = (
    "pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle",
    "awning-tricycle", "bus", "motor", "others",
)
SUPPORTED_FOREGROUND_CLASS_NAMES = (
    "pedestrian", "people", "car",
    "person",  # alias for an explicitly selected COCO checkpoint
)

LIMITATIONS = [
    "Image registration verifies stability but cannot prove that the camera is top-down.",
    "The VisDrone detector can still miss extremely small, occluded, blurred, or domain-specific actors; inspect the masks and ground image.",
    "Stage 1 does not infer metric scale, terrain height, object height, or Unity geometry.",
]


@dataclass(frozen=True)
class Stage1Options:
    sample_frames: int = 41
    tile: int = 256
    orb_features: int = 6000
    ratio_test: float = 0.75
    ransac_px: float = 2.5
    min_matches: int = 40
    min_inliers: int = 30
    min_inlier_ratio: float = 0.45
    max_reprojection_px: float = 2.0
    min_frame_overlap_pct: float = 75.0
    max_center_motion_pct: float = 3.0
    max_corner_motion_pct: float = 12.0
    max_rotation_deg: float = 1.0
    max_scale_change_pct: float = 18.0
    frame_audit_orb_features: int = 2500
    min_frame_geometry_estimates_pct: float = 99.0
    max_frame_scale_step_pct: float = 1.0
    max_frame_center_step_pct: float = 0.5
    max_frame_rotation_step_deg: float = 0.25
    min_registered_pct: float = 90.0
    min_output_coverage_pct: float = 90.0
    min_observation_fraction: float = 0.30
    blur_threshold: float = 100.0
    max_blur_pct: float = 25.0
    timestamp_jitter_pct: float = 2.0
    foreground_model: str | None = DEFAULT_FOREGROUND_MODEL
    foreground_imgsz: int = 1280
    foreground_conf: float = 0.08
    residual_foreground_conf: float = 0.25
    foreground_iou: float = 0.45
    foreground_max_det: int = 3000
    foreground_class_names: tuple[str, ...] = SUPPORTED_FOREGROUND_CLASS_NAMES
    foreground_tile_size: int = 640
    foreground_tile_overlap: float = 0.25
    foreground_full_frame: bool = True
    foreground_tta: bool = True
    foreground_box_padding_fraction: float = 0.10
    foreground_dilate_px: int = 9
    semantic_consensus_fraction: float = 0.15
    min_semantic_consensus_frames: int = 2
    inpaint_radius_px: float = 5.0
    max_inpaint_component_pct: float = 0.75
    max_inpainted_pct: float = 2.0
    max_unresolved_hole_pct: float = 0.10
    max_residual_foreground_detections: int = 0
    residual_repair_passes: int = 2

    def validate(self):
        if self.sample_frames < 5:
            raise ValueError("sample_frames must be at least 5")
        if self.tile < 32:
            raise ValueError("tile must be at least 32 pixels")
        if self.max_center_motion_pct < 0:
            raise ValueError("max_center_motion_pct cannot be negative")
        if self.max_corner_motion_pct < 0:
            raise ValueError("max_corner_motion_pct cannot be negative")
        if self.max_scale_change_pct < 0:
            raise ValueError("max_scale_change_pct cannot be negative")
        if self.frame_audit_orb_features < 500:
            raise ValueError("frame_audit_orb_features must be at least 500")
        if not 0 < self.min_frame_geometry_estimates_pct <= 100:
            raise ValueError("min_frame_geometry_estimates_pct must be in (0, 100]")
        if self.max_frame_scale_step_pct <= 0:
            raise ValueError("max_frame_scale_step_pct must be positive")
        if self.max_frame_center_step_pct <= 0:
            raise ValueError("max_frame_center_step_pct must be positive")
        if self.max_frame_rotation_step_deg <= 0:
            raise ValueError("max_frame_rotation_step_deg must be positive")
        if not 0 < self.min_observation_fraction <= 1:
            raise ValueError("min_observation_fraction must be in (0, 1]")
        if self.foreground_dilate_px < 0:
            raise ValueError("foreground_dilate_px cannot be negative")
        if self.foreground_imgsz < 320:
            raise ValueError("foreground_imgsz must be at least 320")
        if not 0 < self.foreground_conf < 1:
            raise ValueError("foreground_conf must be in (0, 1)")
        if not 0 < self.residual_foreground_conf < 1:
            raise ValueError("residual_foreground_conf must be in (0, 1)")
        if not 0 < self.foreground_iou < 1:
            raise ValueError("foreground_iou must be in (0, 1)")
        if self.foreground_max_det < 1:
            raise ValueError("foreground_max_det must be at least 1")
        if self.foreground_tile_size != 0 and self.foreground_tile_size < 128:
            raise ValueError("foreground_tile_size must be 0 or at least 128")
        if not 0 <= self.foreground_tile_overlap < 0.9:
            raise ValueError("foreground_tile_overlap must be in [0, 0.9)")
        if not 0 <= self.foreground_box_padding_fraction <= 1:
            raise ValueError("foreground_box_padding_fraction must be in [0, 1]")
        if not self.foreground_class_names:
            raise ValueError("foreground_class_names cannot be empty")
        if not 0 < self.semantic_consensus_fraction <= 1:
            raise ValueError("semantic_consensus_fraction must be in (0, 1]")
        if self.min_semantic_consensus_frames < 1:
            raise ValueError("min_semantic_consensus_frames must be at least 1")
        if self.max_inpaint_component_pct < 0:
            raise ValueError("max_inpaint_component_pct cannot be negative")
        if self.residual_repair_passes < 0:
            raise ValueError("residual_repair_passes cannot be negative")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _video_timing(path: Path, jitter_pct: float):
    """Inspect actual presentation timestamps instead of trusting container FPS fields."""
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate,avg_frame_rate,width,height,duration",
        "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("ffprobe failed: " + result.stderr[-500:])
    data = json.loads(result.stdout)
    streams = data.get("streams") or []
    timestamps = [float(frame["best_effort_timestamp_time"])
                  for frame in data.get("frames", [])
                  if frame.get("best_effort_timestamp_time") is not None]
    if not streams or len(timestamps) < 2:
        raise RuntimeError("video has no usable timestamp sequence")
    deltas = np.diff(np.asarray(timestamps, np.float64))
    deltas = deltas[deltas > 0]
    if not len(deltas):
        raise RuntimeError("video timestamps are not increasing")
    median_dt = float(np.median(deltas))
    deviations = np.abs(deltas - median_dt) / median_dt * 100.0
    variable = bool(np.percentile(deviations, 99) > jitter_pct)
    return {
        "timestamp_count": len(timestamps),
        "median_frame_duration_s": median_dt,
        "measured_fps": 1.0 / median_dt,
        "timestamp_jitter_p99_pct": float(np.percentile(deviations, 99)),
        "variable_frame_rate": variable,
        "stream": streams[0],
    }


def _prepare_cfr(source: Path, temp_dir: Path, options: Stage1Options):
    timing = _video_timing(source, options.timestamp_jitter_pct)
    if not timing["variable_frame_rate"]:
        return source, timing, False
    normalized = temp_dir / "video_cfr.mp4"
    fps = timing["measured_fps"]
    command = [
        "ffmpeg", "-y", "-v", "error", "-i", str(source),
        "-vf", f"fps={fps:.9f}", "-an", "-c:v", "libx264", "-crf", "15",
        "-preset", "fast", "-pix_fmt", "yuv420p", str(normalized),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("CFR normalization failed: " + result.stderr[-500:])
    return normalized, timing, True


def _detect_scene_cuts(path: Path):
    try:
        from scenedetect import AdaptiveDetector, detect
    except ImportError as exc:
        raise RuntimeError("scenedetect is required for Stage 1") from exc
    scenes = detect(str(path), AdaptiveDetector(), show_progress=False)
    if len(scenes) <= 1:
        return []
    return [round(float(scene[1].get_seconds()), 4) for scene in scenes[:-1]]


def _video_info(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    try:
        info = {
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": float(cap.get(cv2.CAP_PROP_FPS)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        cap.release()
    if any(info[key] <= 0 for key in info):
        raise RuntimeError("video metadata is incomplete or invalid")
    return info


def _normalized_class_name(name: str) -> str:
    return str(name).strip().lower().replace("_", "-").replace(" ", "-")


def _tile_origins(span: int, tile_size: int, overlap: float) -> list[int]:
    """Return deterministic crop origins with complete edge coverage."""
    if tile_size <= 0 or tile_size >= span:
        return [0]
    stride = max(1, int(round(tile_size * (1.0 - overlap))))
    last = span - tile_size
    origins = list(range(0, last + 1, stride))
    if origins[-1] != last:
        origins.append(last)
    return origins


def _class_agnostic_nms(candidates: list[dict], iou: float) -> list[dict]:
    """Merge duplicate full-frame/tile detections before building the actor mask."""
    if not candidates:
        return []
    boxes = []
    scores = []
    for candidate in candidates:
        x1, y1, x2, y2 = candidate["bbox_xyxy"]
        boxes.append([float(x1), float(y1), max(0.0, float(x2 - x1)),
                      max(0.0, float(y2 - y1))])
        scores.append(float(candidate["confidence"]))
    indices = cv2.dnn.NMSBoxes(boxes, scores, 0.0, float(iou))
    if indices is None or len(indices) == 0:
        return []
    flattened = np.asarray(indices).reshape(-1)
    return [candidates[int(index)] for index in flattened]


def _detection_preview(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    preview = frame.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox_xyxy"]
        color = (30, 220, 255)
        cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)
        label = f'{detection["class"]} {detection["confidence"]:.2f}'
        cv2.putText(preview, label, (x1, max(14, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return preview


class _YoloForegroundMasker:
    def __init__(self, options: Stage1Options):
        from ultralytics import YOLO
        from .track_agents import resolve_device

        self.options = options
        self.device = resolve_device("auto")
        requested = Path(str(options.foreground_model))
        repository_model = Path(__file__).resolve().parents[1] / requested
        model_source = (repository_model if not requested.is_absolute() and
                        repository_model.exists() else requested)
        if (str(options.foreground_model) == DEFAULT_FOREGROUND_MODEL and
                not model_source.exists()):
            raise FileNotFoundError(
                f"drone detector is missing: {repository_model}. "
                "Run '.venv\\Scripts\\python.exe setup_drone_model.py' once.")
        log("stage1", f"foreground model={model_source} device={self.device}; "
                      f"full_frame={options.foreground_full_frame}, "
                      f"tile={options.foreground_tile_size}, tta={options.foreground_tta}")
        self.model_source = str(model_source)
        self.model = YOLO(self.model_source)
        raw_names = self.model.names
        self.model_names = {
            int(class_id): str(name) for class_id, name in
            (raw_names.items() if isinstance(raw_names, dict) else enumerate(raw_names))
        }
        self.selected_names = {
            _normalized_class_name(name) for name in options.foreground_class_names
        }
        available = {_normalized_class_name(name) for name in self.model_names.values()}
        if not self.selected_names.intersection(available):
            raise ValueError(
                "foreground model has no requested actor classes; "
                f"model classes={sorted(available)}")

    @property
    def metadata(self):
        path = Path(self.model_source)
        return {
            "model": self.model_source,
            "model_sha256": _file_sha256(path) if path.is_file() else None,
            "device": str(self.device),
            "model_classes": self.model_names,
            "selected_classes": sorted(self.selected_names),
            "confidence": self.options.foreground_conf,
            "residual_confidence": self.options.residual_foreground_conf,
            "iou": self.options.foreground_iou,
            "imgsz": self.options.foreground_imgsz,
            "full_frame": self.options.foreground_full_frame,
            "tile_size": self.options.foreground_tile_size,
            "tile_overlap": self.options.foreground_tile_overlap,
            "test_time_augmentation": self.options.foreground_tta,
        }

    def _predict(self, image: np.ndarray, offset_x: int, offset_y: int,
                 inference_pass: str, confidence: float) -> list[dict]:
        result = self.model.predict(
            image, imgsz=self.options.foreground_imgsz,
            conf=confidence, iou=self.options.foreground_iou,
            max_det=self.options.foreground_max_det, device=self.device,
            augment=self.options.foreground_tta, agnostic_nms=True,
            end2end=False, verbose=False)[0]
        candidates = []
        if result.boxes is None:
            return candidates
        for box, cls, confidence in zip(result.boxes.xyxy.cpu().numpy(),
                                        result.boxes.cls.cpu().numpy(),
                                        result.boxes.conf.cpu().numpy()):
            class_id = int(cls)
            class_name = self.model_names.get(class_id, str(class_id))
            if _normalized_class_name(class_name) not in self.selected_names:
                continue
            x1, y1, x2, y2 = (float(value) for value in box)
            candidates.append({
                "class_id": class_id,
                "class": class_name,
                "confidence": float(confidence),
                "bbox_xyxy": [x1 + offset_x, y1 + offset_y,
                              x2 + offset_x, y2 + offset_y],
                "inference_pass": inference_pass,
            })
        return candidates

    def mask(self, frame: np.ndarray, confidence: float | None = None):
        height, width = frame.shape[:2]
        confidence = self.options.foreground_conf if confidence is None else confidence
        candidates = []
        if self.options.foreground_full_frame:
            candidates.extend(self._predict(frame, 0, 0, "full", confidence))
        tile_size = self.options.foreground_tile_size
        tiled = tile_size > 0 and (width > tile_size or height > tile_size)
        if tiled:
            tile_width = min(tile_size, width)
            tile_height = min(tile_size, height)
            x_origins = _tile_origins(width, tile_width,
                                      self.options.foreground_tile_overlap)
            y_origins = _tile_origins(height, tile_height,
                                      self.options.foreground_tile_overlap)
            for y0 in y_origins:
                for x0 in x_origins:
                    crop = frame[y0:y0 + tile_height, x0:x0 + tile_width]
                    candidates.extend(self._predict(
                        crop, x0, y0, f"tile:{x0},{y0}", confidence))
        if not self.options.foreground_full_frame and not tiled:
            candidates.extend(self._predict(frame, 0, 0, "single", confidence))

        merged = _class_agnostic_nms(candidates, self.options.foreground_iou)
        mask = np.zeros(frame.shape[:2], np.uint8)
        detections = []
        for candidate in merged:
            x1, y1, x2, y2 = candidate["bbox_xyxy"]
            pad_x = (x2 - x1) * self.options.foreground_box_padding_fraction
            pad_y = (y2 - y1) * self.options.foreground_box_padding_fraction
            x1 = max(0, int(math.floor(x1 - pad_x)))
            y1 = max(0, int(math.floor(y1 - pad_y)))
            x2 = min(width - 1, int(math.ceil(x2 + pad_x)))
            y2 = min(height - 1, int(math.ceil(y2 + pad_y)))
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            detections.append({
                "class_id": candidate["class_id"],
                "class": candidate["class"],
                "confidence": round(candidate["confidence"], 4),
                "bbox_xyxy": [x1, y1, x2, y2],
            })
        radius = self.options.foreground_dilate_px
        if radius and np.any(mask):
            size = 2 * radius + 1
            mask = cv2.dilate(mask, np.ones((size, size), np.uint8))
        return mask, detections


def _interpolate_frame_transforms(frame_count: int, sample_indices: np.ndarray,
                                  transforms: np.ndarray, accepted: np.ndarray):
    """Interpolate similarity parameters onto every CFR frame for later stages."""
    slots = np.flatnonzero(accepted)
    if len(slots) < 2:
        raise ValueError("at least two accepted transforms are needed for interpolation")
    known_frames = sample_indices[slots].astype(np.float64)
    query = np.arange(frame_count, dtype=np.float64)
    values = np.column_stack([
        transforms[slots, 0, 0], transforms[slots, 1, 0],
        transforms[slots, 0, 2], transforms[slots, 1, 2],
    ])
    interpolated = np.column_stack([
        np.interp(query, known_frames, values[:, column]) for column in range(4)
    ])
    result = np.zeros((frame_count, 3, 3), np.float64)
    result[:, 0, 0] = interpolated[:, 0]
    result[:, 0, 1] = -interpolated[:, 1]
    result[:, 1, 0] = interpolated[:, 1]
    result[:, 1, 1] = interpolated[:, 0]
    result[:, 0, 2] = interpolated[:, 2]
    result[:, 1, 2] = interpolated[:, 3]
    result[:, 2, 2] = 1.0
    return result


def _inpaint_small_holes(image: np.ndarray, holes: np.ndarray,
                         options: Stage1Options):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8), 8)
    eligible = np.zeros_like(holes, np.uint8)
    max_area = image.shape[0] * image.shape[1] * options.max_inpaint_component_pct / 100.0
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] <= max_area:
            eligible[labels == label] = 255
    repaired = image
    if np.any(eligible):
        repaired = cv2.inpaint(image, eligible, options.inpaint_radius_px, cv2.INPAINT_TELEA)
    unresolved = holes & (eligible == 0)
    return repaired, eligible, unresolved


def _semantic_exclusion(foreground: np.ndarray, accepted: np.ndarray,
                        options: Stage1Options):
    """Find repeatedly detected actors that a median could otherwise preserve."""
    accepted_count = int(np.count_nonzero(accepted))
    required = max(options.min_semantic_consensus_frames,
                   int(math.ceil(accepted_count * options.semantic_consensus_fraction)))
    if accepted_count < required:
        return np.zeros(foreground.shape[1:], bool), required
    detections = np.asarray(foreground[accepted], dtype=bool)
    return np.count_nonzero(detections, axis=0) >= required, required


def _publish_directory(staging: Path, output: Path):
    """Replace a generated output directory with rollback if the rename fails."""
    backup = output.with_name(f".{output.name}.backup-{uuid.uuid4().hex}")
    had_output = output.exists()
    if had_output:
        if not output.is_dir():
            raise OSError(f"output path exists and is not a directory: {output}")
        os.replace(output, backup)
    try:
        os.replace(staging, output)
    except Exception:
        if had_output and backup.exists():
            os.replace(backup, output)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def _validate_output_target(source: Path, output: Path):
    """Refuse broad or unrelated directories before atomic replacement."""
    if output == Path(output.anchor):
        raise ValueError("Stage 1 output cannot be a filesystem root")
    if source == output or (source.exists() and source.is_relative_to(output)):
        raise ValueError("Stage 1 output cannot contain the input video")
    if not output.exists():
        return
    if not output.is_dir():
        raise ValueError(f"Stage 1 output exists and is not a directory: {output}")
    if not any(output.iterdir()):
        return
    marker = output / ".stage1-output"
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() == SCHEMA_VERSION:
        return
    report_path = output / "report.json"
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            report = {}
        if report.get("schema_version") == SCHEMA_VERSION:
            return
    raise ValueError(
        f"refusing to replace nonempty directory that is not a {SCHEMA_VERSION} bundle: {output}")
