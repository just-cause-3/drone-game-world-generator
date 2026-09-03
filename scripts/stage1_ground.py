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


def _sample_indices(frame_count: int, requested: int) -> np.ndarray:
    if frame_count < 5:
        raise ValueError(f"video is too short: only {frame_count} frames")
    return np.unique(np.linspace(0, frame_count - 1, min(frame_count, requested)).astype(int))


def _decode_samples(video: Path, indices: np.ndarray, storage: np.memmap):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video}")
    wanted = {int(frame_number): slot for slot, frame_number in enumerate(indices)}
    decoded = 0
    frame_number = 0
    try:
        while wanted:
            ok, frame = cap.read()
            if not ok:
                break
            slot = wanted.pop(frame_number, None)
            if slot is not None:
                if frame.shape != storage.shape[1:]:
                    raise RuntimeError("video dimensions changed while decoding")
                storage[slot] = frame
                decoded += 1
            frame_number += 1
    finally:
        cap.release()
    storage.flush()
    if wanted:
        missing = sorted(wanted)[:5]
        raise RuntimeError(f"failed to decode sampled video frames: {missing}")
    return decoded


def _sharpness(frame: np.ndarray, valid_mask: np.ndarray | None = None) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    if valid_mask is None:
        return float(laplacian.var())
    interior = cv2.erode(valid_mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    values = laplacian[interior]
    return float(values.var()) if values.size else 0.0


def _reference_slot(frames: np.memmap, blur_threshold: float) -> int:
    """Use the temporal center unless it is genuinely blurred.

    Picking the absolute sharpest frame biased the reference toward one end of clips
    with a slow altitude change, unnecessarily reducing overlap at the opposite end.
    """
    count = len(frames)
    center = count // 2
    center_sharpness = _sharpness(np.asarray(frames[center]))
    if center_sharpness >= blur_threshold:
        return center
    radius = max(1, count // 10)
    candidates = range(max(0, center - radius), min(count, center + radius + 1))
    return max(candidates, key=lambda i: _sharpness(np.asarray(frames[i])))


def _features(frame: np.ndarray, options: Stage1Options):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.ORB_create(options.orb_features).detectAndCompute(gray, None)


def _registration(frame: np.ndarray, reference: np.ndarray, options: Stage1Options,
                  reference_features=None):
    """Estimate a constrained similarity transform from frame -> reference."""
    kp, desc = _features(frame, options)
    ref_kp, ref_desc = reference_features or _features(reference, options)
    if desc is None or ref_desc is None:
        return None, {"accepted": False, "reason": "insufficient_features"}

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(desc, ref_desc, k=2)
    good = [first for pair in pairs if len(pair) == 2
            for first, second in [pair]
            if first.distance < options.ratio_test * second.distance]
    if len(good) < options.min_matches:
        return None, {"accepted": False, "reason": "insufficient_matches",
                      "matches": len(good)}

    source = np.float32([kp[m.queryIdx].pt for m in good])
    target = np.float32([ref_kp[m.trainIdx].pt for m in good])
    affine, mask = cv2.estimateAffinePartial2D(
        source, target, method=cv2.RANSAC,
        ransacReprojThreshold=options.ransac_px,
        maxIters=5000, confidence=0.999, refineIters=20)
    if affine is None or mask is None:
        return None, {"accepted": False, "reason": "transform_failed",
                      "matches": len(good)}

    inlier_mask = mask.ravel().astype(bool)
    inliers = int(inlier_mask.sum())
    ratio = inliers / len(good)
    H = np.vstack([affine, [0.0, 0.0, 1.0]]).astype(np.float64)
    projected = cv2.transform(source[inlier_mask, None, :], affine)[:, 0]
    reprojection = float(np.median(np.linalg.norm(projected - target[inlier_mask], axis=1)))

    height, width = frame.shape[:2]
    corners = np.float32([[[0, 0]], [[width - 1, 0]],
                          [[width - 1, height - 1]], [[0, height - 1]]])
    moved = cv2.perspectiveTransform(corners, H)
    diagonal = math.hypot(width, height)
    corner_motion_pct = 100.0 * float(
        np.max(np.linalg.norm(moved[:, 0] - corners[:, 0], axis=1))) / diagonal
    center = np.float32([[[0.5 * (width - 1), 0.5 * (height - 1)]]])
    moved_center = cv2.perspectiveTransform(center, H)
    center_motion_pct = 100.0 * float(
        np.linalg.norm(moved_center[0, 0] - center[0, 0])) / diagonal
    valid = cv2.warpPerspective(np.full((height, width), 255, np.uint8), H,
                                (width, height), flags=cv2.INTER_NEAREST)
    overlap_pct = 100.0 * float(np.mean(valid > 0))
    scale = float(math.hypot(affine[0, 0], affine[1, 0]))
    rotation_deg = float(math.degrees(math.atan2(affine[1, 0], affine[0, 0])))
    scale_change_pct = 100.0 * abs(scale - 1.0)

    checks = [
        (inliers >= options.min_inliers, "too_few_inliers"),
        (ratio >= options.min_inlier_ratio, "low_inlier_ratio"),
        (reprojection <= options.max_reprojection_px, "high_reprojection_error"),
        (abs(rotation_deg) <= options.max_rotation_deg, "camera_rotated_too_far"),
        (center_motion_pct <= options.max_center_motion_pct, "camera_moved_too_far"),
        (scale_change_pct <= options.max_scale_change_pct, "camera_zoomed_too_far"),
        (corner_motion_pct <= options.max_corner_motion_pct, "frame_transform_too_large"),
        (overlap_pct >= options.min_frame_overlap_pct, "low_frame_overlap"),
    ]
    failure = next((reason for ok, reason in checks if not ok), None)
    metrics = {
        "accepted": failure is None,
        "reason": failure,
        "matches": len(good),
        "inliers": inliers,
        "inlier_ratio": round(ratio, 4),
        "median_reprojection_px": round(reprojection, 4),
        "frame_overlap_pct": round(overlap_pct, 3),
        "center_motion_pct": round(center_motion_pct, 3),
        "corner_motion_pct": round(corner_motion_pct, 3),
        "rotation_deg": round(rotation_deg, 4),
        "scale_factor": round(scale, 7),
        "scale_change_pct": round(scale_change_pct, 4),
    }
    return (H if failure is None else None), metrics


def _identity_metrics():
    return {
        "accepted": True,
        "reason": None,
        "matches": None,
        "inliers": None,
        "inlier_ratio": 1.0,
        "median_reprojection_px": 0.0,
        "frame_overlap_pct": 100.0,
        "center_motion_pct": 0.0,
        "corner_motion_pct": 0.0,
        "rotation_deg": 0.0,
        "scale_factor": 1.0,
        "scale_change_pct": 0.0,
    }


def _frame_geometry_audit(path: Path, fps: float, options: Stage1Options):
    """Measure every consecutive frame so jumps cannot hide between samples."""
    audit_options = replace(
        options,
        orb_features=options.frame_audit_orb_features,
        min_matches=min(options.min_matches, 25),
        min_inliers=min(options.min_inliers, 18),
        min_frame_overlap_pct=0.0,
        max_center_motion_pct=100.0,
        max_corner_motion_pct=100.0,
        max_rotation_deg=180.0,
        max_scale_change_pct=100.0,
    )
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video for frame geometry audit: {path}")
    ok, previous = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("could not decode first frame for geometry audit")
    previous_features = _features(previous, audit_options)
    pairs = []
    frame_number = 1
    try:
        while True:
            ok, current = cap.read()
            if not ok:
                break
            _, metrics = _registration(
                current, previous, audit_options, previous_features)
            metrics["from_frame"] = frame_number - 1
            metrics["to_frame"] = frame_number
            metrics["time_s"] = round(frame_number / fps, 6)
            pairs.append(metrics)
            previous = current
            previous_features = _features(previous, audit_options)
            frame_number += 1
    finally:
        cap.release()

    trusted = [metrics for metrics in pairs if metrics.get("accepted")]
    estimated_pct = 100.0 * len(trusted) / len(pairs) if pairs else 0.0

    def values(name, absolute=False):
        result = [float(metrics[name]) for metrics in trusted if name in metrics]
        return [abs(value) for value in result] if absolute else result

    scale_steps = values("scale_change_pct")
    center_steps = values("center_motion_pct")
    rotation_steps = values("rotation_deg", absolute=True)
    summary = {
        "frame_pairs": len(pairs),
        "estimated_pairs": len(trusted),
        "estimated_pct": round(estimated_pct, 4),
        "max_scale_step_pct": round(max(scale_steps, default=0.0), 4),
        "p95_scale_step_pct": round(float(np.percentile(scale_steps, 95)), 4)
        if scale_steps else None,
        "max_center_step_pct": round(max(center_steps, default=0.0), 4),
        "p95_center_step_pct": round(float(np.percentile(center_steps, 95)), 4)
        if center_steps else None,
        "max_rotation_step_deg": round(max(rotation_steps, default=0.0), 4),
        "failure_reasons": dict(sorted(Counter(
            metrics.get("reason") or "unknown" for metrics in pairs
            if not metrics.get("accepted")).items())),
    }
    summary["smooth"] = bool(
        estimated_pct >= options.min_frame_geometry_estimates_pct and
        summary["max_scale_step_pct"] <= options.max_frame_scale_step_pct and
        summary["max_center_step_pct"] <= options.max_frame_center_step_pct and
        summary["max_rotation_step_deg"] <= options.max_frame_rotation_step_deg
    )
    return {"summary": summary, "pairs": pairs}


def _build_background(frames: np.memmap, valid: np.memmap, foreground: np.memmap,
                      accepted: np.ndarray, options: Stage1Options):
    height, width = frames.shape[1:3]
    ground = np.zeros((height, width, 4), np.uint8)
    heat = np.zeros((height, width), np.float32)
    holes = np.zeros((height, width), bool)
    foreground_frequency = np.zeros((height, width), np.float32)
    accepted_slots = np.flatnonzero(accepted)
    minimum = max(3, int(math.ceil(len(accepted_slots) * options.min_observation_fraction)))
    semantic_exclusion, semantic_consensus_frames = _semantic_exclusion(
        foreground, accepted, options)

    for y0 in range(0, height, options.tile):
        y1 = min(height, y0 + options.tile)
        for x0 in range(0, width, options.tile):
            x1 = min(width, x0 + options.tile)
            pixels = np.asarray(frames[accepted_slots, y0:y1, x0:x1], dtype=np.float32)
            geometric = np.asarray(valid[accepted_slots, y0:y1, x0:x1], dtype=bool)
            actors = np.asarray(foreground[accepted_slots, y0:y1, x0:x1], dtype=bool)
            # A pixel occupied in several sampled frames is not necessarily a static
            # actor: dense traffic repeatedly crosses the same road pixels. Keep the
            # consensus map as review evidence, but reconstruct from every genuinely
            # actor-free observation. A persistent actor naturally leaves too few
            # observations, while a missed one is caught by the residual detector.
            observed = geometric & ~actors
            masked = np.where(observed[..., None], pixels, np.nan)
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.filterwarnings("ignore", message="All-NaN slice encountered",
                                        category=RuntimeWarning)
                median = np.nanmedian(masked, axis=0)
                deviation = np.nanmedian(np.abs(masked - median[None, ...]), axis=(0, 3))
            enough_geometry = geometric.sum(axis=0) >= minimum
            enough_background = observed.sum(axis=0) >= minimum
            median = np.nan_to_num(median, nan=0.0)
            ground[y0:y1, x0:x1, :3] = np.clip(median, 0, 255).astype(np.uint8)
            ground[y0:y1, x0:x1, 3] = np.where(enough_geometry, 255, 0).astype(np.uint8)
            heat[y0:y1, x0:x1] = np.nan_to_num(deviation, nan=0.0)
            holes[y0:y1, x0:x1] = enough_geometry & ~enough_background
            foreground_frequency[y0:y1, x0:x1] = actors.mean(axis=0)

    return (ground, heat, holes, foreground_frequency, semantic_exclusion,
            minimum, semantic_consensus_frames)


def _write_image(path: Path, image: np.ndarray):
    if not cv2.imwrite(str(path), image):
        raise OSError(f"failed to write image: {path}")


def _run_stage1(video, output_dir, options: Stage1Options):
    options.validate()
    source = Path(video).resolve()
    output = Path(output_dir).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"video not found: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)

    # ExitStack closes Windows memory maps before TemporaryDirectory removes them.
    # Relying on garbage collection here makes an unrelated processing exception get
    # replaced by a confusing WinError 32 during cleanup.
    with tempfile.TemporaryDirectory(prefix=".drone-stage1-", dir=output.parent) as temp_name, \
            ExitStack() as cleanup:
        temp_root = Path(temp_name)
        result_dir = temp_root / "result"
        result_dir.mkdir()
        work_source, timing, normalized = _prepare_cfr(source, temp_root, options)
        info = _video_info(work_source)
        frame_count, fps = info["frame_count"], info["fps"]
        width, height = info["width"], info["height"]
        cuts = _detect_scene_cuts(work_source)
        indices = _sample_indices(frame_count, options.sample_frames)
        log("stage1", f"{source.name}: {width}x{height}, {fps:.3f} fps, "
                      f"{frame_count} frames; sampling {len(indices)}")
        frame_audit = _frame_geometry_audit(work_source, fps, options)
        audit_summary = frame_audit["summary"]
        log("stage1", f"frame audit {audit_summary['estimated_pairs']}/"
                      f"{audit_summary['frame_pairs']}; max scale step="
                      f"{audit_summary['max_scale_step_pct']:.3f}%; max center step="
                      f"{audit_summary['max_center_step_pct']:.3f}%")

        raw_path = temp_root / "registered.dat"
        valid_path = temp_root / "valid.dat"
        foreground_path = temp_root / "foreground.dat"
        frames = np.memmap(raw_path, np.uint8, mode="w+",
                           shape=(len(indices), height, width, 3))
        cleanup.callback(frames._mmap.close)
        valid = np.memmap(valid_path, np.uint8, mode="w+",
                          shape=(len(indices), height, width))
        cleanup.callback(valid._mmap.close)
        foreground = np.memmap(foreground_path, np.uint8, mode="w+",
                               shape=(len(indices), height, width))
        cleanup.callback(foreground._mmap.close)
        foreground[:] = 0
        _decode_samples(work_source, indices, frames)

        sharpness = [_sharpness(np.asarray(frames[i])) for i in range(len(indices))]
        normalized_sharpness = np.full(len(indices), np.nan, np.float64)
        reference_slot = _reference_slot(frames, options.blur_threshold)
        reference = np.asarray(frames[reference_slot]).copy()
        reference_features = _features(reference, options)
        masker = _YoloForegroundMasker(options) if options.foreground_model else None
        transforms = np.full((len(indices), 3, 3), np.nan, np.float64)
        accepted = np.zeros(len(indices), bool)
        registrations = []
        detected_classes = Counter()
        reference_detections = []
        reference_foreground_mask = np.zeros((height, width), np.uint8)

        for slot, frame_number in enumerate(indices):
            source_frame = np.asarray(frames[slot]).copy()
            if slot == reference_slot:
                H, metrics = np.eye(3, dtype=np.float64), _identity_metrics()
            else:
                H, metrics = _registration(source_frame, reference, options,
                                           reference_features)
            metrics["sample_slot"] = slot
            metrics["video_frame"] = int(frame_number)
            metrics["time_s"] = round(float(frame_number / fps), 4)
            metrics["sharpness"] = round(sharpness[slot], 3)
            registrations.append(metrics)
            if H is None:
                valid[slot] = 0
                continue
            transforms[slot] = H
            accepted[slot] = True
            actor_mask = np.zeros((height, width), np.uint8)
            detections = []
            if masker:
                actor_mask, detections = masker.mask(source_frame)
                detected_classes.update(detection["class"] for detection in detections)
                if slot == reference_slot:
                    reference_detections = detections
                    reference_foreground_mask = actor_mask.copy()
            metrics["foreground_detections"] = len(detections)
            metrics["foreground_by_class"] = dict(sorted(Counter(
                detection["class"] for detection in detections).items()))
            aligned = cv2.warpPerspective(
                source_frame, H, (width, height), flags=cv2.INTER_LANCZOS4)
            aligned_valid = cv2.warpPerspective(
                np.full((height, width), 255, np.uint8), H, (width, height),
                flags=cv2.INTER_NEAREST)
            normalized_sharpness[slot] = _sharpness(aligned, aligned_valid)
            metrics["normalized_sharpness"] = round(normalized_sharpness[slot], 3)
            metrics["sharpness_retention_pct"] = round(
                100.0 * normalized_sharpness[slot] / max(sharpness[slot], 1e-9), 3)
            frames[slot] = aligned
            valid[slot] = aligned_valid
            foreground[slot] = cv2.warpPerspective(
                actor_mask, H, (width, height), flags=cv2.INTER_NEAREST)

        frames.flush()
        valid.flush()
        foreground.flush()
        registered = int(accepted.sum())
        registered_pct = 100.0 * registered / len(indices)
        blur_pct = 100.0 * float(np.mean(np.asarray(sharpness) < options.blur_threshold))
        accepted_normalized_sharpness = normalized_sharpness[accepted]
        normalized_blur_pct = 100.0 * float(np.mean(
            accepted_normalized_sharpness < options.blur_threshold))
        sharpness_retention = np.asarray([
            float(metrics["sharpness_retention_pct"])
            for metrics in registrations if metrics.get("accepted")
        ], np.float64)
        accepted_metrics = [r for r in registrations if r["accepted"]]
        max_motion = max(float(r["corner_motion_pct"]) for r in accepted_metrics)
        max_center_motion = max(float(r["center_motion_pct"]) for r in accepted_metrics)
        max_rotation = max(abs(float(r["rotation_deg"])) for r in accepted_metrics)
        max_scale = max(float(r["scale_change_pct"]) for r in accepted_metrics)
        reprojections = [float(r["median_reprojection_px"]) for r in accepted_metrics
                         if r["matches"] is not None]
        rejected_reasons = Counter(r["reason"] for r in registrations if not r["accepted"])

        failures = []
        if cuts:
            failures.append(f"detected {len(cuts)} scene cut(s): {cuts}")
        if audit_summary["estimated_pct"] < options.min_frame_geometry_estimates_pct:
            failures.append(
                f"frame geometry measured {audit_summary['estimated_pct']:.1f}% < "
                f"{options.min_frame_geometry_estimates_pct:.1f}%")
        if audit_summary["max_scale_step_pct"] > options.max_frame_scale_step_pct:
            failures.append(
                f"sudden zoom step {audit_summary['max_scale_step_pct']:.3f}% > "
                f"{options.max_frame_scale_step_pct:.3f}%")
        if audit_summary["max_center_step_pct"] > options.max_frame_center_step_pct:
            failures.append(
                f"sudden center movement {audit_summary['max_center_step_pct']:.3f}% > "
                f"{options.max_frame_center_step_pct:.3f}%")
        if audit_summary["max_rotation_step_deg"] > options.max_frame_rotation_step_deg:
            failures.append(
                f"sudden rotation step {audit_summary['max_rotation_step_deg']:.3f} deg > "
                f"{options.max_frame_rotation_step_deg:.3f} deg")
        if registered_pct < options.min_registered_pct:
            failures.append(f"registered {registered_pct:.1f}% < {options.min_registered_pct:.1f}%")
        if blur_pct > options.max_blur_pct:
            failures.append(f"blurry samples {blur_pct:.1f}% > {options.max_blur_pct:.1f}%")
        if normalized_blur_pct > options.max_blur_pct:
            failures.append(
                f"blurry normalized samples {normalized_blur_pct:.1f}% > "
                f"{options.max_blur_pct:.1f}%")

        final_names = {
            "ground": "ground.png",
            "valid_mask": "valid_mask.png",
            "temporal_variation": "temporal_variation.png",
            "foreground_frequency": "foreground_frequency.png",
            "detection_preview": "detection_preview.png",
            "reference_foreground_mask": "reference_foreground_mask.png",
            "semantic_exclusion": "semantic_exclusion_mask.png",
            "inpaint_mask": "inpaint_mask.png",
            "sample_transforms": "sample_transforms_to_reference.npy",
            "frame_transforms": "frame_transforms_to_reference.npy",
            "registration": "registration.json",
            "frame_geometry": "frame_geometry.json",
            "report": "report.json",
            "video_cfr": f"video_cfr{work_source.suffix.lower() or '.mp4'}",
        }
        # Paths inside the artifact manifest are relative to the directory containing
        # report.json, so a completed Stage 1 bundle can be moved as a unit.
        output_paths = dict(final_names)
        np.save(result_dir / final_names["sample_transforms"], transforms)
        registration_path = result_dir / final_names["registration"]
        registration_path.write_text(json.dumps(registrations, indent=2), encoding="utf-8")
        (result_dir / final_names["frame_geometry"]).write_text(
            json.dumps(frame_audit, indent=2), encoding="utf-8")
        shutil.copy2(work_source, result_dir / final_names["video_cfr"])
        if masker:
            _write_image(result_dir / final_names["detection_preview"],
                         _detection_preview(reference, reference_detections))
            _write_image(result_dir / final_names["reference_foreground_mask"],
                         reference_foreground_mask)

        coverage_pct = 0.0
        inpainted_pct = 0.0
        unresolved_pct = 0.0
        foreground_observation_pct = 0.0
        semantic_exclusion_pct = 0.0
        semantic_consensus_frames = 0
        residual_detections = []
        residual_detections_before_repair = []
        residual_repair_pct = 0.0
        minimum_observations = 0
        if registered >= 3:
            frame_transforms = _interpolate_frame_transforms(
                frame_count, indices, transforms, accepted)
            np.save(result_dir / final_names["frame_transforms"], frame_transforms)
            (ground, heat, holes, frequency, semantic_exclusion,
             minimum_observations, semantic_consensus_frames) = _build_background(
                frames, valid, foreground, accepted, options)
            repaired, inpaint_mask, unresolved = _inpaint_small_holes(
                ground[:, :, :3], holes, options)
            ground[:, :, :3] = repaired
            ground[unresolved, 3] = 0
            foreground_observation_pct = 100.0 * float(np.mean(frequency))
            semantic_exclusion_pct = 100.0 * float(np.mean(semantic_exclusion))
            if masker:
                residual_mask, residual_detections = masker.mask(
                    ground[:, :, :3], options.residual_foreground_conf)
                residual_detections_before_repair = list(residual_detections)
                residual_repair_mask = np.zeros_like(residual_mask)
                for _ in range(options.residual_repair_passes):
                    if not residual_detections:
                        break
                    repaired, eligible, residual_unresolved = _inpaint_small_holes(
                        ground[:, :, :3], residual_mask > 0, options)
                    if not np.any(eligible):
                        unresolved |= residual_unresolved
                        break
                    ground[:, :, :3] = repaired
                    residual_repair_mask = np.maximum(residual_repair_mask, eligible)
                    inpaint_mask = np.maximum(inpaint_mask, eligible)
                    unresolved |= residual_unresolved
                    ground[unresolved, 3] = 0
                    residual_mask, residual_detections = masker.mask(
                        ground[:, :, :3], options.residual_foreground_conf)
                residual_repair_pct = 100.0 * float(
                    np.mean(residual_repair_mask > 0))

            coverage = ground[:, :, 3] > 0
            coverage_pct = 100.0 * float(np.mean(coverage))
            inpainted_pct = 100.0 * float(np.mean(inpaint_mask > 0))
            unresolved_pct = 100.0 * float(np.mean(unresolved))

            _write_image(result_dir / final_names["ground"], ground)
            _write_image(result_dir / final_names["valid_mask"], ground[:, :, 3])
            valid_heat = heat[coverage]
            heat_limit = max(1.0, float(np.percentile(valid_heat, 99))) if valid_heat.size else 1.0
            heat_u8 = np.clip(255.0 * heat / heat_limit, 0, 255).astype(np.uint8)
            _write_image(result_dir / final_names["temporal_variation"], heat_u8)
            _write_image(result_dir / final_names["foreground_frequency"],
                         np.clip(frequency * 255.0, 0, 255).astype(np.uint8))
            _write_image(result_dir / final_names["semantic_exclusion"],
                         semantic_exclusion.astype(np.uint8) * 255)
            _write_image(result_dir / final_names["inpaint_mask"], inpaint_mask)

            if coverage_pct < options.min_output_coverage_pct:
                failures.append(f"output coverage {coverage_pct:.1f}% < "
                                f"{options.min_output_coverage_pct:.1f}%")
            if inpainted_pct > options.max_inpainted_pct:
                failures.append(f"inpainted area {inpainted_pct:.2f}% > "
                                f"{options.max_inpainted_pct:.2f}%")
            if unresolved_pct > options.max_unresolved_hole_pct:
                failures.append(f"unresolved foreground holes {unresolved_pct:.3f}% > "
                                f"{options.max_unresolved_hole_pct:.3f}%")
            if (masker and len(residual_detections) >
                    options.max_residual_foreground_detections):
                failures.append(f"static plate still contains {len(residual_detections)} "
                                "detectable foreground object(s)")
        else:
            np.save(result_dir / final_names["frame_transforms"],
                    np.full((frame_count, 3, 3), np.nan, np.float64))
            failures.append(f"only {registered} sampled frames registered; need at least 3")

        published_outputs = {
            key: path for key, path in output_paths.items()
            if key == "report" or (result_dir / final_names[key]).exists()
        }
        report = {
            "schema_version": SCHEMA_VERSION,
            "stage": 1,
            "contract": INPUT_CONTRACT,
            "input": {
                "path": str(source),
                "sha256": _file_sha256(source),
                "width": width,
                "height": height,
                "fps": fps,
                "frame_count": frame_count,
                "duration_s": round(frame_count / fps, 3),
                "variable_frame_rate_detected": timing["variable_frame_rate"],
                "normalized_to_cfr": normalized,
                "timestamp_jitter_p99_pct": round(timing["timestamp_jitter_p99_pct"], 4),
                "scene_cuts_s": cuts,
            },
            "reference": {
                "sample_slot": reference_slot,
                "video_frame": int(indices[reference_slot]),
                "time_s": round(float(indices[reference_slot] / fps), 4),
            },
            "detector": masker.metadata if masker else {"enabled": False},
            "quality": {
                "sampled_frames": len(indices),
                "registered_frames": registered,
                "registered_pct": round(registered_pct, 3),
                "output_coverage_pct": round(coverage_pct, 3),
                "blur_pct": round(blur_pct, 3),
                "normalized_blur_pct": round(normalized_blur_pct, 3),
                "normalized_sharpness_min": round(
                    float(np.min(accepted_normalized_sharpness)), 3),
                "normalized_sharpness_median": round(
                    float(np.median(accepted_normalized_sharpness)), 3),
                "sharpness_retention_min_pct": round(
                    float(np.min(sharpness_retention)), 3),
                "sharpness_retention_median_pct": round(
                    float(np.median(sharpness_retention)), 3),
                "normalization_interpolation": "lanczos4",
                "frame_geometry": audit_summary,
                "max_corner_motion_pct": round(max_motion, 3),
                "max_center_motion_pct": round(max_center_motion, 3),
                "max_rotation_deg": round(max_rotation, 4),
                "max_scale_change_pct": round(max_scale, 4),
                "median_registration_error_px": (
                    round(float(np.median(reprojections)), 4) if reprojections else None),
                "minimum_observations_per_pixel": minimum_observations,
                "rejected_frames_by_reason": dict(sorted(rejected_reasons.items())),
                "detected_foreground_by_class": dict(sorted(detected_classes.items())),
                "foreground_masked_observations_pct": round(foreground_observation_pct, 4),
                "semantic_exclusion_area_pct": round(semantic_exclusion_pct, 4),
                "semantic_consensus_frames": semantic_consensus_frames,
                "inpainted_area_pct": round(inpainted_pct, 4),
                "unresolved_foreground_hole_pct": round(unresolved_pct, 4),
                "residual_detections_before_repair": residual_detections_before_repair,
                "residual_repair_area_pct": round(residual_repair_pct, 4),
                "residual_foreground_detections": residual_detections,
            },
            "assumptions": ASSUMPTIONS,
            "limitations": LIMITATIONS,
            "coordinates": {
                "space": "reference_image_pixels",
                "origin": "top_left",
                "x_axis": "right",
                "y_axis": "down",
                "transform_direction": "source_frame_pixels_to_reference_image_pixels",
                "metric_scale": None,
            },
            "options": asdict(options),
            "pass": not failures,
            "failure_reasons": failures,
            "outputs": published_outputs,
        }
        (result_dir / final_names["report"]).write_text(
            json.dumps(report, indent=2), encoding="utf-8")
        (result_dir / ".stage1-output").write_text(SCHEMA_VERSION, encoding="utf-8")
        log("stage1", f"registered {registered}/{len(indices)}; coverage={coverage_pct:.1f}%; "
                      f"blur={blur_pct:.1f}%; max motion={max_motion:.1f}%")
        frames.flush(); valid.flush(); foreground.flush()
        _publish_directory(result_dir, output)
        del frames, valid, foreground
        return report


def _publish_failure_report(video, output_dir, options: Stage1Options,
                            error: Exception):
    """Atomically publish a minimal report when processing cannot reach quality gates."""
    source = Path(video).resolve()
    output = Path(output_dir).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    input_details = {"path": str(source)}
    if source.is_file():
        input_details["sha256"] = _file_sha256(source)
    report = {
        "schema_version": SCHEMA_VERSION,
        "stage": 1,
        "contract": INPUT_CONTRACT,
        "input": input_details,
        "assumptions": ASSUMPTIONS,
        "limitations": LIMITATIONS,
        "coordinates": {
            "space": "reference_image_pixels",
            "origin": "top_left",
            "x_axis": "right",
            "y_axis": "down",
            "transform_direction": "source_frame_pixels_to_reference_image_pixels",
            "metric_scale": None,
        },
        "options": asdict(options),
        "pass": False,
        "failure_reasons": [f"{type(error).__name__}: {error}"],
        "error": {"type": type(error).__name__, "message": str(error)},
        "outputs": {"report": "report.json"},
    }
    with tempfile.TemporaryDirectory(prefix=".drone-stage1-failure-",
                                     dir=output.parent) as temp_name:
        result_dir = Path(temp_name) / "result"
        result_dir.mkdir()
        (result_dir / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
        (result_dir / ".stage1-output").write_text(SCHEMA_VERSION, encoding="utf-8")
        _publish_directory(result_dir, output)
    return report


def run_stage1(video, output_dir, options: Stage1Options | None = None):
    """Run Stage 1 and guarantee a report for every catchable processing failure."""
    resolved_options = options or Stage1Options()
    source = Path(video).resolve()
    output = Path(output_dir).resolve()
    # Unsafe target errors intentionally do not publish into that target.
    _validate_output_target(source, output)
    try:
        return _run_stage1(video, output_dir, resolved_options)
    except Exception as error:
        try:
            _publish_failure_report(video, output_dir, resolved_options, error)
        except Exception as report_error:
            if hasattr(error, "add_note"):
                error.add_note(f"Could not publish Stage 1 failure report: {report_error}")
        raise
