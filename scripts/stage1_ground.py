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
