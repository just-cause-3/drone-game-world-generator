"""Phase 3B step 1: reference selection + calibration (manual clicks or auto-fallback).

Manual calibration is BOUND to its exact reference frame (filename + sha1): clicked
pixels mean nothing on any other frame, and 'ref_frame: auto' re-picking a different
frame must never silently invalidate them.

Run interactively for metric scale:   python -m scripts.calibrate [config.yaml]
(click 4+ ground points on the reference frame, type each point's real X Z in meters,
press 's' to save, 'u' to undo, ESC to abort)
"""
import glob
import hashlib
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from .common import log, die, load_config, p


def _sha1(path):
    return hashlib.sha1(open(path, "rb").read()).hexdigest()


def pick_reference(frames, mode):
    """'auto' = frame with most ORB features (sampled); else an explicit index."""
    if mode != "auto":
        return int(mode)
    orb = cv2.ORB_create(2000)
    idxs = np.unique(np.linspace(0, len(frames) - 1, min(len(frames), 40)).astype(int))
    best, best_n = len(frames) // 2, -1
    for i in idxs:
        g = cv2.imread(frames[i], 0)
        kp = orb.detect(g, None)
        if len(kp) > best_n:
            best, best_n = int(i), len(kp)
    log("calib", f"auto reference: frame index {best} ({best_n} ORB features)")
    return best


def resolve_reference(cfg, frames, work_dir):
    """Returns (ref_idx, binding_or_None). The binding overrides ref_frame:auto."""
    bpath = os.path.join(work_dir, "calibration.json")
    if os.path.exists(bpath):
        b = json.load(open(bpath, encoding="utf-8"))
        ref_path = next((f for f in frames if Path(f).name == b["ref_frame_file"]), None)
        if ref_path is None:
            die("calibration is bound to a frame that no longer exists - "
                "delete out/calibration.json or recalibrate")
        if _sha1(ref_path) != b["ref_frame_sha1"]:
            die("reference frame content changed (crop/stride/re-encode) - "
                "delete out/calibration.json and recalibrate")
        log("calib", f"using bound reference {b['ref_frame_file']} (metric calibration)")
        return frames.index(ref_path), b
    return pick_reference(frames, cfg["geometry"]["ref_frame"]), None
