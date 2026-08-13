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


def estimate_ppm_from_cars(cfg, ref_path, default_ppm):
    """Scale from car sizes, orientation-filtered: an end-on car contributes its ~1.8 m
    WIDTH as 'length' and would skew the scale by up to 2.5x."""
    from ultralytics import YOLO
    from .track_agents import resolve_device
    det = cfg["detect"]
    res = YOLO(det["model"])(ref_path, imgsz=int(det["imgsz"]),
                             device=resolve_device(det.get("device", "auto")),
                             verbose=False)[0]
    dims = [(float(x2 - x1), float(y2 - y1))
            for x1, y1, x2, y2, conf, c in res.boxes.data.tolist() if int(c) == 2]  # COCO 2 = car
    lengths = [max(a, b) for a, b in dims if max(a, b) / max(1.0, min(a, b)) > 1.6]
    if len(lengths) < 5:
        log("calib", f"only {len(lengths)} usable cars in reference - staying unscaled")
        return default_ppm, "unscaled"
    spread = np.percentile(lengths, 75) / max(1.0, np.percentile(lengths, 25))
    ppm = float(np.median(lengths)) / 4.5  # median car length ~= 4.5 m
    status = "auto_scaled" if spread < 1.4 else "auto_scaled_lowconf"
    log("calib", f"car-size scale: {ppm:.1f} px/m from {len(lengths)} cars ({status})")
    return ppm, status


def calibrate_H(cfg, binding, ref_path):
    """Returns (H_ref_to_world pixels->meters, scale_status)."""
    if binding:
        pts = np.array(binding["points"], np.float32)
        H, _ = cv2.findHomography(pts[:, :2], pts[:, 2:])
        if H is None:
            die("calibration points are degenerate (collinear?) - recalibrate")
        return H, "metric"
    c = cfg["calibrate"]
    if c["auto"] == "car_size":
        ppm, status = estimate_ppm_from_cars(cfg, ref_path, float(c["ppm"]))
    else:
        ppm, status = float(c["ppm"]), "unscaled"
    img_h = cv2.imread(ref_path).shape[0]
    # v-down pixels mapped straight to +Z would build a MIRRORED world: negate v, shift so Z >= 0
    H = np.float32([[1 / ppm, 0, 0], [0, -1 / ppm, img_h / ppm], [0, 0, 1]])
    return H, status
