#!/usr/bin/env python3
"""run.py [config.yaml] [--force] - one command: video in -> Unity-ready world out.

Stages (plan v3): ingest -> track -> calibrate -> geometry -> GATE -> mosaic -> map -> deploy.
Cached artifacts are reused (frames/, tracks) so the calibrate-then-rerun workflow is cheap;
--force redoes everything.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np

from scripts.common import log, die, load_config, p
from scripts.ingest import ingest_video
from scripts.track_agents import run_tracking
from scripts.calibrate import resolve_reference, calibrate_H
from scripts.geometry import register_all
from scripts.build_mosaic import build_median_mosaic
from scripts.map_to_world import map_tracks


def blur_pct(frames, thresh):
    v = [cv2.Laplacian(cv2.imread(f, 0), cv2.CV_64F).var() for f in frames]
    return 100.0 * float(np.mean(np.array(v) < thresh))


def gate(cfg, inliers, n_tracks, blur):
    g = cfg["gates"]
    reg_pct = 100.0 * float(np.mean(np.array(inliers) >= cfg["geometry"]["min_inliers"]))
    ok = (reg_pct >= g["min_frames_registered_pct"]
          and n_tracks >= g["min_tracks"]
          and blur <= g["max_blur_pct"])
    return ok, {"frames_registered_pct": round(reg_pct, 1), "tracks": n_tracks,
                "blur_pct": round(blur, 1), "pass": bool(ok)}
