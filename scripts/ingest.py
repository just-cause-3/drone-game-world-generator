"""Phase 1: inspect, split on cuts, crop, normalize fps, extract stride frames.

One ffmpeg encode produces out/shot_clean.mp4 (constant fps, cropped), then a second
pass extracts every STRIDE-th frame. Extracted file i (1-based frame_%06d) is video
frame n = (i-1)*stride - exact by construction, never by directory-listing zip.
"""
import glob
import json
import os
import shutil
from pathlib import Path

from .common import log, die, ffprobe_info, run_cmd


def _longest_shot(video):
    """Return (start_s, end_s) of the longest continuous shot, or None if uncut /
    scenedetect unavailable."""
    try:
        from scenedetect import detect, AdaptiveDetector
    except ImportError:
        log("ingest", "WARNING: scenedetect not installed - assuming one continuous shot")
        return None
    scenes = detect(str(video), AdaptiveDetector())
    if len(scenes) <= 1:
        return None
    start, end = max(scenes, key=lambda s: s[1].get_seconds() - s[0].get_seconds())
    log("ingest", f"{len(scenes)} shots detected; keeping longest "
                  f"{start.get_seconds():.1f}s - {end.get_seconds():.1f}s")
    return start.get_seconds(), end.get_seconds()
