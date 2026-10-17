#!/usr/bin/env python3
"""Run the balanced fast Stage 1 -> Stage 4A pipeline for one video."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

from full_pipeline import PipelineGateFailure, run_full_pipeline
from scripts.common import log
from scripts.stage3_tracking import Stage3Options


FAST_PIPELINE_MODE = "fast_balanced"


def fast_stage3_options() -> Stage3Options:
    """Keep aerial detector quality, but infer alternate frames."""
    return Stage3Options(detector_frame_stride=2)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _stage4_resume_runners(video, run_root):
    """Return Stage 1-3 report loaders after a Stage 4-only failed fast run."""
    source = Path(video).resolve()
    root = Path(run_root).resolve()
    pipeline_path = root / "pipeline_report.json"
    required = (
        pipeline_path,
        root / "stage1" / "report.json",
        root / "stage2" / "world_calibration.json",
        root / "stage3" / "report.json",
    )
    if not all(path.is_file() for path in required):
        return None
    try:
        pipeline, stage1, stage2, stage3 = map(_read_json, required)
        resumable_status = (
            pipeline.get("status") == "error" or
            (pipeline.get("status") == "quality_gate_failed" and
             pipeline.get("failed_stage") == 4))
        if not resumable_status or pipeline.get("pipeline_mode") != FAST_PIPELINE_MODE:
            return None
        video_data = pipeline.get("video") or {}
        if (Path(video_data.get("path", "")).resolve() != source or
                video_data.get("sha256") != _sha256(source)):
            return None
        if not (
                stage1.get("schema_version") == "stage1-ground-v1" and
                stage2.get("schema_version") == "stage2-calibration-v1" and
                stage3.get("schema_version") == "stage3-tracks-v1" and
                stage1.get("pass") is True and stage2.get("pass") is True and
                stage3.get("pass") is True and
                (stage1.get("input") or {}).get("sha256") == video_data["sha256"] and
                (stage3.get("options") or {}).get("detector_frame_stride") == 2):
            return None
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None

    def reuse_stage1(_video, _output, _options):
        log("fast", "Resume: reusing passing Stage 1 artifacts")
        return stage1

    def reuse_stage2(_stage1, _controls, _output, _options):
        log("fast", "Resume: reusing passing Stage 2 artifacts")
        return stage2

    def reuse_stage3(_stage1, _stage2, _output, _options):
        log("fast", "Resume: reusing passing Stage 3 artifacts")
        return stage3

    return reuse_stage1, reuse_stage2, reuse_stage3
