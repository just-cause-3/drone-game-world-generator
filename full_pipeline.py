#!/usr/bin/env python3
"""Run the fail-closed Stage 1 -> Stage 4A pipeline for one video."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import sys
import traceback

from scripts.common import log
from scripts.stage1_ground import Stage1Options, run_stage1
from scripts.stage2_calibration import (
    CONTROL_POINTS_SCHEMA,
    Stage2Options,
    run_stage2,
)
from scripts.stage3_tracking import Stage3Options, run_stage3
from scripts.stage4_semantics import Stage4Options, run_stage4_semantics


PIPELINE_SCHEMA = "droneworld-full-pipeline-v1"
RUN_MARKER = ".droneworld-full-pipeline-run"


class PipelineGateFailure(RuntimeError):
    """A stage completed normally but rejected its input/artifacts."""

    def __init__(self, stage: int, reasons: list[str]):
        super().__init__(f"Stage {stage} quality gate failed")
        self.stage = stage
        self.reasons = reasons


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prepare_run_root(run_root: Path, video: Path, pipeline_mode: str) -> None:
    if run_root == run_root.parent:
        raise ValueError("the filesystem root cannot be a pipeline run folder")
    if run_root.exists() and not run_root.is_dir():
        raise ValueError(f"pipeline run path is not a directory: {run_root}")
    if run_root.exists() and any(run_root.iterdir()):
        marker = run_root / RUN_MARKER
        if not marker.is_file():
            raise ValueError(
                f"refusing to use unrelated non-empty run directory: {run_root}")
        marker_data = json.loads(marker.read_text(encoding="utf-8"))
        previous_video = Path(marker_data.get("video", "")).resolve()
        if previous_video != video:
            raise ValueError(
                "run directory belongs to a different input video; choose another folder")
        previous_mode = marker_data.get("pipeline_mode", "accuracy")
        if previous_mode != pipeline_mode:
            raise ValueError(
                f"run directory belongs to {previous_mode!r} mode; choose a separate "
                f"folder for {pipeline_mode!r} mode")
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / RUN_MARKER).write_text(
        json.dumps({
            "schema_version": PIPELINE_SCHEMA,
            "video": str(video),
            "pipeline_mode": pipeline_mode,
        }, indent=2)
        + "\n",
        encoding="utf-8",
    )


def _write_approximate_control_points(
        path: Path, width: int, height: int, world_width: float) -> None:
    if width < 2 or height < 2:
        raise ValueError("Stage 1 declared invalid image dimensions")
    if world_width <= 0.0:
        raise ValueError("assumed world width must be positive")
    half_width = world_width * 0.5
    half_depth = world_width * height / width * 0.5
    max_u = width - 1
    max_v = height - 1
    data = {
        "schema_version": CONTROL_POINTS_SCHEMA,
        "units": "meters",
        "calibration_mode": "approximate_image_extent",
        "assumed_world_width_m": world_width,
        "origin_description": "approximate image centre",
        "x_axis_description": "image right",
        "z_axis_description": "image up",
        "ground_y_m": 0.0,
        "points": [
            {"id": "auto_tl", "image": {"u": 0.0, "v": 0.0},
             "world": {"x": -half_width, "z": half_depth}},
            {"id": "auto_tr", "image": {"u": max_u, "v": 0.0},
             "world": {"x": half_width, "z": half_depth}},
            {"id": "auto_br", "image": {"u": max_u, "v": max_v},
             "world": {"x": half_width, "z": -half_depth}},
            {"id": "auto_bl", "image": {"u": 0.0, "v": max_v},
             "world": {"x": -half_width, "z": -half_depth}},
        ],
    }
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _stage_summary(report: dict, directory: Path) -> dict:
    return {
        "pass": report.get("pass") is True,
        "failure_reasons": list(report.get("failure_reasons") or []),
        "warnings": list(report.get("warnings") or []),
        "artifact_directory": str(directory),
    }


def _release_stage_resources() -> None:
    """Release cyclic ML objects and unused CUDA cache between heavy stages."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass
