#!/usr/bin/env python3
"""Run the fail-closed Stage 1 -> Stage 4A pipeline for one video."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback
import time

from scripts.common import log
from scripts.progress import duration
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
    if not math.isfinite(world_width) or world_width <= 0.0:
        raise ValueError("assumed world width must be finite and positive")
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
    primary = directory / ("world_calibration.json" if directory.name == "stage2" else "report.json")
    return {
        "pass": report.get("pass") is True,
        "failure_reasons": list(report.get("failure_reasons") or []),
        "warnings": list(report.get("warnings") or []),
        "artifact_directory": str(directory),
        "report_sha256": _sha256(primary) if primary.is_file() else None,
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


def run_full_pipeline(
        video, run_root, assumed_world_width_m: float = 40.0,
        *, stage1_runner=run_stage1, stage2_runner=run_stage2,
        stage3_runner=run_stage3, stage4_runner=run_stage4_semantics,
        stage3_options: Stage3Options | None = None,
        pipeline_mode: str = "accuracy") -> dict:
    """Run Stages 1-4A; operator approval and 4B/C geometry are separate steps.

    Unity import is deliberately a separate caller step, not part of Python's pass.
    """
    source = Path(video).resolve()
    root = Path(run_root).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"input video does not exist: {source}")
    if not pipeline_mode.strip():
        raise ValueError("pipeline_mode cannot be empty")
    if not math.isfinite(assumed_world_width_m) or assumed_world_width_m <= 0.0:
        raise ValueError("assumed world width must be finite and positive")
    tracking_options = stage3_options or Stage3Options()
    tracking_options.validate()
    _prepare_run_root(root, source, pipeline_mode)

    report_path = root / "pipeline_report.json"
    pipeline_report = {
        "schema_version": PIPELINE_SCHEMA,
        "pass": False,
        "status": "running",
        "pipeline_mode": pipeline_mode,
        "requires_stage4_human_review": True,
        "ready_for_stage4_geometry": False,
        "processing_scope": "stages_1_to_4a",
        "end_to_end_complete": False,
        "unity_import_completed": False,
        "pending_steps": ["unity_import", "semantic_review_and_acceptance", "stage4b_c_geometry"],
        "active_stage": 1,
        "video": {"path": str(source), "sha256": _sha256(source)},
        "assumed_world_width_m": float(assumed_world_width_m),
        "started_utc": _utc_now(),
        "finished_utc": None,
        "stages": {},
    }

    def publish() -> None:
        temporary_report = report_path.with_suffix(".json.tmp")
        temporary_report.write_text(
            json.dumps(pipeline_report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary_report.replace(report_path)

    publish()
    paths = {stage: root / f"stage{stage}" for stage in range(1, 5)}
    started = time.monotonic()

    def timed(stage, runner, *args):
        began = time.monotonic()
        result = runner(*args)
        log("full", f"Stage {stage} processing finished in {duration(time.monotonic() - began)}; "
            f"pipeline elapsed {duration(time.monotonic() - started)}. "
            "Frame ETAs cover their phase only, not the remaining pipeline.")
        return result

    try:
        log("full", "Stage 1/4: validating video and reconstructing static ground")
        stage1 = timed(1, stage1_runner, source, paths[1], Stage1Options())
        pipeline_report["stages"]["stage1"] = _stage_summary(stage1, paths[1])
        publish()
        if stage1.get("pass") is not True:
            raise PipelineGateFailure(1, list(stage1.get("failure_reasons") or []))

        dimensions = stage1.get("input") or {}
        controls = root / "stage2_control_points.json"
        _write_approximate_control_points(
            controls, int(dimensions["width"]), int(dimensions["height"]),
            float(assumed_world_width_m))
        log(
            "full",
            f"Stage 2/4: assigning approximate {assumed_world_width_m:g} m image width")
        pipeline_report["active_stage"] = 2
        publish()
        stage2 = timed(2, stage2_runner, paths[1], controls, paths[2], Stage2Options())
        pipeline_report["stages"]["stage2"] = _stage_summary(stage2, paths[2])
        publish()
        if stage2.get("pass") is not True:
            raise PipelineGateFailure(2, list(stage2.get("failure_reasons") or []))

        log("full", "Stage 3/4: detecting and tracking cars and pedestrians")
        pipeline_report["active_stage"] = 3
        publish()
        stage3 = timed(3, stage3_runner,
            paths[1], paths[2], paths[3], tracking_options)
        pipeline_report["stages"]["stage3"] = _stage_summary(stage3, paths[3])
        publish()
        if stage3.get("pass") is not True:
            raise PipelineGateFailure(3, list(stage3.get("failure_reasons") or []))

        _release_stage_resources()
        log("full", "Stage 4/4: proposing aerial static semantic labels")
        pipeline_report["active_stage"] = 4
        publish()
        stage4 = timed(4, stage4_runner, paths[1], paths[2], paths[4], Stage4Options())
        pipeline_report["stages"]["stage4"] = _stage_summary(stage4, paths[4])
        publish()
        if stage4.get("pass") is not True:
            raise PipelineGateFailure(4, list(stage4.get("failure_reasons") or []))

        pipeline_report["pass"] = True
        pipeline_report["status"] = "pass_review_required"
        pipeline_report["active_stage"] = None
        log(
            "full",
            "PASS: Stages 1-4A passed; Stage 4 semantic review is still required")
    except PipelineGateFailure as error:
        pipeline_report["status"] = "quality_gate_failed"
        pipeline_report["failed_stage"] = error.stage
        pipeline_report["failure_reasons"] = error.reasons
        raise
    except KeyboardInterrupt:
        pipeline_report["status"] = "cancelled"
        pipeline_report["error"] = {
            "type": "KeyboardInterrupt", "message": "pipeline cancelled by operator",
        }
        raise
    except Exception as error:
        pipeline_report["status"] = "error"
        pipeline_report["failed_stage"] = pipeline_report["active_stage"]
        pipeline_report["error"] = {
            "type": type(error).__name__, "message": str(error),
        }
        raise
    finally:
        pipeline_report["finished_utc"] = _utc_now()
        pipeline_report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        publish()
    return pipeline_report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="stationary, top-down input video")
    parser.add_argument("--run-root", required=True,
                        help="owned folder for isolated Stage 1-4 artifacts")
    parser.add_argument("--assumed-world-width", type=float, default=40.0,
                        help="approximate width represented by the image (default: 40 m)")
    args = parser.parse_args(argv)
    try:
        run_full_pipeline(args.video, args.run_root, args.assumed_world_width)
        return 0
    except PipelineGateFailure as error:
        log("full", f"STOPPED: Stage {error.stage} quality gate failed")
        for reason in error.reasons:
            log("full", f"  - {reason}")
        return 2
    except Exception as error:
        print(f"FATAL: full pipeline error: {error}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
