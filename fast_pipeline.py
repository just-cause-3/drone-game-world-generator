#!/usr/bin/env python3
"""Run the balanced fast Stage 1 -> Stage 4A pipeline for one video."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback

from full_pipeline import PipelineGateFailure, run_full_pipeline
from scripts.common import log
from scripts.stage3_tracking import Stage3Options, _load_inputs, CachedActorDetector, _bundle_path


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


def _stage4_resume_runners(video, run_root, assumed_world_width_m: float = 40.0):
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
        completed = pipeline.get("stages") or {}
        if (pipeline.get("failed_stage", pipeline.get("active_stage", 4)) != 4 or
                any((completed.get(f"stage{stage}") or {}).get("pass") is not True
                    for stage in (1, 2, 3))):
            log("fast", "Cannot resume Stage 4: this attempt did not pass Stages 1-3.")
            return None
        previous_width = float(pipeline.get("assumed_world_width_m", float("nan")))
        calibration_width = float((stage2.get("coordinate_system") or {}).get(
            "assumed_world_width_m", float("nan")))
        if not all(math.isclose(value, assumed_world_width_m, rel_tol=1e-12, abs_tol=1e-12)
                   for value in (previous_width, calibration_width)):
            log("fast", "Cannot resume Stage 4: approximate world width changed or is missing.")
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
        # A pass flag alone cannot establish that artifacts still belong together.
        # Revalidate the actual video, transforms, calibration and saved detections.
        inputs = _load_inputs(root / "stage1", root / "stage2")
        cached = CachedActorDetector(root / "stage3")
        cached.validate_inputs(inputs)
        cached.validate_options(fast_stage3_options())
        outputs = stage3.get("outputs") or {}
        for label in ("tracks", "tracking_preview", "track_map", "frame_counts"):
            artifact = _bundle_path(root / "stage3", outputs.get(label), label)
            expected_hash = (stage3.get("artifact_hashes") or {}).get(label + "_sha256")
            if expected_hash and _sha256(artifact) != expected_hash:
                raise ValueError(f"Stage 3 {label} changed since processing")
    except (OSError, ValueError, KeyError, TypeError) as error:
        log("fast", f"Cannot resume Stage 4: {error}")
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


def run_fast_pipeline(video, run_root, assumed_world_width_m: float = 40.0) -> dict:
    """Run all stages with a full-rate timeline and half-rate Stage 3 inference."""
    options = fast_stage3_options()
    log(
        "fast",
        "Balanced fast mode: Stage 3 detects alternate frames and the tracker "
        "interpolates the full source frame clock; resolution, tiles, and TTA remain enabled.")
    runners = _stage4_resume_runners(video, run_root, assumed_world_width_m)
    runner_arguments = {}
    if runners is not None:
        log(
            "fast",
            "Previous fast run passed Stages 1-3 and stopped in Stage 4; "
            "resuming at Stage 4 without repeating detector work.")
        runner_arguments = {
            "stage1_runner": runners[0],
            "stage2_runner": runners[1],
            "stage3_runner": runners[2],
        }
    return run_full_pipeline(
        video,
        run_root,
        assumed_world_width_m,
        stage3_options=options,
        pipeline_mode=FAST_PIPELINE_MODE,
        **runner_arguments,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="stationary, top-down input video")
    parser.add_argument("--run-root", required=True,
                        help="separate owned folder for fast-mode artifacts")
    parser.add_argument("--assumed-world-width", type=float, default=40.0,
                        help="approximate image width (default: 40 m)")
    args = parser.parse_args(argv)
    try:
        run_fast_pipeline(args.video, args.run_root, args.assumed_world_width)
        return 0
    except PipelineGateFailure as error:
        log("fast", f"STOPPED: Stage {error.stage} quality gate failed")
        for reason in error.reasons:
            log("fast", f"  - {reason}")
        return 2
    except Exception as error:
        print(f"FATAL: fast pipeline error: {error}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
