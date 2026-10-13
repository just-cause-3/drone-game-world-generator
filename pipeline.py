#!/usr/bin/env python3
"""Fail-closed video-to-Unity pipeline, implemented one stage at a time."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from scripts.common import log
from scripts.stage1_ground import DEFAULT_FOREGROUND_MODEL, Stage1Options, run_stage1
from scripts.stage2_calibration import Stage2Options, run_stage2
from scripts.stage3_tracking import CachedActorDetector, Stage3Options, run_stage3
from scripts.stage4_semantics import (
    DEFAULT_SEMANTIC_BACKEND,
    DEFAULT_SEMANTIC_MODEL,
    LEGACY_SEMANTIC_MODEL,
    Stage4Options,
    run_stage4_semantics,
)
from scripts.unity_stage1 import import_stage1_bundle


REPOSITORY_ROOT = Path(__file__).resolve().parent
DEFAULT_UNITY_PROJECT = REPOSITORY_ROOT / "UnityProject" / "DroneWorldRecreation"


def _stage1_parser(subparsers):
    parser = subparsers.add_parser(
        "stage1", help="stationary top-down video -> validated ground -> Unity preview")
    parser.add_argument("video", help="stationary, top-down video with bounded zoom")
    parser.add_argument("--output", default="stage1_out", help="Stage 1 artifact directory")
    parser.add_argument("--samples", type=int, default=41)
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--foreground-model", default=DEFAULT_FOREGROUND_MODEL)
    parser.add_argument("--no-foreground-mask", action="store_true",
                        help="diagnostic/testing only; must not be used for production worlds")
    parser.add_argument("--unity-project", default=str(DEFAULT_UNITY_PROJECT))
    parser.add_argument("--unity-editor",
                        help="Unity executable; normally detected from ProjectVersion.txt")
    return parser


def _stage2_parser(subparsers):
    parser = subparsers.add_parser(
        "stage2",
        help="passing Stage 1 ground + surveyed control points -> metric calibration")
    parser.add_argument("stage1_bundle", help="passing Stage 1 artifact directory")
    parser.add_argument("control_points", help="Stage 2 control-points JSON")
    parser.add_argument("--output", default="stage2_out",
                        help="Stage 2 calibration artifact directory")
    parser.add_argument("--max-rmse-m", type=float, default=0.25)
    parser.add_argument("--max-error-m", type=float, default=0.75)
    return parser


def _stage3_parser(subparsers):
    parser = subparsers.add_parser(
        "stage3",
        help="passing Stage 1 + Stage 2 -> stabilized car/person trajectories")
    parser.add_argument("stage1_bundle", help="passing Stage 1 artifact directory")
    parser.add_argument("stage2_bundle", help="passing Stage 2 calibration directory")
    parser.add_argument("--output", default="stage3_out",
                        help="Stage 3 tracking artifact directory")
    parser.add_argument("--model", default=DEFAULT_FOREGROUND_MODEL)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--reuse-detections-from")
    return parser


def _stage4_parser(subparsers):
    parser = subparsers.add_parser(
        "stage4",
        help="passing Stage 1 + Stage 2 -> review-gated static semantic map")
    parser.add_argument("stage1_bundle", help="passing Stage 1 artifact directory")
    parser.add_argument("stage2_bundle", help="passing Stage 2 calibration directory")
    parser.add_argument("--output", default="stage4_out")
    parser.add_argument(
        "--backend",
        choices=("hybrid-openearthmap-road", "openearthmap-mask2former",
                 "legacy-cityscapes-yolo"),
        default=DEFAULT_SEMANTIC_BACKEND)
    parser.add_argument("--model")
    parser.add_argument("--road-model", default=LEGACY_SEMANTIC_MODEL)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--no-flip-tta", action="store_true")
    parser.add_argument("--semantic-labels")
    return parser


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    _stage1_parser(subparsers)
    _stage2_parser(subparsers)
    _stage3_parser(subparsers)
    _stage4_parser(subparsers)
    args = parser.parse_args(argv)

    if args.command == "stage1":
        foreground_model = None if args.no_foreground_mask else args.foreground_model
        options = Stage1Options(sample_frames=args.samples, tile=args.tile,
                                foreground_model=foreground_model)
        try:
            report = run_stage1(args.video, args.output, options)
        except Exception as exception:
            print(f"FATAL: Stage 1 processing error: {exception}", file=sys.stderr)
            return 1
        if not report["pass"]:
            log("pipeline", "STOPPED: Stage 1 quality gate failed; Unity was not modified")
            for reason in report["failure_reasons"]:
                log("pipeline", f"  - {reason}")
            return 2
        try:
            unity_log = import_stage1_bundle(
                args.unity_project, args.output, args.unity_editor)
        except Exception as exception:
            print(f"FATAL: Unity publication error: {exception}", file=sys.stderr)
            return 3
        log("pipeline", f"PASS: Unity Stage 1 preview created; log={unity_log}")
        return 0

    if args.command == "stage2":
        options = Stage2Options(
            max_reprojection_rmse_m=args.max_rmse_m,
            max_reprojection_error_m=args.max_error_m)
        try:
            report = run_stage2(
                args.stage1_bundle, args.control_points, args.output, options)
        except Exception as exception:
            print(f"FATAL: Stage 2 calibration error: {exception}", file=sys.stderr)
            return 1
        if not report["pass"]:
            log("pipeline", "STOPPED: Stage 2 calibration quality gate failed")
            for reason in report["failure_reasons"]:
                log("pipeline", f"  - {reason}")
            return 2
        quality = report["quality"]
        log(
            "pipeline",
            f"PASS: metric calibration created from {quality['point_count']} points; "
            f"RMSE={quality['reprojection_rmse_m']:.4f} m")
        for warning in report["warnings"]:
            log("pipeline", f"WARNING: {warning}")
        return 0

    if args.command == "stage3":
        options = Stage3Options(
            detector_model=args.model,
            detector_imgsz=args.imgsz,
            detector_tile_size=args.tile_size,
            detector_tta=not args.no_tta,
        )
        try:
            detector = (CachedActorDetector(args.reuse_detections_from)
                        if args.reuse_detections_from else None)
            report = run_stage3(
                args.stage1_bundle, args.stage2_bundle, args.output, options, detector)
        except Exception as exception:
            print(f"FATAL: Stage 3 tracking error: {exception}", file=sys.stderr)
            return 1
        if not report["pass"]:
            log("pipeline", "STOPPED: Stage 3 tracking quality gate failed")
            for reason in report["failure_reasons"]:
                log("pipeline", f"  - {reason}")
            return 2
        quality = report["quality"]
        log(
            "pipeline",
            f"PASS: Stage 3 created {quality['confirmed_tracks']} confirmed "
            f"tracks from {quality['detections']} detections")
        for warning in report["warnings"]:
            log("pipeline", f"WARNING: {warning}")
        return 0

    if args.command == "stage4":
        default_model = (DEFAULT_SEMANTIC_MODEL
                         if args.backend != "legacy-cityscapes-yolo"
                         else LEGACY_SEMANTIC_MODEL)
        options = Stage4Options(
            semantic_backend=args.backend,
            semantic_model=args.model or default_model,
            road_specialist_model=args.road_model,
            inference_imgsz=args.imgsz,
            tile_size=args.tile_size,
            horizontal_flip_tta=not args.no_flip_tta,
        )
        try:
            report = run_stage4_semantics(
                args.stage1_bundle,
                args.stage2_bundle,
                args.output,
                options,
                external_labels=args.semantic_labels,
            )
        except Exception as exception:
            print(f"FATAL: Stage 4A semantic error: {exception}", file=sys.stderr)
            return 1
        if not report["pass"]:
            log("pipeline", "STOPPED: Stage 4A structural quality gate failed")
            for reason in report["failure_reasons"]:
                log("pipeline", f"  - {reason}")
            return 2
        quality = report["quality"]
        log(
            "pipeline",
            f"PASS: Stage 4A labelled {quality['known_valid_area_pct']:.2f}% "
            "of valid pixels; operator review remains required")
        for warning in report["warnings"]:
            log("pipeline", f"WARNING: {warning}")
        return 0

    parser.error(f"unsupported command: {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
