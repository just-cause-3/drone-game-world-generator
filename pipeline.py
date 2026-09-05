#!/usr/bin/env python3
"""Fail-closed video-to-Unity pipeline, implemented one stage at a time."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from scripts.common import log
from scripts.stage1_ground import Stage1Options, run_stage1
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
    parser.add_argument("--foreground-model", default="yolo11m.pt")
    parser.add_argument("--no-foreground-mask", action="store_true",
                        help="diagnostic/testing only; must not be used for production worlds")
    parser.add_argument("--unity-project", default=str(DEFAULT_UNITY_PROJECT))
    parser.add_argument("--unity-editor",
                        help="Unity executable; normally detected from ProjectVersion.txt")
    return parser


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    _stage1_parser(subparsers)
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

    parser.error(f"unsupported command: {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
