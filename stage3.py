#!/usr/bin/env python3
"""Stage 3: detect, stabilize, track, and map cars and people for Unity."""
from __future__ import annotations

import argparse
import sys

from scripts.common import log
from scripts.stage1_ground import DEFAULT_FOREGROUND_MODEL
from scripts.stage3_tracking import CachedActorDetector, Stage3Options, run_stage3


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage1_bundle", help="passing Stage 1 artifact directory")
    parser.add_argument("stage2_bundle", help="passing Stage 2 calibration directory")
    parser.add_argument("--output", default="stage3_out",
                        help="Stage 3 tracking artifact directory")
    parser.add_argument("--model", default=DEFAULT_FOREGROUND_MODEL)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--detector-frame-stride", type=int, default=1,
                        help="inference cadence; use 2 when reusing fast-mode detections")
    parser.add_argument("--no-tta", action="store_true",
                        help="faster but potentially lower-recall diagnostic run")
    parser.add_argument(
        "--reuse-detections-from",
        help="reuse detections from a provenance-matching Stage 3 bundle")
    args = parser.parse_args(argv)
    options = Stage3Options(
        detector_model=args.model,
        detector_imgsz=args.imgsz,
        detector_tile_size=args.tile_size,
        detector_frame_stride=args.detector_frame_stride,
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
        log("stage3", "FAIL: Stage 3 artifact quality gate failed")
        for reason in report["failure_reasons"]:
            log("stage3", f"  - {reason}")
        return 2
    quality = report["quality"]
    log(
        "stage3",
        f"PASS: {quality['detections']} detections, "
        f"{quality['confirmed_tracks']} confirmed tracks")
    for warning in report["warnings"]:
        log("stage3", f"WARNING: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
