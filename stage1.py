#!/usr/bin/env python3
"""Stage 1: stationary/top-down drone video -> validated static ground image."""
import argparse
from pathlib import Path
import sys

from scripts.common import log
from scripts.stage1_ground import DEFAULT_FOREGROUND_MODEL, Stage1Options, run_stage1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="stationary or gently drifting top-down drone video")
    parser.add_argument("--output", default="stage1_out", help="output directory")
    parser.add_argument("--samples", type=int, default=41,
                        help="evenly spaced frames used for registration/background extraction")
    parser.add_argument("--tile", type=int, default=256,
                        help="median-processing tile size in pixels")
    parser.add_argument("--foreground-model", default=DEFAULT_FOREGROUND_MODEL,
                        help="aerial YOLO model used to exclude people/vehicles")
    parser.add_argument("--foreground-imgsz", type=int, default=1280,
                        help="detector input size; crops are upscaled to this size")
    parser.add_argument("--foreground-conf", type=float, default=0.08,
                        help="low recall-oriented detector confidence threshold")
    parser.add_argument("--residual-foreground-conf", type=float, default=0.25,
                        help="precision-oriented threshold for the final plate check")
    parser.add_argument("--foreground-tile-size", type=int, default=640,
                        help="overlapping inference crop size; 0 disables tiling")
    parser.add_argument("--foreground-tile-overlap", type=float, default=0.25,
                        help="fractional overlap between detector crops")
    parser.add_argument("--no-foreground-tta", action="store_true",
                        help="disable detector test-time augmentation")
    parser.add_argument("--no-foreground-mask", action="store_true",
                        help="disable semantic masking (diagnostic/testing only)")
    args = parser.parse_args()

    model = None if args.no_foreground_mask else args.foreground_model
    options = Stage1Options(
        sample_frames=args.samples, tile=args.tile, foreground_model=model,
        foreground_imgsz=args.foreground_imgsz, foreground_conf=args.foreground_conf,
        residual_foreground_conf=args.residual_foreground_conf,
        foreground_tile_size=args.foreground_tile_size,
        foreground_tile_overlap=args.foreground_tile_overlap,
        foreground_tta=not args.no_foreground_tta,
    )
    try:
        report = run_stage1(args.video, args.output, options)
    except Exception as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1

    if not report["pass"]:
        log("stage1", "FAIL: input did not satisfy the stationary top-down quality gate")
        for reason in report["failure_reasons"]:
            log("stage1", f"  - {reason}")
        return 2

    ground = Path(args.output).resolve() / report["outputs"]["ground"]
    log("stage1", f"PASS: static ground -> {ground}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
