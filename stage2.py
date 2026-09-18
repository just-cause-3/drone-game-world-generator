#!/usr/bin/env python3
"""Stage 2: calibrate Stage 1 reference pixels to metric Unity X/Z coordinates."""
from __future__ import annotations

import argparse
import sys

from scripts.common import log
from scripts.stage2_calibration import Stage2Options, run_stage2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage1_bundle", help="passing Stage 1 artifact directory")
    parser.add_argument("control_points", help="Stage 2 control-points JSON")
    parser.add_argument("--output", default="stage2_out",
                        help="Stage 2 calibration artifact directory")
    parser.add_argument("--max-rmse-m", type=float, default=0.25)
    parser.add_argument("--max-error-m", type=float, default=0.75)
    args = parser.parse_args(argv)
    options = Stage2Options(
        max_reprojection_rmse_m=args.max_rmse_m,
        max_reprojection_error_m=args.max_error_m)
    try:
        result = run_stage2(
            args.stage1_bundle, args.control_points, args.output, options)
    except Exception as exception:
        print(f"FATAL: Stage 2 calibration error: {exception}", file=sys.stderr)
        return 1
    if not result["pass"]:
        log("stage2", "FAIL: calibration quality gate rejected the control points")
        for reason in result["failure_reasons"]:
            log("stage2", f"  - {reason}")
        return 2
    quality = result["quality"]
    log(
        "stage2",
        f"PASS: {quality['point_count']} points; "
        f"RMSE={quality['reprojection_rmse_m']:.4f} m; "
        f"world_calibration.json -> {args.output}")
    for warning in result["warnings"]:
        log("stage2", f"WARNING: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

