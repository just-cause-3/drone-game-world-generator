#!/usr/bin/env python3
"""Generate provisional Unity geometry from an accepted Stage 4A bundle."""
from __future__ import annotations

import argparse
import sys

from scripts.stage4_geometry import Stage4GeometryOptions, run_stage4_geometry


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Accepted Stage 4A semantics -> provisional 3D geometry bundle")
    parser.add_argument("stage4_bundle", help="accepted Stage 4A semantic bundle")
    parser.add_argument("--output", default="stage4_geometry_out")
    parser.add_argument(
        "--height-policy", choices=("fixed", "footprint_prior"), default="fixed")
    parser.add_argument("--building-height", type=float, default=8.0)
    parser.add_argument("--minimum-building-area", type=float, default=6.0)
    parser.add_argument("--simplify-tolerance", type=float, default=0.25)
    parser.add_argument("--generate-vegetation", action="store_true")
    args = parser.parse_args(argv)
    options = Stage4GeometryOptions(
        height_policy=args.height_policy,
        default_building_height_m=args.building_height,
        minimum_building_area_m2=args.minimum_building_area,
        simplify_tolerance_m=args.simplify_tolerance,
        generate_vegetation_proxies=args.generate_vegetation,
    )
    try:
        report = run_stage4_geometry(args.stage4_bundle, args.output, options)
    except Exception as exception:
        print(f"FATAL: Stage 4B geometry error: {exception}", file=sys.stderr)
        return 1
    print(
        f"PASS: generated {report['quality']['building_count']} provisional "
        f"buildings; inspect geometry_overlay.png and the Unity scene")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
