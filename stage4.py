#!/usr/bin/env python3
"""Stage 4A: create a review-gated static semantic map from Stage 1 and Stage 2."""
from __future__ import annotations

import argparse
import sys

from scripts.common import log
from scripts.stage4_semantics import (
    DEFAULT_SEMANTIC_BACKEND,
    DEFAULT_SEMANTIC_MODEL,
    LEGACY_SEMANTIC_MODEL,
    Stage4Options,
    run_stage4_semantics,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage1_bundle", help="passing Stage 1 artifact directory")
    parser.add_argument("stage2_bundle", help="passing Stage 2 calibration directory")
    parser.add_argument("--output", default="stage4_out")
    parser.add_argument(
        "--backend",
        choices=("hybrid-openearthmap-road", "openearthmap-mask2former",
                 "legacy-cityscapes-yolo"),
        default=DEFAULT_SEMANTIC_BACKEND)
    parser.add_argument(
        "--model", help="model directory/file; defaults depend on --backend")
    parser.add_argument("--road-model", default=LEGACY_SEMANTIC_MODEL)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--no-flip-tta", action="store_true")
    parser.add_argument(
        "--semantic-labels",
        help="corrected canonical one-channel PNG; skips automatic inference")
    args = parser.parse_args(argv)
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
        log("stage4", "FAIL: Stage 4A structural quality gate rejected the proposal")
        for reason in report["failure_reasons"]:
            log("stage4", f"  - {reason}")
        return 2
    quality = report["quality"]
    log(
        "stage4",
        f"PASS: {quality['known_valid_area_pct']:.2f}% labelled; "
        "operator review is still required before geometry")
    for warning in report["warnings"]:
        log("stage4", f"WARNING: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
