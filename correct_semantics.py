#!/usr/bin/env python3
"""Apply an explicit absent-class constraint to a canonical Stage 4 label PNG."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

from scripts.semantic_corrections import remove_class_to_nearest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_labels")
    parser.add_argument("output_labels")
    parser.add_argument("--remove", required=True)
    parser.add_argument("--replace-with", nargs="+", required=True)
    args = parser.parse_args(argv)
    source = Path(args.input_labels).resolve()
    output = Path(args.output_labels).resolve()
    labels = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
    if labels is None:
        print(f"FATAL: canonical label image is unreadable: {source}", file=sys.stderr)
        return 1
    try:
        corrected = remove_class_to_nearest(
            labels, args.remove, tuple(args.replace_with))
        output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output), corrected):
            raise OSError(f"could not write corrected labels: {output}")
    except Exception as exception:
        print(f"FATAL: semantic correction failed: {exception}", file=sys.stderr)
        return 1
    changed = int((corrected != labels).sum())
    print(
        f"Wrote {output}; replaced {changed} '{args.remove}' pixels using nearest "
        f"{', '.join(args.replace_with)} region")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
