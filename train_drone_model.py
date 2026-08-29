#!/usr/bin/env python3
"""Reproduce or further improve the Stage 1 VisDrone detector."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

from scripts.track_agents import resolve_device


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="yolo26x.pt",
                        help="pretrained Ultralytics checkpoint")
    parser.add_argument("--data", default="VisDrone.yaml",
                        help="Ultralytics dataset YAML; VisDrone auto-downloads")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=640,
                        help="use 1280 on a GPU with substantially more than 4 GB VRAM")
    parser.add_argument("--batch", type=int, default=1,
                        help="batch size; -1 lets Ultralytics estimate it")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--project", default="training")
    parser.add_argument("--name", default="visdrone-yolo26x")
    parser.add_argument("--output", type=Path,
                        default=Path("models/visdrone-yolo26x-custom.pt"))
    args = parser.parse_args()

    try:
        from ultralytics import YOLO

        device = resolve_device(args.device)
        model = YOLO(args.base)
        model.train(
            data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
            workers=args.workers, device=device, project=args.project, name=args.name,
            pretrained=True, optimizer="auto", patience=50, cos_lr=True,
            close_mosaic=10, amp=True, cache="disk", deterministic=True, seed=0,
            val=True, plots=True,
        )
        best = Path(model.trainer.best)
        if not best.is_file():
            raise FileNotFoundError(f"training did not produce best.pt: {best}")
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        shutil.copy2(best, temporary)
        temporary.replace(output)
        print(f"Best checkpoint copied to {output}")
    except Exception as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
