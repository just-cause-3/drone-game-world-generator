#!/usr/bin/env python3
"""Download and verify the pinned YOLO26x VisDrone detector used by Stage 1."""
from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
import sys
import tempfile
from urllib.request import Request, urlopen


MODEL_REPOSITORY = "dronefreak/visdrone-yolov26x"
MODEL_REVISION = "2daccb9375d2fce7d176f2555b7bef18ad93eac3"
MODEL_FILENAME = "best.pt"
MODEL_SIZE = 118_363_941
MODEL_SHA256 = "0292dfafe94d08ba9db9cecbef482f1c0dc156e1575284cd2ba6f28547723936"
MODEL_URL = (
    f"https://huggingface.co/{MODEL_REPOSITORY}/resolve/"
    f"{MODEL_REVISION}/{MODEL_FILENAME}"
)
EXPECTED_CLASSES = (
    "pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle",
    "awning-tricycle", "bus", "motor", "others",
)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"model file does not exist: {path}")
    size = path.stat().st_size
    if size != MODEL_SIZE:
        raise ValueError(f"model size mismatch: got {size}, expected {MODEL_SIZE}")
    actual_hash = file_sha256(path)
    if actual_hash != MODEL_SHA256:
        raise ValueError(
            f"model SHA-256 mismatch: got {actual_hash}, expected {MODEL_SHA256}")


def download(output: Path, force: bool) -> None:
    if output.exists() and not force:
        verify(output)
        print(f"Model already verified: {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    request = Request(MODEL_URL, headers={"User-Agent": "DroneWorld-Stage1/1.0"})
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".download", dir=output.parent)
    temporary_path = Path(temporary_name)
    try:
        with open(descriptor, "wb", closefd=True) as temporary:
            with urlopen(request, timeout=60) as response:
                downloaded = 0
                while chunk := response.read(1024 * 1024):
                    temporary.write(chunk)
                    downloaded += len(chunk)
                    percent = 100.0 * downloaded / MODEL_SIZE
                    print(f"\rDownloading {percent:5.1f}%", end="", flush=True)
        print()
        verify(temporary_path)
        temporary_path.replace(output)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    print(f"Verified model installed: {output}")


def verify_ultralytics_classes(path: Path) -> None:
    from ultralytics import YOLO

    model = YOLO(str(path))
    raw_names = model.names
    names = tuple(raw_names.values()) if isinstance(raw_names, dict) else tuple(raw_names)
    if names != EXPECTED_CLASSES:
        raise ValueError(f"unexpected model classes: {names}")
    print("Model classes verified: " + ", ".join(names))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("models/visdrone-yolo26x.pt"))
    parser.add_argument("--force", action="store_true",
                        help="replace an existing file after re-downloading it")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    try:
        if args.verify_only:
            verify(output)
        else:
            download(output, args.force)
        verify_ultralytics_classes(output)
    except Exception as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
