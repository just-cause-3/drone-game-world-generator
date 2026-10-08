#!/usr/bin/env python3
"""Download and verify the pinned OpenCV LaMa model used by Stage 1."""
from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
import sys
import tempfile
from urllib.request import Request, urlopen


MODEL_REPOSITORY = "opencv/inpainting_lama"
MODEL_REVISION = "c47be270660574005b6c951b5904f51a1af28f6d"
MODEL_FILENAME = "inpainting_lama_2025jan.onnx"
MODEL_PATH = Path("models") / MODEL_FILENAME
MODEL_SIZE = 92_591_623
MODEL_SHA256 = "7df918ac3921d3daf0aae1d219776cf0dc4e4935f035af81841b40adcf74fdf2"
MODEL_URL = (
    f"https://huggingface.co/{MODEL_REPOSITORY}/resolve/"
    f"{MODEL_REVISION}/{MODEL_FILENAME}"
)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"LaMa model file does not exist: {path}")
    if path.stat().st_size != MODEL_SIZE:
        raise ValueError(
            f"LaMa model size mismatch: got {path.stat().st_size}, expected {MODEL_SIZE}")
    actual = file_sha256(path)
    if actual != MODEL_SHA256:
        raise ValueError(
            f"LaMa model SHA-256 mismatch: got {actual}, expected {MODEL_SHA256}")


def download(output: Path, force: bool) -> None:
    if output.exists() and not force:
        verify(output)
        print(f"Stage 1 LaMa model already verified: {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    request = Request(MODEL_URL, headers={"User-Agent": "DroneWorld-Stage1/1.0"})
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".download", dir=output.parent)
    temporary = Path(temporary_name)
    try:
        with open(descriptor, "wb", closefd=True) as stream:
            with urlopen(request, timeout=90) as response:
                downloaded = 0
                while block := response.read(1024 * 1024):
                    stream.write(block)
                    downloaded += len(block)
                    print(
                        f"\rDownloading LaMa {100.0 * downloaded / MODEL_SIZE:5.1f}%",
                        end="", flush=True)
        print()
        verify(temporary)
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(f"Verified Stage 1 LaMa model installed: {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    try:
        if args.verify_only:
            verify(output)
        else:
            download(output, args.force)
        # Parsing once catches an incompatible/corrupt graph beyond the byte hash.
        import cv2
        cv2.dnn.readNetFromONNX(str(output))
        print("LaMa ONNX graph verified by OpenCV.")
    except Exception as exception:
        print(f"FATAL: {exception}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
