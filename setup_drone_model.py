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
