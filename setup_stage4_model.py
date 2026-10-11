#!/usr/bin/env python3
"""Download and verify the pinned aerial semantic model used by Stage 4A."""
from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
import sys
import tempfile
from urllib.request import Request, urlopen
import json
import shutil


MODEL_FILENAME = "yolo26x-sem.pt"
MODEL_SIZE = 80_789_279
MODEL_SHA256 = "eae8ff0d95bf96863c59c40bade61037870c7a39f2c53a8c49fd1b5022ed331d"
MODEL_URL = (
    "https://github.com/ultralytics/assets/releases/download/"
    "v8.4.0/yolo26x-sem.pt"
)
EXPECTED_CLASSES = (
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle",
    "bicycle",
)

AERIAL_REPOSITORY = "mfaytin/mask2former-satellite"
AERIAL_REVISION = "558c4959866ddd8dca42c13d24e49ca3e12bb71e"
AERIAL_DIRECTORY = Path("models/openearthmap-mask2former")
AERIAL_FILES = {
    "config.json": (
        2_347,
        "fc5880cfc7a50878061ad9c275908502a0e084f89363eabe09dbb66497feb98f"),
    "model.safetensors": (
        431_639_048,
        "0d10951e953d916cf8f32201d6b896c4ef7804bf442b80e64c3d7c7a7e94f3c7"),
    "preprocessor_config.json": (
        540,
        "a99b574cc14f69b07c8b4b92f032c18e3e5ed0a578ed3e63348c6aaac171d95d"),
    "README.md": (
        3_257,
        "f210cf16e72c421689a6a9a62e5ba9d362d893cbc4fd4ebc17e262ee7686a442"),
}
AERIAL_CLASSES = (
    "bareland", "rangeland", "developed_space", "road", "tree", "water",
    "agriculture", "building",
)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_legacy_model(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"semantic model file does not exist: {path}")
    if path.stat().st_size != MODEL_SIZE:
        raise ValueError(
            f"semantic model size mismatch: got {path.stat().st_size}, "
            f"expected {MODEL_SIZE}")
    actual = file_sha256(path)
    if actual != MODEL_SHA256:
        raise ValueError(
            f"semantic model SHA-256 mismatch: got {actual}, expected {MODEL_SHA256}")


def download(output: Path, force: bool) -> None:
    if output.exists() and not force:
        verify_legacy_model(output)
        print(f"Stage 4 semantic model already verified: {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    request = Request(MODEL_URL, headers={"User-Agent": "DroneWorld-Stage4/1.0"})
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".download", dir=output.parent)
    temporary = Path(temporary_name)
    try:
        with open(descriptor, "wb", closefd=True) as stream:
            with urlopen(request, timeout=60) as response:
                downloaded = 0
                while block := response.read(1024 * 1024):
                    stream.write(block)
                    downloaded += len(block)
                    print(
                        f"\rDownloading {100.0 * downloaded / MODEL_SIZE:5.1f}%",
                        end="", flush=True)
        print()
        verify_legacy_model(temporary)
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(f"Verified Stage 4 semantic model installed: {output}")


def verify_classes(path: Path) -> None:
    from ultralytics import YOLO

    model = YOLO(str(path))
    raw = model.names
    names = tuple(raw.values()) if isinstance(raw, dict) else tuple(raw)
    if names != EXPECTED_CLASSES:
        raise ValueError(f"unexpected semantic model classes: {names}")
    if model.task != "semantic":
        raise ValueError(f"expected semantic task, got {model.task!r}")
    print("Semantic classes verified: " + ", ".join(names))


def verify_aerial_model(directory: Path) -> None:
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(
            f"OpenEarthMap model directory does not exist: {directory}")
    for filename, (expected_size, expected_hash) in AERIAL_FILES.items():
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(f"OpenEarthMap model file is missing: {path}")
        if path.stat().st_size != expected_size:
            raise ValueError(
                f"{filename} size mismatch: got {path.stat().st_size}, "
                f"expected {expected_size}")
        actual = file_sha256(path)
        if actual != expected_hash:
            raise ValueError(
                f"{filename} SHA-256 mismatch: got {actual}, expected {expected_hash}")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if config.get("architectures") != ["Mask2FormerForUniversalSegmentation"]:
        raise ValueError("OpenEarthMap model has an unexpected architecture")
    labels = config.get("id2label")
    if not isinstance(labels, dict) or len(labels) != len(AERIAL_CLASSES):
        raise ValueError(
            "OpenEarthMap checkpoint must expose exactly eight output classes")


def download_aerial_model(output: Path, force: bool) -> None:
    output = Path(output).resolve()
    if output.exists() and not force:
        verify_aerial_model(output)
        print(f"Stage 4 aerial model already verified: {output}")
        return
    from huggingface_hub import snapshot_download

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{output.name}-", dir=output.parent))
    backup = output.with_name(f".{output.name}-previous")
    try:
        snapshot_download(
            AERIAL_REPOSITORY,
            revision=AERIAL_REVISION,
            local_dir=temporary,
            allow_patterns=list(AERIAL_FILES),
        )
        verify_aerial_model(temporary)
        if backup.exists():
            shutil.rmtree(backup)
        if output.exists():
            output.replace(backup)
        temporary.replace(output)
        if backup.exists():
            shutil.rmtree(backup)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        if backup.exists() and not output.exists():
            backup.replace(output)
        raise
    print(f"Verified Stage 4 OpenEarthMap model installed: {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=AERIAL_DIRECTORY)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--legacy-cityscapes", action="store_true",
        help="download/verify the superseded street-view bootstrap instead")
    parser.add_argument(
        "--aerial-only", action="store_true",
        help="install only OpenEarthMap, without the conservative road specialist")
    args = parser.parse_args()
    output = args.output.resolve()
    try:
        if args.legacy_cityscapes:
            legacy_output = (Path("models/yolo26x-sem.pt")
                             if args.output == AERIAL_DIRECTORY else args.output).resolve()
            if args.verify_only:
                verify_legacy_model(legacy_output)
            else:
                download(legacy_output, args.force)
            verify_classes(legacy_output)
        else:
            if args.verify_only:
                verify_aerial_model(output)
            else:
                download_aerial_model(output, args.force)
            print("Aerial classes verified: " + ", ".join(AERIAL_CLASSES))
            if not args.aerial_only:
                legacy_output = Path("models/yolo26x-sem.pt").resolve()
                if args.verify_only:
                    verify_legacy_model(legacy_output)
                else:
                    download(legacy_output, args.force)
                verify_classes(legacy_output)
    except Exception as exception:
        print(f"FATAL: {exception}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
