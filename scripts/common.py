"""Shared helpers: config, project paths, ffprobe, logging, the frame-index convention."""
import json
import subprocess
import sys
import time
from pathlib import Path


def log(stage, msg):
    # ASCII-only output: Windows consoles on cp1252 choke on fancy characters
    print(f"[{time.strftime('%H:%M:%S')}] [{stage}] {msg}", flush=True)


def die(msg, code=1):
    print(f"FATAL: {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


def load_config(path):
    import yaml  # lazy: lets selfcheck run with only numpy+opencv installed
    path = Path(path).resolve()
    if not path.exists():
        die(f"config not found: {path}")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(path.parent)
    return cfg


def p(cfg, *parts):
    """Project-rooted path (everything is relative to the config file's directory)."""
    return str(Path(cfg["_root"], *parts))


def mask_path(masks_dir, n):
    """The one naming convention: masks are keyed by VIDEO frame number n (Part 2.8)."""
    return str(Path(masks_dir) / f"frame_{n:06d}.png")
