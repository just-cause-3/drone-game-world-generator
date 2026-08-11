"""Phase 2: detect + track people/vehicles on the full-fps video; write mover masks.

Points carry the VIDEO frame number n (the one index, Part 2.8). Masks are named by n;
they are only materialized for registration frames (n % stride == 0) since geometry and
the mosaic never read any other frame.  # ponytail: skip 5/6 of mask writes; write all if needed later
"""
import json
import os
import shutil

import cv2
import numpy as np

from .common import log, mask_path


def resolve_device(want):
    if want not in (None, "auto"):
        return want
    try:
        import torch
        if torch.cuda.is_available():
            return 0
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"
