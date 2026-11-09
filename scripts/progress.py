"""Lightweight, monotonic, phase-local ETA and heartbeat logging (no ML changes)."""
from __future__ import annotations

import math
import threading
import time

from .common import log


def duration(seconds: float) -> str:
    seconds = max(0, math.ceil(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
