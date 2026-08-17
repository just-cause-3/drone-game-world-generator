"""Phase 5: image tracks -> world tracks via time-INTERPOLATED homographies,
split on occlusion gaps, smooth per segment, stitch with a max-speed gate.

Nearest-neighbor H lookup gives every detection in a stride-window the same H, then
jumps: tracks stair-step, and no smoothing window can hide it. h_at() interpolates.
"""
import json

import cv2
import numpy as np

from .common import log


def h_at(n, stride, H_stack):
    """Homography for VIDEO frame n, interpolated between registration samples."""
    f = n / stride
    i0 = min(int(f), len(H_stack) - 1)
    i1 = min(i0 + 1, len(H_stack) - 1)
    a = f - i0
    H0 = H_stack[i0] / H_stack[i0][2, 2]
    H1 = H_stack[i1] / H_stack[i1][2, 2]
    return (1 - a) * H0 + a * H1  # adjacent H's are close -> normalized lerp is accurate


def smooth(seq, k):
    if len(seq) < k:
        return list(seq)
    half, out = k // 2, []
    for i in range(len(seq)):
        lo, hi = max(0, i - half), min(len(seq), i + half + 1)
        out.append(sum(seq[lo:hi]) / (hi - lo))
    return out


def split_on_gaps(pts, max_dt):
    """Smoothing must NOT blend across occlusion gaps (it would invent motion)."""
    segs, cur = [], [pts[0]]
    for q in pts[1:]:
        if q["t"] - cur[-1]["t"] <= max_dt:
            cur.append(q)
        else:
            segs.append(cur)
            cur = [q]
    segs.append(cur)
    return segs
