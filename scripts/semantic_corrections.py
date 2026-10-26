"""Explicit operator constraints for canonical Stage 4 semantic label images."""
from __future__ import annotations

import cv2
import numpy as np

from .stage4_semantics import CANONICAL_CLASSES, CANONICAL_IDS


def remove_class_to_nearest(
        labels: np.ndarray, removed_class: str,
        replacement_classes: tuple[str, ...]) -> np.ndarray:
    """Replace one explicitly absent class with the nearest allowed class.

    This is an operator constraint, not an automatic semantic inference. Distances are
    measured in label-image pixels and ties follow replacement_classes order.
    """
    if not isinstance(labels, np.ndarray) or labels.ndim != 2:
        raise ValueError("canonical labels must be a two-dimensional array")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("canonical labels must use an integer dtype")
    if removed_class not in CANONICAL_IDS:
        raise ValueError(f"unknown removed class: {removed_class}")
    if not replacement_classes:
        raise ValueError("at least one replacement class is required")
    if removed_class in replacement_classes:
        raise ValueError("removed class cannot also be a replacement class")
    try:
        replacement_ids = [CANONICAL_IDS[name] for name in replacement_classes]
    except KeyError as exception:
        raise ValueError(f"unknown replacement class: {exception.args[0]}") \
            from exception
    minimum = int(labels.min())
    maximum = int(labels.max())
    if minimum < 0 or maximum >= len(CANONICAL_CLASSES):
        raise ValueError(
            f"canonical labels contain IDs outside 0..{len(CANONICAL_CLASSES) - 1}")

    result = labels.astype(np.uint8, copy=True)
    removed = result == CANONICAL_IDS[removed_class]
    if not np.any(removed):
        return result
    distances = []
    for class_id in replacement_ids:
        target = result == class_id
        if not np.any(target):
            distances.append(np.full(result.shape, np.inf, dtype=np.float32))
            continue
        distances.append(cv2.distanceTransform(
            (~target).astype(np.uint8), cv2.DIST_L2, 5))
    stack = np.stack(distances, axis=0)
    if not np.any(np.isfinite(stack[:, removed])):
        raise ValueError("none of the requested replacement classes exist in the map")
    nearest = np.argmin(stack, axis=0)
    for index, class_id in enumerate(replacement_ids):
        result[removed & (nearest == index)] = class_id
    return result
