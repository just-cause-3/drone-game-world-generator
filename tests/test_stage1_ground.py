import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from scripts.stage1_ground import (
    Stage1Options,
    _class_agnostic_nms,
    _build_background,
    _frame_geometry_audit,
    _inpaint_small_holes,
    _registration,
    _semantic_exclusion,
    _tile_origins,
    run_stage1,
)


def _synthetic_world(width=320, height=240):
    image = np.full((height, width, 3), 55, np.uint8)
    for x in range(0, width, 20):
        cv2.line(image, (x, 0), (x, height - 1), (75, 75, 75), 1)
    for y in range(0, height, 20):
        cv2.line(image, (0, y), (width - 1, y), (75, 75, 75), 1)
    cv2.rectangle(image, (35, 35), (125, 105), (170, 170, 170), -1)
    cv2.circle(image, (250, 70), 35, (25, 130, 40), -1)
    cv2.putText(image, "ROAD", (120, 190), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (235, 235, 235), 2, cv2.LINE_AA)
    return image


class Stage1GroundTests(unittest.TestCase):
    def test_detector_tiles_cover_entire_axis_and_anchor_both_edges(self):
        origins = _tile_origins(1280, 640, 0.25)
        self.assertEqual(origins[0], 0)
        self.assertEqual(origins[-1], 640)
        covered = np.zeros(1280, bool)
        for origin in origins:
            covered[origin:origin + 640] = True
        self.assertTrue(np.all(covered))

    def test_detector_nms_removes_cross_pass_duplicates(self):
        candidates = [
            {"bbox_xyxy": [10, 10, 30, 30], "confidence": 0.9, "class": "car"},
            {"bbox_xyxy": [11, 11, 31, 31], "confidence": 0.8, "class": "car"},
            {"bbox_xyxy": [70, 70, 80, 80], "confidence": 0.7, "class": "pedestrian"},
        ]
        merged = _class_agnostic_nms(candidates, 0.45)
        self.assertEqual(len(merged), 2)
        self.assertIn(candidates[0], merged)
        self.assertIn(candidates[2], merged)

    def test_rejects_invalid_detector_overlap(self):
        with self.assertRaisesRegex(ValueError, "tile_overlap"):
            Stage1Options(foreground_model=None,
                          foreground_tile_overlap=0.95).validate()


if __name__ == "__main__":
    unittest.main()
