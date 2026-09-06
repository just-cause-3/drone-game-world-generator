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

    def test_frame_geometry_audit_flags_a_sudden_zoom_jump(self):
        with tempfile.TemporaryDirectory() as temp_name:
            video = Path(temp_name) / "zoom_jump.avi"
            world = _synthetic_world()
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"),
                                     15.0, (320, 240))
            self.assertTrue(writer.isOpened())
            for index in range(20):
                if index < 10:
                    frame = world
                else:
                    transform = cv2.getRotationMatrix2D((160, 120), 0.0, 0.90)
                    frame = cv2.warpAffine(world, transform, (320, 240),
                                           borderMode=cv2.BORDER_REFLECT)
                writer.write(frame)
            writer.release()
            options = Stage1Options(foreground_model=None,
                                    frame_audit_orb_features=1500)

            audit = _frame_geometry_audit(video, 15.0, options)

            self.assertFalse(audit["summary"]["smooth"])
            self.assertGreater(audit["summary"]["max_scale_step_pct"],
                               options.max_frame_scale_step_pct)

    def test_similarity_registration_recovers_small_hover_drift(self):
        reference = _synthetic_world()
        transform = cv2.getRotationMatrix2D((160, 120), 0.8, 1.01)
        transform[:, 2] += (5, -3)
        frame = cv2.warpAffine(reference, transform, (320, 240))
        options = Stage1Options(min_matches=15, min_inliers=12,
                                max_rotation_deg=3, max_scale_change_pct=5,
                                foreground_model=None)
        H, metrics = _registration(frame, reference, options)
        self.assertIsNotNone(H, metrics)
        self.assertTrue(metrics["accepted"], metrics)
        aligned = cv2.warpPerspective(frame, H, (320, 240))
        self.assertLess(float(cv2.absdiff(aligned, reference).mean()), 12.0)

    def test_rejects_large_zoom_outside_stage_contract(self):
        reference = _synthetic_world()
        transform = cv2.getRotationMatrix2D((160, 120), 0.0, 1.30)
        frame = cv2.warpAffine(reference, transform, (320, 240))
        options = Stage1Options(min_matches=15, min_inliers=12,
                                max_scale_change_pct=15, foreground_model=None)
        H, metrics = _registration(frame, reference, options)
        self.assertIsNone(H)
        self.assertEqual(metrics["reason"], "camera_zoomed_too_far")

    def test_accepts_centered_zoom_within_relaxed_stage_contract(self):
        reference = _synthetic_world()
        transform = cv2.getRotationMatrix2D((160, 120), 0.0, 0.86)
        frame = cv2.warpAffine(reference, transform, (320, 240),
                               borderMode=cv2.BORDER_REFLECT)
        options = Stage1Options(min_matches=15, min_inliers=12,
                                foreground_model=None)
        H, metrics = _registration(frame, reference, options)
        self.assertIsNotNone(H, metrics)
        self.assertTrue(metrics["accepted"], metrics)
        self.assertLessEqual(metrics["scale_change_pct"], 18.0)
        self.assertLessEqual(metrics["center_motion_pct"], 3.0)

    def test_end_to_end_builds_clean_registered_background(self):
        with tempfile.TemporaryDirectory() as temp_name:
            temp = Path(temp_name)
            video = temp / "hover.avi"
            output = temp / "stage1"
            output.mkdir()
            (output / ".stage1-output").write_text(
                "stage1-ground-v1", encoding="utf-8")
            (output / "stale.txt").write_text("old run", encoding="utf-8")
            world = _synthetic_world()
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"),
                                     15.0, (320, 240))
            self.assertTrue(writer.isOpened())
            for index in range(30):
                dx = int(round(4 * np.sin(index / 5)))
                dy = int(round(3 * np.cos(index / 6)))
                frame = cv2.warpAffine(world, np.float32([[1, 0, dx], [0, 1, dy]]),
                                       (320, 240), borderMode=cv2.BORDER_REFLECT)
                x = 10 + index * 9
                cv2.rectangle(frame, (x, 130), (min(x + 18, 319), 148), (0, 0, 255), -1)
                writer.write(frame)
            writer.release()

            options = Stage1Options(sample_frames=21, tile=64, min_matches=15,
                                    min_inliers=12, min_output_coverage_pct=85,
                                    foreground_model=None)
            report = run_stage1(video, output, options)
            self.assertTrue(report["pass"], report["failure_reasons"])
            self.assertGreaterEqual(report["quality"]["registered_pct"], 90)
            self.assertEqual(report["quality"]["frame_geometry"]["estimated_pct"], 100.0)
            self.assertIn("normalized_sharpness_median", report["quality"])
            self.assertEqual(report["quality"]["normalization_interpolation"], "lanczos4")
            self.assertEqual(report["schema_version"], "stage1-ground-v1")
            self.assertFalse((output / "stale.txt").exists())
            transforms = np.load(output / "frame_transforms_to_reference.npy")
            self.assertEqual(transforms.shape, (30, 3, 3))
            ground = cv2.imread(str(output / "ground.png"), cv2.IMREAD_UNCHANGED)
            self.assertEqual(ground.shape, (240, 320, 4))
            self.assertGreater(float(np.mean(ground[:, :, 3] == 255)), 0.85)
            # The red foreground rectangle crosses most of the road but is visible at
            # each location in fewer than half the samples, so the median must remove it.
            red_foreground = ((ground[:, :, 2].astype(int) - ground[:, :, 1]) > 70) & (
                (ground[:, :, 2].astype(int) - ground[:, :, 0]) > 70)
            self.assertLess(int(red_foreground.sum()), 25)

    def test_registration_failure_still_publishes_report(self):
        with tempfile.TemporaryDirectory() as temp_name:
            temp = Path(temp_name)
            video = temp / "blank.avi"
            output = temp / "stage1"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"),
                                     15.0, (160, 120))
            self.assertTrue(writer.isOpened())
            for _ in range(15):
                writer.write(np.full((120, 160, 3), 80, np.uint8))
            writer.release()
            options = Stage1Options(sample_frames=9, tile=64, foreground_model=None)
            report = run_stage1(video, output, options)
            self.assertFalse(report["pass"])
            self.assertTrue((output / "report.json").exists())
            self.assertNotIn("ground", report["outputs"])

    def test_inpainting_is_limited_to_small_holes(self):
        image = np.full((100, 100, 3), 100, np.uint8)
        holes = np.zeros((100, 100), bool)
        holes[45:55, 45:55] = True
        options = Stage1Options(foreground_model=None, max_inpaint_component_pct=2.0)
        repaired, mask, unresolved = _inpaint_small_holes(image, holes, options)
        self.assertEqual(int(np.count_nonzero(mask)), 100)
        self.assertFalse(np.any(unresolved))
        self.assertEqual(repaired.shape, image.shape)

    def test_unexpected_processing_error_publishes_atomic_failure_report(self):
        with tempfile.TemporaryDirectory() as temp_name:
            temp = Path(temp_name)
            missing_video = temp / "missing.mp4"
            output = temp / "stage1"
            output.mkdir()
            (output / ".stage1-output").write_text(
                "stage1-ground-v1", encoding="utf-8")
            (output / "stale.txt").write_text("old run", encoding="utf-8")

            with self.assertRaises(FileNotFoundError):
                run_stage1(missing_video, output, Stage1Options(foreground_model=None))

            self.assertFalse((output / "stale.txt").exists())
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            self.assertFalse(report["pass"])
            self.assertEqual(report["error"]["type"], "FileNotFoundError")
            self.assertEqual(set(report["outputs"]), {"report"})


if __name__ == "__main__":
    unittest.main()
