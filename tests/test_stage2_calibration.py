import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

import pipeline
from scripts.stage2_calibration import (
    CALIBRATION_SCHEMA,
    CONTROL_POINTS_SCHEMA,
    Stage2Options,
    run_stage2,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stage1_bundle(root: Path, width=400, height=300, passing=True) -> Path:
    bundle = root / "stage1"
    bundle.mkdir()
    image = np.zeros((height, width, 4), dtype=np.uint8)
    image[..., 0] = np.arange(width, dtype=np.uint8)
    image[..., 1] = np.arange(height, dtype=np.uint8)[:, None]
    image[..., 2] = 100
    image[..., 3] = 255
    self_mask = np.full((height, width), 255, dtype=np.uint8)
    cv2.imwrite(str(bundle / "ground.png"), image)
    cv2.imwrite(str(bundle / "valid_mask.png"), self_mask)
    report = {
        "schema_version": "stage1-ground-v1",
        "stage": 1,
        "pass": passing,
        "failure_reasons": [] if passing else ["fixture rejection"],
        "input": {
            "path": "fixture.mp4",
            "sha256": "a" * 64,
            "width": width,
            "height": height,
        },
        "reference": {"video_frame": 20, "time_s": 1.0},
        "outputs": {"ground": "ground.png", "valid_mask": "valid_mask.png"},
    }
    (bundle / "report.json").write_text(
        json.dumps(report), encoding="utf-8")
    return bundle


def _write_points(path: Path, image_points, world_points) -> Path:
    value = {
        "schema_version": CONTROL_POINTS_SCHEMA,
        "units": "meters",
        "origin_description": "fixture origin",
        "x_axis_description": "east",
        "z_axis_description": "north",
        "ground_y_m": 0,
        "points": [
            {
                "id": f"p{index + 1}",
                "image": {"u": float(image[0]), "v": float(image[1])},
                "world": {"x": float(world[0]), "z": float(world[1])},
            }
            for index, (image, world) in enumerate(
                zip(image_points, world_points))
        ],
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


class Stage2CalibrationTests(unittest.TestCase):
    def test_recovers_metric_homography_and_binds_stage1_hashes(self):
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            stage1 = _stage1_bundle(root)
            image_points = np.array([
                [20, 20], [200, 15], [380, 25],
                [25, 275], [205, 285], [375, 270],
            ], dtype=np.float64)
            expected_h = np.array([
                [0.04, 0.002, -3.0],
                [-0.001, -0.05, 18.0],
                [0.00001, -0.00002, 1.0],
            ], dtype=np.float64)
            world_points = cv2.perspectiveTransform(
                image_points.reshape(-1, 1, 2), expected_h).reshape(-1, 2)
            points_path = _write_points(
                root / "points.json", image_points, world_points)
            output = root / "stage2"

            result = run_stage2(stage1, points_path, output)

            self.assertTrue(result["pass"])
            self.assertEqual(result["schema_version"], CALIBRATION_SCHEMA)
            self.assertEqual(result["confidence"], "well_constrained")
            self.assertLess(result["quality"]["reprojection_rmse_m"], 1e-6)
            self.assertGreater(result["quality"]["image_convex_hull_area_pct"], 50)
            recovered_h = np.array(result["reference_pixel_to_world_xz"])
            recovered = cv2.perspectiveTransform(
                image_points.reshape(-1, 1, 2), recovered_h).reshape(-1, 2)
            np.testing.assert_allclose(recovered, world_points, atol=1e-6)
            self.assertEqual(
                result["source_stage1"]["ground_sha256"],
                _sha256(stage1 / "ground.png"))
            self.assertTrue((output / "world_calibration.json").is_file())
            self.assertTrue((output / "calibration_overlay.png").is_file())
            self.assertTrue((output / "control_points_used.json").is_file())

    def test_four_points_pass_but_are_marked_unverified(self):
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            stage1 = _stage1_bundle(root)
            image_points = np.array([
                [20, 20], [380, 20], [380, 280], [20, 280],
            ], dtype=np.float64)
            world_points = np.array([
                [0, 20], [30, 20], [30, 0], [0, 0],
            ], dtype=np.float64)
            points_path = _write_points(
                root / "points.json", image_points, world_points)

            result = run_stage2(stage1, points_path, root / "stage2")

            self.assertTrue(result["pass"])
            self.assertEqual(result["confidence"], "minimum_unverified")
            self.assertEqual(
                result["quality"]["independent_residual_degrees_of_freedom"], 0)
            self.assertTrue(result["warnings"])

    def test_approximate_image_extent_mode_needs_no_measurements(self):
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            stage1 = _stage1_bundle(root, width=400, height=300)
            image_points = np.array([
                [0, 0], [399, 0], [399, 299], [0, 299],
            ], dtype=np.float64)
            world_points = np.array([
                [-20, 15], [20, 15], [20, -15], [-20, -15],
            ], dtype=np.float64)
            points_path = _write_points(
                root / "points.json", image_points, world_points)
            point_data = json.loads(points_path.read_text(encoding="utf-8"))
            point_data["calibration_mode"] = "approximate_image_extent"
            point_data["assumed_world_width_m"] = 40.0
            points_path.write_text(json.dumps(point_data), encoding="utf-8")

            result = run_stage2(stage1, points_path, root / "stage2")

            self.assertTrue(result["pass"])
            self.assertEqual(result["confidence"], "approximate_assumed_scale")
            self.assertTrue(
                result["coordinate_system"]["metric_scale_is_assumed"])
            self.assertEqual(
                result["coordinate_system"]["assumed_world_width_m"], 40.0)
            self.assertTrue(any(
                "not a physical measurement" in warning
                for warning in result["warnings"]))

    def test_rejects_clustered_control_points(self):
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            stage1 = _stage1_bundle(root)
            image_points = np.array([
                [100, 100], [120, 100], [120, 120],
                [100, 120], [110, 105], [105, 115],
            ], dtype=np.float64)
            world_points = image_points * np.array([0.05, -0.05])
            points_path = _write_points(
                root / "points.json", image_points, world_points)

            result = run_stage2(stage1, points_path, root / "stage2")

            self.assertFalse(result["pass"])
            self.assertTrue(any(
                "control points cover" in reason
                for reason in result["failure_reasons"]))

    def test_rejects_inconsistent_measured_point(self):
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            stage1 = _stage1_bundle(root)
            image_points = np.array([
                [20, 20], [200, 15], [380, 25],
                [25, 275], [205, 285], [375, 270],
            ], dtype=np.float64)
            world_points = image_points * np.array([0.05, -0.05])
            world_points[2] += np.array([4.0, 3.0])
            points_path = _write_points(
                root / "points.json", image_points, world_points)

            result = run_stage2(
                stage1, points_path, root / "stage2",
                Stage2Options(max_reprojection_rmse_m=0.1,
                              max_reprojection_error_m=0.2))

            self.assertFalse(result["pass"])
            self.assertTrue(any(
                "reprojection" in reason or "control-point error" in reason
                for reason in result["failure_reasons"]))

    def test_refuses_failed_stage1_bundle(self):
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            stage1 = _stage1_bundle(root, passing=False)
            points_path = root / "points.json"
            points_path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "failed its quality gate"):
                run_stage2(stage1, points_path, root / "stage2")


if __name__ == "__main__":
    unittest.main()
