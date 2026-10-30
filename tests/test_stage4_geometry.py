import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import pipeline
from scripts.stage4_geometry import (
    GEOMETRY_SCHEMA,
    Stage4GeometryOptions,
    _extract_buildings,
    _extract_surface_regions,
    run_stage4_geometry,
)
from scripts.stage4_semantics import CANONICAL_IDS, run_stage4_semantics


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FakeSemanticProvider:
    def __init__(self, labels):
        self.labels = labels
        self.metadata = {
            "implementation": "synthetic-semantic-provider",
            "domain": "unit_test",
            "automatic": False,
        }

    def predict(self, image):
        return self.labels.copy()


class Stage4GeometryTests(unittest.TestCase):
    def test_component_area_uses_local_calibration_not_global_scale_guess(self):
        labels = np.zeros((100, 100), np.uint8)
        labels[20:40, 20:40] = CANONICAL_IDS["building"]
        labels[50:70, 50:70] = CANONICAL_IDS["road"]
        # The median scale argument can be much smaller than a region's local
        # scale in a projective calibration; it must not pre-filter its area.
        options = Stage4GeometryOptions(simplify_tolerance_m=0.001)
        homography = np.diag([0.5, 0.5, 1.0])
        buildings, _ = _extract_buildings(labels, homography, 0.01, options)
        surfaces = _extract_surface_regions(labels, homography, 0.01, options)
        self.assertEqual(1, len(buildings))
        self.assertEqual(1, len(surfaces))
        self.assertAlmostEqual(90.25, buildings[0]["area_m2"])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def make_accepted_bundle(self, accepted=True):
        width, height = 160, 100
        stage1 = self.root / "stage1"
        stage2 = self.root / "stage2"
        stage4 = self.root / "stage4"
        stage1.mkdir()
        stage2.mkdir()
        ground = np.full((height, width, 3), (80, 85, 90), np.uint8)
        valid = np.full((height, width), 255, np.uint8)
        self.assertTrue(cv2.imwrite(str(stage1 / "ground.png"), ground))
        self.assertTrue(cv2.imwrite(str(stage1 / "valid_mask.png"), valid))
        source_hash = "a" * 64
        stage1_report = {
            "schema_version": "stage1-ground-v1",
            "stage": 1,
            "pass": True,
            "input": {
                "path": "fixture.mp4",
                "sha256": source_hash,
                "width": width,
                "height": height,
            },
            "outputs": {
                "ground": "ground.png",
                "valid_mask": "valid_mask.png",
            },
        }
        (stage1 / "report.json").write_text(
            json.dumps(stage1_report), encoding="utf-8")
        calibration = {
            "schema_version": "stage2-calibration-v1",
            "stage": 2,
            "pass": True,
            "confidence": "approximate_assumed_scale",
            "source_stage1": {
                "report_sha256": sha256(stage1 / "report.json"),
                "ground_sha256": sha256(stage1 / "ground.png"),
                "source_video_sha256": source_hash,
                "pixel_width": width,
                "pixel_height": height,
            },
            "coordinate_system": {
                "metric_scale_is_assumed": True,
                "ground_y_m": 0.0,
            },
            "reference_pixel_to_world_xz": [
                [0.1, 0.0, -8.0],
                [0.0, -0.1, 5.0],
                [0.0, 0.0, 1.0],
            ],
        }
        (stage2 / "world_calibration.json").write_text(
            json.dumps(calibration), encoding="utf-8")
        labels = np.full((height, width), CANONICAL_IDS["road"], np.uint8)
        labels[:25, :] = CANONICAL_IDS["pavement"]
        labels[5:35, 10:70] = CANONICAL_IDS["building"]
        labels[55:95, 110:155] = CANONICAL_IDS["vegetation"]
        run_stage4_semantics(
            stage1, stage2, stage4, provider=FakeSemanticProvider(labels))
        acceptance = {
            "schema_version": "stage4-semantic-acceptance-v1",
            "stage": 4,
            "accepted": accepted,
            "stage4_report_sha256": sha256(stage4 / "report.json"),
            "semantic_labels_sha256": sha256(stage4 / "semantic_labels.png"),
            "accepted_utc": "2026-01-01T00:00:00Z",
            "note": "unit test",
        }
        (stage4 / "semantic_acceptance.json").write_text(
            json.dumps(acceptance), encoding="utf-8")
        return stage4

    def test_accepted_map_generates_hash_bound_geometry(self):
        stage4 = self.make_accepted_bundle()
        output = self.root / "geometry"

        report = run_stage4_geometry(stage4, output)

        self.assertTrue(report["pass"])
        self.assertEqual(report["schema_version"], GEOMETRY_SCHEMA)
        self.assertTrue(report["ready_for_unity_geometry"])
        self.assertEqual(report["quality"]["building_count"], 1)
        self.assertTrue((output / "geometry_overlay.png").is_file())
        geometry = json.loads((output / "geometry.json").read_text(encoding="utf-8"))
        building = geometry["buildings"][0]
        self.assertEqual(building["height_m"], 8.0)
        self.assertTrue(building["height_is_assumed"])
        self.assertGreaterEqual(len(building["footprint_xz"]), 4)
        self.assertEqual(
            len(building["roof_triangles"]),
            3 * (len(building["footprint_xz"]) - 2))
        self.assertEqual(
            report["artifact_hashes"]["geometry_sha256"],
            sha256(output / "geometry.json"))


if __name__ == "__main__":
    unittest.main()
