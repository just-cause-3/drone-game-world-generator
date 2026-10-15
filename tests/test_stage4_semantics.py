import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

import pipeline
from scripts.stage4_semantics import (
    CANONICAL_IDS,
    STAGE4_SCHEMA,
    Stage4Options,
    _fuse_aerial_with_road_specialist,
    run_stage4_semantics,
)


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
        self.image_shape = image.shape
        return self.labels.copy()


class Stage4SemanticTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def make_bundles(self, width=160, height=100, passing=True):
        stage1 = self.root / "stage1"
        stage2 = self.root / "stage2"
        stage1.mkdir()
        stage2.mkdir()
        ground = np.zeros((height, width, 4), np.uint8)
        ground[:, :, :3] = (80, 85, 90)
        ground[:, :, 3] = 255
        valid = np.full((height, width), 255, np.uint8)
        self.assertTrue(cv2.imwrite(str(stage1 / "ground.png"), ground))
        self.assertTrue(cv2.imwrite(str(stage1 / "valid_mask.png"), valid))
        source_hash = "4" * 64
        report = {
            "schema_version": "stage1-ground-v1",
            "stage": 1,
            "pass": passing,
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
            json.dumps(report), encoding="utf-8")
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
        return stage1, stage2

    def labels(self, width=160, height=100):
        labels = np.full((height, width), CANONICAL_IDS["road"], np.uint8)
        labels[:25, :] = CANONICAL_IDS["pavement"]
        labels[5:20, 10:45] = CANONICAL_IDS["building"]
        labels[5:20, 120:150] = CANONICAL_IDS["vegetation"]
        return labels

    def test_hybrid_specialist_can_only_promote_road(self):
        aerial = np.array([
            [CANONICAL_IDS["building"], CANONICAL_IDS["pavement"]],
            [CANONICAL_IDS["vegetation"], CANONICAL_IDS["terrain"]],
        ], dtype=np.uint8)
        specialist = np.array([
            [CANONICAL_IDS["building"], CANONICAL_IDS["road"]],
            [CANONICAL_IDS["road"], CANONICAL_IDS["building"]],
        ], dtype=np.uint8)

        fused = _fuse_aerial_with_road_specialist(aerial, specialist)

        np.testing.assert_array_equal(fused, np.array([
            [CANONICAL_IDS["building"], CANONICAL_IDS["road"]],
            [CANONICAL_IDS["road"], CANONICAL_IDS["terrain"]],
        ], dtype=np.uint8))

    def test_emits_review_bundle_with_upstream_and_artifact_hashes(self):
        stage1, stage2 = self.make_bundles()
        output = self.root / "stage4"
        provider = FakeSemanticProvider(self.labels())

        report = run_stage4_semantics(
            stage1, stage2, output, provider=provider)

        self.assertTrue(report["pass"])
        self.assertEqual(report["schema_version"], STAGE4_SCHEMA)
        self.assertTrue(report["requires_human_review"])
        self.assertFalse(report["ready_for_geometry"])
        self.assertEqual(
            report["source_stage1"]["report_sha256"],
            sha256(stage1 / "report.json"))
        self.assertEqual(
            report["source_stage2"]["calibration_sha256"],
            sha256(stage2 / "world_calibration.json"))
        for filename in (
                "report.json", "semantic_labels.png", "semantic_overlay.png",
                "components.json", "mask_road.png", "mask_building.png"):
            self.assertTrue((output / filename).is_file(), filename)
        self.assertEqual(
            report["artifact_hashes"]["semantic_labels_sha256"],
            sha256(output / "semantic_labels.png"))
        imported = cv2.imread(
            str(output / "semantic_labels.png"), cv2.IMREAD_UNCHANGED)
        np.testing.assert_array_equal(imported, self.labels())

    def test_small_components_are_removed_to_unknown(self):
        stage1, stage2 = self.make_bundles()
        labels = self.labels()
        labels[50, 50] = CANONICAL_IDS["building"]
        options = Stage4Options(minimum_component_area_px=4)

        report = run_stage4_semantics(
            stage1, stage2, self.root / "stage4", options,
            provider=FakeSemanticProvider(labels))

        self.assertTrue(report["pass"])
        self.assertEqual(report["quality"]["cleanup"]["removed_components"], 1)
        cleaned = cv2.imread(
            str(self.root / "stage4" / "semantic_labels.png"),
            cv2.IMREAD_UNCHANGED)
        self.assertEqual(cleaned[50, 50], CANONICAL_IDS["unknown"])

    def test_excessive_unknown_area_fails_without_geometry_readiness(self):
        stage1, stage2 = self.make_bundles()
        labels = np.zeros((100, 160), np.uint8)
        labels[:, :20] = CANONICAL_IDS["road"]

        report = run_stage4_semantics(
            stage1, stage2, self.root / "stage4",
            provider=FakeSemanticProvider(labels))

        self.assertFalse(report["pass"])
        self.assertFalse(report["ready_for_geometry"])
        self.assertTrue(any(
            "unknown semantic area" in reason
            for reason in report["failure_reasons"]))

    def test_rejects_unknown_canonical_label_id(self):
        stage1, stage2 = self.make_bundles()
        labels = self.labels()
        labels[0, 0] = 99
        with self.assertRaisesRegex(ValueError, "outside 0"):
            run_stage4_semantics(
                stage1, stage2, self.root / "stage4",
                provider=FakeSemanticProvider(labels))

    def test_masks_unsupported_pixels_to_unknown(self):
        stage1, stage2 = self.make_bundles()
        valid = cv2.imread(str(stage1 / "valid_mask.png"), cv2.IMREAD_GRAYSCALE)
        valid[:, :10] = 0
        cv2.imwrite(str(stage1 / "valid_mask.png"), valid)
        labels = self.labels()

        report = run_stage4_semantics(
            stage1, stage2, self.root / "stage4",
            provider=FakeSemanticProvider(labels))

        self.assertTrue(report["pass"])
        self.assertGreater(
            report["quality"]["provider_labels_outside_valid_mask"], 0)
        cleaned = cv2.imread(
            str(self.root / "stage4" / "semantic_labels.png"),
            cv2.IMREAD_UNCHANGED)
        self.assertTrue(np.all(cleaned[:, :10] == CANONICAL_IDS["unknown"]))

    def test_rejects_stale_stage2_binding(self):
        stage1, stage2 = self.make_bundles()
        report_path = stage1 / "report.json"
        value = json.loads(report_path.read_text(encoding="utf-8"))
        value["input"]["path"] = "changed.mp4"
        report_path.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "stale"):
            run_stage4_semantics(
                stage1, stage2, self.root / "stage4",
                provider=FakeSemanticProvider(self.labels()))


if __name__ == "__main__":
    unittest.main()
