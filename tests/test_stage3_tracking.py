import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from scripts.stage3_tracking import (
    CachedActorDetector,
    StabilizedTracker,
    Stage3Options,
    _canonical_nms,
    _project_point,
    run_stage3,
)


class FakeDetector:
    def __init__(self, detections_by_frame):
        self.detections_by_frame = detections_by_frame
        self.metadata = {
            "model": "synthetic-fixture",
            "model_sha256": "0" * 64,
            "device": "cpu",
            "model_classes": {"0": "pedestrian", "3": "car"},
            "canonical_classes": {"0": "person", "3": "car"},
        }

    def detect(self, frame, frame_index):
        del frame
        return [dict(item) for item in self.detections_by_frame.get(frame_index, [])]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def detection(kind, cx, cy, confidence=0.9, size=10.0):
    half = size / 2.0
    return {
        "kind": kind,
        "detector_class_id": 3 if kind == "car" else 0,
        "detector_class": kind,
        "confidence": confidence,
        "bbox_xyxy": [cx - half, cy - half, cx + half, cy + half],
        "inference_passes": ["fixture"],
    }


class Stage3TrackingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def make_bundles(self, frame_count=10, fps=10.0, transforms=None):
        width, height = 160, 100
        stage1 = self.root / "stage1"
        stage2 = self.root / "stage2"
        stage1.mkdir()
        stage2.mkdir()
        ground = np.full((height, width, 4), 180, np.uint8)
        ground[:, :, 3] = 255
        self.assertTrue(cv2.imwrite(str(stage1 / "ground.png"), ground))

        video_path = stage1 / "video_cfr.avi"
        writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
        self.assertTrue(writer.isOpened())
        for index in range(frame_count):
            frame = np.full((height, width, 3), 55 + index, np.uint8)
            cv2.line(frame, (0, 50), (159, 50), (160, 160, 160), 2)
            writer.write(frame)
        writer.release()

        if transforms is None:
            transforms = np.repeat(np.eye(3)[None, :, :], frame_count, axis=0)
        np.save(stage1 / "frame_transforms_to_reference.npy", transforms)
        source_hash = "1" * 64
        report = {
            "schema_version": "stage1-ground-v1",
            "stage": 1,
            "pass": True,
            "input": {
                "path": "fixture.avi",
                "sha256": source_hash,
                "width": width,
                "height": height,
                "fps": fps,
                "frame_count": frame_count,
            },
            "reference": {"video_frame": frame_count // 2},
            "outputs": {
                "ground": "ground.png",
                "video_cfr": "video_cfr.avi",
                "frame_transforms": "frame_transforms_to_reference.npy",
            },
        }
        (stage1 / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
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
            json.dumps(calibration, indent=2), encoding="utf-8")
        return stage1, stage2

    def test_coordinate_projection_normalizes_homogeneous_value(self):
        matrix = np.array([[2.0, 0.0, 4.0], [0.0, 3.0, 6.0], [0.0, 0.0, 2.0]])
        self.assertEqual(_project_point(matrix, 3.0, 2.0), (5.0, 6.0))

    def test_nms_keeps_different_canonical_kinds(self):
        candidates = [
            {**detection("car", 50, 50, 0.9), "inference_pass": "full"},
            {**detection("car", 51, 50, 0.8), "inference_pass": "tile"},
            {**detection("person", 50, 50, 0.85), "inference_pass": "tile"},
        ]
        merged = _canonical_nms(candidates, 0.45)
        self.assertEqual(len(merged), 2)
        self.assertEqual({item["kind"] for item in merged}, {"car", "person"})
        car = next(item for item in merged if item["kind"] == "car")
        self.assertEqual(car["inference_passes"], ["full", "tile"])

    def test_appearance_prevents_nearby_car_identity_swap(self):
        tracker = StabilizedTracker(100, 100, 10.0, Stage3Options())

        def tracked_detection(x, descriptor):
            item = detection("car", x, 50, 0.9, 12)
            item.update({
                "frame": 0,
                "reference_anchor": [x, 50],
                "reference_bbox_xyxy": item["bbox_xyxy"],
                "world_anchor": [x, 50],
                "_appearance": np.asarray(descriptor, dtype=np.float32),
            })
            return item

        red = tracked_detection(48, [1.0, 0.0])
        blue = tracked_detection(52, [0.0, 1.0])
        tracker.update(0, [red, blue])
        next_blue = tracked_detection(49, [0.0, 1.0])
        next_blue["frame"] = 1
        next_red = tracked_detection(51, [1.0, 0.0])
        next_red["frame"] = 1
        tracker.update(1, [next_blue, next_red])
        self.assertEqual(next_red["track_id"], red["track_id"])
        self.assertEqual(next_blue["track_id"], blue["track_id"])

    def test_end_to_end_tracks_cars_and_people_on_exact_clock(self):
        stage1, stage2 = self.make_bundles()
        frames = {}
        for index in range(10):
            frames[index] = [
                detection("car", 25 + 3 * index, 40,
                          confidence=0.15 if index == 4 else 0.9, size=14),
                detection("person", 120, 25 + index, 0.8, size=8),
            ]
        output = self.root / "stage3"
        report = run_stage3(
            stage1, stage2, output,
            Stage3Options(min_confirmed_observations=3), FakeDetector(frames))
        self.assertTrue(report["pass"], report["failure_reasons"])
        self.assertEqual(report["quality"]["detections"], 20)
        self.assertEqual(report["quality"]["confirmed_tracks"], 2)
        tracks = json.loads((output / "tracks.json").read_text(encoding="utf-8"))
        self.assertEqual(tracks["schema_version"], "stage3-tracks-v1")
        self.assertEqual({track["kind"] for track in tracks["tracks"]}, {"car", "person"})
        for track in tracks["tracks"]:
            self.assertEqual(track["samples"][5]["t"], 0.5)
            self.assertTrue(all(
                sample["frame"] == index for index, sample in enumerate(track["samples"])))
        self.assertTrue((output / "tracking_preview.mp4").is_file())
        self.assertTrue((output / "track_map.png").is_file())

        cached_output = self.root / "stage3_cached"
        cached_report = run_stage3(
            stage1, stage2, cached_output,
            Stage3Options(min_confirmed_observations=3),
            CachedActorDetector(output))
        self.assertTrue(cached_report["pass"])
        self.assertEqual(cached_report["quality"]["detections"], 20)
        self.assertEqual(cached_report["quality"]["confirmed_tracks"], 2)
        self.assertTrue(cached_report["detector"]["reused_detections"])

    def test_camera_translation_is_removed_before_association(self):
        frame_count = 8
        transforms = np.repeat(np.eye(3)[None, :, :], frame_count, axis=0)
        for index in range(frame_count):
            transforms[index, 0, 2] = -4.0 * index
        stage1, stage2 = self.make_bundles(frame_count=frame_count, transforms=transforms)
        frames = {
            index: [detection("car", 50 + 4 * index, 50, 0.9, 12)]
            for index in range(frame_count)
        }
        output = self.root / "stable"
        report = run_stage3(
            stage1, stage2, output, Stage3Options(), FakeDetector(frames))
        self.assertTrue(report["pass"])
        self.assertEqual(report["quality"]["confirmed_tracks"], 1)
        track = json.loads((output / "tracks.json").read_text())["tracks"][0]
        xs = [sample["x"] for sample in track["samples"]]
        self.assertLess(max(xs) - min(xs), 1e-5)


if __name__ == "__main__":
    unittest.main()
