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
    _playback_track,
    _Track,
    YoloActorDetector,
    run_stage3,
)


class FakeDetector:
    def __init__(self, detections_by_frame):
        self.detections_by_frame = detections_by_frame
        self.calls = []
        self.metadata = {
            "model": "synthetic-fixture",
            "model_sha256": "0" * 64,
            "device": "cpu",
            "model_classes": {"0": "pedestrian", "3": "car"},
            "canonical_classes": {"0": "person", "3": "car"},
        }

    def detect(self, frame, frame_index):
        del frame
        self.calls.append(frame_index)
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

    def make_track(self, frames, positions, kind="car", confidences=None):
        items = []
        for index, (frame, position) in enumerate(zip(frames, positions)):
            x, y = position
            item = detection(kind, x, y, (confidences or [0.9] * len(frames))[index])
            item.update(frame=frame, reference_anchor=[x, y], world_anchor=[x, y],
                        reference_bbox_xyxy=item["bbox_xyxy"])
            items.append(item)
        track = _Track(1, items[0])
        for item in items[1:]:
            track.append(item)
        return track

    def test_smoothing_is_invariant_to_world_scale_and_rotation(self):
        points = [(20 + index * 2, 40 + 0.08 * index ** 2) for index in range(25)]
        track = self.make_track(range(25), points)
        options = Stage3Options()
        base = _playback_track(track, 24, options, np.eye(3))
        transform = np.array([[0, -0.15, 8], [0.15, 0, -2], [0, 0, 1]])
        mapped = _playback_track(track, 24, options, transform)
        for first, second in zip(base["samples"], mapped["samples"]):
            expected = _project_point(transform, first["x"], first["z"])
            np.testing.assert_allclose([second["x"], second["z"]], expected, atol=1e-6)
        # Ensure a turning path has not been flattened into constant velocity.
        error = np.linalg.norm(np.array([[s["x"], s["z"]] for s in base["samples"]]) - points, axis=1)
        self.assertLess(float(error.max()), 1.0)

    def test_stopped_car_keeps_previous_heading(self):
        points = [(100 - 2 * index, 50) for index in range(20)] + [(62, 50)] * 25
        track = self.make_track(range(len(points)), points)
        payload = _playback_track(track, 24, Stage3Options(), np.eye(3))
        self.assertAlmostEqual(payload["samples"][-1]["heading_deg"], -90.0, places=3)

    def test_never_moving_car_heading_uses_calibrated_axes(self):
        track = self.make_track(range(6), [(50, 50)] * 6)
        transform = np.array([[0, -0.1, 5], [0.1, 0, -5], [0, 0, 1]])
        payload = _playback_track(track, 24, Stage3Options(), transform)
        self.assertAlmostEqual(payload["samples"][0]["heading_deg"], 0.0)

    def test_smoother_cannot_push_an_actor_off_the_ground_rectangle(self):
        points = [(40 + 2 * index, 50) for index in range(20)] + [(78, 50)] * 20
        track = self.make_track(range(len(points)), points)
        payload = _playback_track(track, 24, Stage3Options(), np.eye(3), (79, 100))
        self.assertTrue(all(0 <= sample["x"] <= 78 for sample in payload["samples"]))

    def test_large_gap_is_neither_smoothed_nor_used_for_velocity(self):
        track = self.make_track([0, 1, 2, 8, 9, 10], [(20, 20)] * 3 + [(90, 90)] * 3)
        payload = _playback_track(track, 24, Stage3Options(), np.eye(3))
        self.assertEqual([0, 1, 2, 8, 9, 10], [s["frame"] for s in payload["samples"]])
        self.assertTrue(all(s["speed_units_per_s"] == 0.0 for s in payload["samples"]))
        self.assertAlmostEqual(payload["samples"][2]["x"], 20)

    def test_tracker_gap_budget_counts_missing_frames(self):
        for missing in (0, 3):
            with self.subTest(missing=missing):
                options = Stage3Options(max_lost_frames=missing, max_interpolation_frames=missing)
                tracker = StabilizedTracker(160, 100, 24, options)
                track = self.make_track([0, missing + 1], [(50, 50), (51, 50)])
                tracker.update(0, [track.observations[0]])
                for frame in range(1, missing + 1):
                    tracker.update(frame, [])
                tracker.update(missing + 1, [track.observations[1]])
                self.assertEqual(1, len(tracker.all_tracks()))

    def test_weak_person_continues_existing_track_but_cannot_start_one(self):
        tracker = StabilizedTracker(160, 100, 24, Stage3Options())
        track = self.make_track([0, 1, 2], [(50, 50), (51, 50), (52, 50)],
                                "person", [0.9, 0.15, 0.9])
        weak_new = self.make_track([1], [(100, 50)], "person", [0.4]).observations[0]
        tracker.update(0, [track.observations[0]])
        tracker.update(1, [track.observations[1], weak_new])
        tracker.update(2, [track.observations[2]])
        self.assertEqual(1, len(tracker.all_tracks()))
        self.assertEqual(3, len(tracker.all_tracks()[0].observations))
        self.assertNotIn("track_id", weak_new)

    def test_actual_yolo_adapter_retains_weak_people_for_association(self):
        # Exercise the production adapter without downloading/loading a model.
        class Tensor:
            def __init__(self, values): self.values = np.asarray(values)
            def cpu(self): return self
            def numpy(self): return self.values

        class Boxes:
            xyxy = Tensor([[10, 10, 20, 20]])
            cls = Tensor([0])
            conf = Tensor([0.15])

        class Result:
            boxes = Boxes()

        class Model:
            def predict(self, *args, **kwargs): return [Result()]

        detector = YoloActorDetector.__new__(YoloActorDetector)
        detector.options = Stage3Options()
        detector.model = Model()
        detector.class_kinds = {0: "person"}
        detector.model_names = {0: "pedestrian"}
        detector.device = "cpu"
        candidates = detector._predict(np.zeros((48, 64, 3), np.uint8), 0, 0, "full")
        self.assertEqual(1, len(candidates))
        self.assertAlmostEqual(0.15, candidates[0]["confidence"])

    def test_rejects_nonfinite_options_and_unbridgeable_stride(self):
        for options in (Stage3Options(measurement_noise=float("nan")),
                        Stage3Options(detector_frame_stride=5)):
            with self.assertRaises(ValueError):
                options.validate()


if __name__ == "__main__":
    unittest.main()
