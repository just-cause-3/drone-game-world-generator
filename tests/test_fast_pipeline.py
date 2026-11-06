import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from full_pipeline import _write_approximate_control_points
from scripts.stage2_calibration import run_stage2
from scripts.stage3_tracking import run_stage3

from fast_pipeline import (
    FAST_PIPELINE_MODE,
    _stage4_resume_runners,
    fast_stage3_options,
)


class FastPipelineTests(unittest.TestCase):
    def test_balanced_preset_only_reduces_detection_cadence(self):
        options = fast_stage3_options()
        self.assertEqual(FAST_PIPELINE_MODE, "fast_balanced")
        self.assertEqual(options.detector_frame_stride, 2)
        self.assertEqual(options.detector_imgsz, 1280)
        self.assertEqual(options.detector_tile_size, 640)
        self.assertTrue(options.detector_full_frame)
        self.assertTrue(options.detector_tta)
        self.assertEqual(options.detector_car_confidence, 0.20)
        self.assertEqual(options.detector_person_confidence, 0.60)

    def test_failed_stage4_run_can_reuse_passing_first_three_stages(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "input.mp4"
            video.write_bytes(b"video")
            video_hash = hashlib.sha256(b"video").hexdigest()
            run = root / "run"
            for stage in (1, 2, 3):
                (run / f"stage{stage}").mkdir(parents=True)
            (run / "pipeline_report.json").write_text(json.dumps({
                "status": "error",
                "pipeline_mode": FAST_PIPELINE_MODE,
                "assumed_world_width_m": 40.0,
                "video": {"path": str(video), "sha256": video_hash},
            }))
            stage1 = run / "stage1"
            image = np.full((48, 64, 3), 100, np.uint8)
            cv2.imwrite(str(stage1 / "ground.png"), image)
            writer = cv2.VideoWriter(str(stage1 / "video_cfr.avi"),
                                     cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
            self.assertTrue(writer.isOpened())
            for _ in range(6):
                writer.write(image)
            writer.release()
            np.save(stage1 / "transforms.npy", np.repeat(np.eye(3)[None], 6, axis=0))
            (run / "stage1" / "report.json").write_text(json.dumps({
                "schema_version": "stage1-ground-v1",
                "stage": 1, "pass": True,
                "input": {"sha256": video_hash, "width": 64, "height": 48,
                          "fps": 10.0, "frame_count": 6},
                "reference": {"video_frame": 3},
                "outputs": {"ground": "ground.png", "video_cfr": "video_cfr.avi",
                            "frame_transforms": "transforms.npy"},
            }))
            controls = run / "controls.json"
            _write_approximate_control_points(controls, 64, 48, 40.0)
            run_stage2(stage1, controls, run / "stage2")

            class EmptyDetector:
                metadata = {"model": "fixture"}

                def detect(self, frame, frame_index):
                    return []

            run_stage3(stage1, run / "stage2", run / "stage3",
                       fast_stage3_options(), EmptyDetector())

            runners = _stage4_resume_runners(video, run)
            self.assertIsNotNone(runners)
            self.assertTrue(runners[0](None, None, None)["pass"])
            self.assertTrue(runners[1](None, None, None, None)["pass"])
            self.assertTrue(runners[2](None, None, None, None)["pass"])
            self.assertIsNone(_stage4_resume_runners(video, run, 80.0))

            tracks = run / "stage3" / "tracks.json"
            original_tracks = tracks.read_bytes()
            tracks.write_bytes(original_tracks + b"\n")
            self.assertIsNone(_stage4_resume_runners(video, run))
            tracks.write_bytes(original_tracks)

            transforms_path = stage1 / "transforms.npy"
            original_transforms = transforms_path.read_bytes()
            transforms = np.load(transforms_path)
            transforms[:, 0, 2] = 1.0
            np.save(transforms_path, transforms)
            self.assertIsNone(_stage4_resume_runners(video, run))
            transforms_path.write_bytes(original_transforms)

            (run / "stage3" / "track_map.png").unlink()
            self.assertIsNone(_stage4_resume_runners(video, run))


if __name__ == "__main__":
    unittest.main()
