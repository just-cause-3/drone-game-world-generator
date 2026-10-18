import hashlib
import json
from pathlib import Path
import tempfile
import unittest

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
                "video": {"path": str(video), "sha256": video_hash},
            }))
            (run / "stage1" / "report.json").write_text(json.dumps({
                "schema_version": "stage1-ground-v1",
                "pass": True,
                "input": {"sha256": video_hash, "width": 10, "height": 10},
            }))
            (run / "stage2" / "world_calibration.json").write_text(json.dumps({
                "schema_version": "stage2-calibration-v1", "pass": True,
            }))
            (run / "stage3" / "report.json").write_text(json.dumps({
                "schema_version": "stage3-tracks-v1",
                "pass": True,
                "options": {"detector_frame_stride": 2},
            }))

            runners = _stage4_resume_runners(video, run)
            self.assertIsNotNone(runners)
            self.assertTrue(runners[0](None, None, None)["pass"])
            self.assertTrue(runners[1](None, None, None, None)["pass"])
            self.assertTrue(runners[2](None, None, None, None)["pass"])


if __name__ == "__main__":
    unittest.main()
