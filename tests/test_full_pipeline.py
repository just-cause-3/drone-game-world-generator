import json
from pathlib import Path
import tempfile
import unittest

from full_pipeline import (
    PIPELINE_SCHEMA,
    PipelineGateFailure,
    RUN_MARKER,
    run_full_pipeline,
)


class FullPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.video = self.root / "input.mp4"
        self.video.write_bytes(b"fixture-video")
        self.run_root = self.root / "run"

    def tearDown(self):
        self.temporary.cleanup()

    def test_runs_all_stages_in_order_and_writes_approximate_scale(self):
        calls = []

        def stage1(video, output, options):
            calls.append((1, Path(video), Path(output)))
            return {
                "pass": True, "failure_reasons": [], "warnings": [],
                "input": {"width": 1280, "height": 720},
            }

        def stage2(stage1_bundle, controls, output, options):
            calls.append((2, Path(stage1_bundle), Path(output)))
            data = json.loads(Path(controls).read_text(encoding="utf-8"))
            self.assertEqual("approximate_image_extent", data["calibration_mode"])
            self.assertEqual(50.0, data["assumed_world_width_m"])
            self.assertEqual(4, len(data["points"]))
            self.assertEqual(
                {(-25.0, 14.0625), (25.0, 14.0625),
                 (25.0, -14.0625), (-25.0, -14.0625)},
                {(point["world"]["x"], point["world"]["z"])
                 for point in data["points"]})
            return {"pass": True, "failure_reasons": [], "warnings": []}

        def stage3(stage1_bundle, stage2_bundle, output, options):
            calls.append((3, Path(stage2_bundle), Path(output)))
            return {"pass": True, "failure_reasons": [], "warnings": ["fixture"]}

        def stage4(stage1_bundle, stage2_bundle, output, options):
            calls.append((4, Path(stage2_bundle), Path(output)))
            return {"pass": True, "failure_reasons": [], "warnings": []}

        report = run_full_pipeline(
            self.video, self.run_root, 50.0,
            stage1_runner=stage1, stage2_runner=stage2,
            stage3_runner=stage3, stage4_runner=stage4)

        self.assertEqual([1, 2, 3, 4], [item[0] for item in calls])
        self.assertTrue(report["pass"])
        self.assertEqual("pass_review_required", report["status"])
        self.assertEqual("accuracy", report["pipeline_mode"])
        self.assertTrue(report["requires_stage4_human_review"])
        self.assertFalse(report["ready_for_stage4_geometry"])
        saved = json.loads(
            (self.run_root / "pipeline_report.json").read_text(encoding="utf-8"))
        self.assertEqual(PIPELINE_SCHEMA, saved["schema_version"])
        self.assertTrue((self.run_root / RUN_MARKER).is_file())

    def test_stops_at_first_failed_gate(self):
        calls = []

        def stage1(video, output, options):
            calls.append(1)
            return {
                "pass": False,
                "failure_reasons": ["camera cut detected"],
                "warnings": [],
                "input": {"width": 100, "height": 100},
            }

        def must_not_run(*args, **kwargs):
            calls.append(99)
            raise AssertionError("a downstream stage ran after rejection")

        with self.assertRaises(PipelineGateFailure) as caught:
            run_full_pipeline(
                self.video, self.run_root,
                stage1_runner=stage1, stage2_runner=must_not_run,
                stage3_runner=must_not_run, stage4_runner=must_not_run)
        self.assertEqual(1, caught.exception.stage)
        self.assertEqual([1], calls)
        report = json.loads(
            (self.run_root / "pipeline_report.json").read_text(encoding="utf-8"))
        self.assertEqual("quality_gate_failed", report["status"])
        self.assertEqual(1, report["failed_stage"])
        self.assertEqual(["camera cut detected"], report["failure_reasons"])


if __name__ == "__main__":
    unittest.main()
