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
        self.assertFalse(report["end_to_end_complete"])
        self.assertFalse(report["unity_import_completed"])
        self.assertEqual("stages_1_to_4a", report["processing_scope"])
        self.assertIn("stage4b_c_geometry", report["pending_steps"])
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

    def test_invalid_scale_rejected_before_creating_run_or_processing(self):
        for width in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(width=width), self.assertRaisesRegex(ValueError, "finite and positive"):
                run_full_pipeline(self.video, self.run_root, width)
            self.assertFalse(self.run_root.exists())

    def test_every_gate_and_exception_stops_downstream_stages(self):
        for failing in (1, 2, 3, 4):
            for crash in (False, True):
                with self.subTest(failing=failing, crash=crash):
                    calls = []
                    def runner(stage):
                        def run(*args):
                            calls.append(stage)
                            if stage == failing and crash:
                                raise RuntimeError("fixture failure")
                            return {"pass": stage != failing, "input": {"width": 32, "height": 24},
                                    "failure_reasons": ["fixture rejection"] if stage == failing else []}
                        return run
                    with self.assertRaises(RuntimeError if crash else PipelineGateFailure):
                        run_full_pipeline(self.video, self.run_root,
                                          **{f"stage{s}_runner": runner(s) for s in (1, 2, 3, 4)})
                    self.assertEqual(calls, list(range(1, failing + 1)))
                    report = json.loads((self.run_root / "pipeline_report.json").read_text())
                    self.assertFalse(report["pass"])
                    self.assertEqual(failing, report["failed_stage"])
                    self.assertEqual("error" if crash else "quality_gate_failed", report["status"])

    def test_refuses_unowned_nonempty_run_directory(self):
        self.run_root.mkdir()
        (self.run_root / "unrelated.txt").write_text("do not replace", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unrelated non-empty"):
            run_full_pipeline(self.video, self.run_root)
        self.assertEqual(
            "do not replace",
            (self.run_root / "unrelated.txt").read_text(encoding="utf-8"))

    def test_refuses_reusing_run_folder_for_another_video(self):
        second = self.root / "second.mp4"
        second.write_bytes(b"another")
        self.run_root.mkdir()
        (self.run_root / RUN_MARKER).write_text(
            json.dumps({"schema_version": PIPELINE_SCHEMA, "video": str(second)}),
            encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "different input video"):
            run_full_pipeline(self.video, self.run_root)

    def test_fast_and_accuracy_modes_cannot_share_a_run_folder(self):
        self.run_root.mkdir()
        (self.run_root / RUN_MARKER).write_text(
            json.dumps({
                "schema_version": PIPELINE_SCHEMA,
                "video": str(self.video.resolve()),
                "pipeline_mode": "fast_balanced",
            }),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "separate folder"):
            run_full_pipeline(self.video, self.run_root)


if __name__ == "__main__":
    unittest.main()
