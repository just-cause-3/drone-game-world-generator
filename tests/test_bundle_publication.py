import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import stage1_ground, stage2_calibration, stage3_tracking, stage4_semantics, stage4_geometry


class BundlePublicationTests(unittest.TestCase):
    def test_nonfinite_values_cannot_disable_stage_gates(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            options = (
                stage1_ground.Stage1Options(max_frame_scale_step_pct=value),
                stage2_calibration.Stage2Options(max_reprojection_rmse_m=value),
                stage3_tracking.Stage3Options(measurement_noise=value),
                stage4_semantics.Stage4Options(fragmentation_warning_components_per_megapixel=value),
                stage4_geometry.Stage4GeometryOptions(default_building_height_m=value),
            )
            for option in options:
                with self.subTest(stage=type(option).__name__, value=value):
                    with self.assertRaisesRegex(ValueError, "finite"):
                        option.validate()

    def test_failed_promotion_restores_previous_bundle_for_every_downstream_stage(self):
        for stage in (stage2_calibration, stage3_tracking, stage4_semantics, stage4_geometry):
            with self.subTest(stage=stage.__name__), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                output, incoming = root / "output", root / "incoming"
                output.mkdir()
                incoming.mkdir()
                (output / stage.OUTPUT_MARKER).write_text("owned")
                (output / "valuable.txt").write_text("previous passing result")
                (incoming / "valuable.txt").write_text("new result")
                replace = os.replace

                def fail_promotion(source, target):
                    if Path(source) == incoming:
                        raise PermissionError("simulated locked destination")
                    return replace(source, target)

                with patch("scripts.common.os.replace", side_effect=fail_promotion):
                    with self.assertRaises(PermissionError):
                        stage._publish_directory(incoming, output)
                self.assertEqual("previous passing result", (output / "valuable.txt").read_text())
                self.assertTrue(incoming.is_dir())

    def test_success_does_not_delete_a_preexisting_previous_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, incoming = root / "output", root / "incoming"
            output.mkdir()
            incoming.mkdir()
            (output / stage4_semantics.OUTPUT_MARKER).write_text("owned")
            (incoming / "result.txt").write_text("new result")
            unrelated = root / ".output-previous"
            unrelated.mkdir()
            (unrelated / "keep.txt").write_text("keep")
            stage4_semantics._publish_directory(incoming, output)
            self.assertEqual("new result", (output / "result.txt").read_text())
            self.assertEqual("keep", (unrelated / "keep.txt").read_text())


if __name__ == "__main__":
    unittest.main()
