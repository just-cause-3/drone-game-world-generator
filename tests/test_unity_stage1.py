import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import pipeline
from scripts.unity_stage1 import (
    find_unity_editor,
    import_stage1_bundle,
    unity_project_version,
    validate_unity_project,
)


def _unity_project(root: Path):
    (root / "Assets").mkdir(parents=True)
    (root / "Packages").mkdir()
    (root / "ProjectSettings").mkdir()
    (root / "Packages" / "manifest.json").write_text("{}", encoding="utf-8")
    (root / "ProjectSettings" / "ProjectSettings.asset").write_text(
        "settings", encoding="utf-8")
    (root / "ProjectSettings" / "ProjectVersion.txt").write_text(
        "m_EditorVersion: 6000.5.7f1\n", encoding="utf-8")


class UnityStage1Tests(unittest.TestCase):
    def test_project_validation_and_version(self):
        with tempfile.TemporaryDirectory() as temp_name:
            project = Path(temp_name) / "UnityProject"
            _unity_project(project)
            self.assertEqual(validate_unity_project(project), project.resolve())
            self.assertEqual(unity_project_version(project), "6000.5.7f1")

    def test_override_editor_must_exist(self):
        with tempfile.TemporaryDirectory() as temp_name:
            project = Path(temp_name) / "UnityProject"
            _unity_project(project)
            with self.assertRaises(FileNotFoundError):
                find_unity_editor(project, project / "missing-Unity.exe")

    def test_unity_command_uses_exact_project_and_bundle_paths(self):
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            project = root / "Unity Project"
            bundle = root / "Stage 1 Bundle"
            editor = root / "Unity.exe"
            _unity_project(project)
            bundle.mkdir()
            (bundle / "report.json").write_text(json.dumps({"pass": True}), encoding="utf-8")
            editor.write_bytes(b"test executable placeholder")
            completed = Mock(returncode=0)
            runner = Mock(return_value=completed)

            log_path = import_stage1_bundle(project, bundle, editor, runner=runner)

            command = runner.call_args.args[0]
            self.assertIn(str(project.resolve()), command)
            self.assertIn(str(bundle.resolve()), command)
            self.assertEqual(log_path, bundle.resolve() / "unity_import.log")
            self.assertFalse(runner.call_args.kwargs["check"])


if __name__ == "__main__":
    unittest.main()
