"""Unity-side publication for a passing Stage 1 artifact bundle."""
from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess


UNITY_IMPORT_METHOD = (
    "DroneWorld.Stage1.Editor.Stage1BundleImporter.ImportFromCommandLine"
)


def validate_unity_project(project_dir) -> Path:
    project = Path(project_dir).resolve()
    required = [
        project / "Assets",
        project / "Packages" / "manifest.json",
        project / "ProjectSettings" / "ProjectSettings.asset",
        project / "ProjectSettings" / "ProjectVersion.txt",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise ValueError("not a valid initialized Unity project; missing: " + ", ".join(missing))
    return project


def unity_project_version(project_dir) -> str:
    project = validate_unity_project(project_dir)
    contents = (project / "ProjectSettings" / "ProjectVersion.txt").read_text(
        encoding="utf-8")
    match = re.search(r"^m_EditorVersion:\s*(\S+)\s*$", contents, re.MULTILINE)
    if not match:
        raise ValueError("Unity ProjectVersion.txt has no m_EditorVersion")
    return match.group(1)


def find_unity_editor(project_dir, override=None) -> Path:
    if override:
        editor = Path(override).resolve()
        if not editor.is_file():
            raise FileNotFoundError(f"Unity editor not found: {editor}")
        return editor

    version = unity_project_version(project_dir)
    candidates = []
    if os.name == "nt":
        program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        candidates.append(program_files / "Unity" / "Hub" / "Editor" / version /
                          "Editor" / "Unity.exe")
    elif os.uname().sysname == "Darwin":
        candidates.append(Path("/Applications/Unity/Hub/Editor") / version /
                          "Unity.app/Contents/MacOS/Unity")
    else:
        candidates.append(Path.home() / "Unity" / "Hub" / "Editor" / version /
                          "Editor" / "Unity")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Unity editor {version} is not installed at the expected Hub location; "
        "pass --unity-editor explicitly")
