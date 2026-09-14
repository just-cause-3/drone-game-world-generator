# Drone World Recreation — Unity project

Open this folder with Unity `6000.5.7f1`.

This is the clean Unity project for the staged rebuild. It deliberately does not reuse
the assets, coordinate conversions, tracks, or scenes from `UnityProject/My project`.

## Stage 1 workflow

1. In Unity, select **Drone World → Run Stage 1 Pipeline...**.
2. Choose a stationary, top-down video with bounded zoom and press **Run Stage 1**.
3. The editor runs the repository Python environment without blocking the Unity UI.
4. A failed gate displays its reasons and leaves the previous Unity scene untouched.
5. On a pass, Unity copies the validated static plate and review masks into an ignored
   `Assets/Generated/Stage1_<source-hash>` directory and creates
   `Assets/Scenes/Stage1Preview.unity`.

For command-line automation with Unity closed, run from the repository root:

```powershell
.venv\Scripts\python.exe pipeline.py stage1 path\to\video.mp4
```

The lower-level **Import Passing Stage 1 Bundle...** menu remains available for importing
an already-created bundle without rerunning computer vision.

The preview surface preserves the Stage 1 top-left pixel convention but uses a temporary
10-unit width. Those units are explicitly not meters. The next stage will replace this
preview scale using `world_calibration.json` and at least four ground control points.

Failed Stage 1 reports are rejected and cannot replace the preview scene.

The editor integration has a batch-mode self-test (`DroneWorld.Stage1.Editor.Stage1UnitySelfTest.Run`)
that imports a temporary passing fixture, verifies the mesh/UV convention, then removes
the fixture and restores the blocked scene.
