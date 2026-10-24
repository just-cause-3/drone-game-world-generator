# Drone World Recreation — Unity project

Open this folder with Unity `6000.5.7f1`.

For a one-click balanced run, use **Drone World -> Run Full Pipeline - Fast Mode
(Stages 1-4A)...** and keep its artifacts in a separate `runs/<name>_fast` folder.
This preserves full-resolution tiled/TTA detection while Stage 3 infers alternate
frames and reconstructs the full-rate playback timeline. The existing **Run Full
Pipeline (Stages 1-4A)...** menu remains the maximum-accuracy, every-frame option.

New users should follow [`../../docs/HOW_TO_RUN.md`](../../docs/HOW_TO_RUN.md) from the
repository root for installation and the complete Stage 1 → Stage 4A procedure.

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
10-unit width. Those units are explicitly not meters. Stage 2 replaces this preview
scale using either measured ground-control points or an explicitly assumed image width.

Failed Stage 1 reports are rejected and cannot replace the preview scene.

## Stage 2 workflow

1. Import a passing Stage 1 bundle.
2. Select **Drone World → Run Stage 2 Metric Calibration...**.
3. Click fixed landmarks on the clean reference image.
4. For every clicked point, enter its measured Unity `X/Z` coordinate in metres.
5. Use at least four non-collinear points; six or more spread around the image are
   recommended.
6. Press **Validate Calibration and Build Metric Scene**.

Python validates point coverage, reprojection error, projective stability, local
distortion, and Stage 1 provenance. Unity imports only a passing calibration, replaces
the arbitrary preview plane with a metric four-corner mesh, adds a mesh collider, and
creates `Assets/Scenes/Stage2MetricWorld.unity`.

Inspect `stage2_out/calibration_overlay.png` before accepting the calibration. Four
points solve the transform but cannot independently validate measurement error, so a
four-point result is explicitly labelled `minimum_unverified`.

If no measurements are available, leave **Use approximate automatic scale** enabled,
choose a convenient complete-image width, and build the scene without clicking any
landmarks. Unity preserves the image aspect ratio and marks the calibration as assumed;
later distances and playback speeds are therefore approximate.

## Optional Stage 3 sample actors

The real Stage 3 tracking pipeline is implemented below. Two dependency-free placeholder
prefabs are also available for testing visuals without running detection:

1. Select **Drone World -> Stage 3 -> Create Sample Actor Prefabs**.
2. Select **Drone World -> Stage 3 -> Place Sample Actors In Current Scene**.
3. Press the Unity **Play** button. The car follows an oval path and the person follows
   a smaller figure-eight path with their heading aligned to movement.
4. Save the scene if you want to retain the samples.

The first command generates a 4.2 x 1.8 x 1.5 unit car and a 1.75 unit person under
`Assets/DroneWorld/Stage3/Prefabs`. The second places one of each on the current Stage 2
ground. Both roots have a ground-level pivot, a collider, and local `+Z` forward. They
are intentionally lightweight playback/collision stand-ins, not final art or measured
individual actor dimensions.

If the actors were already placed before playback was added, run **Place Sample Actors
In Current Scene** again. It upgrades the existing root instead of creating duplicates.

## Stage 3 real tracking workflow

1. Open the Stage 2 metric scene.
2. Select **Drone World -> Stage 3 -> Run Stage 3 Tracking Pipeline...**.
3. Confirm the Stage 2 calibration and `stage3_out` artifact folder.
4. Press **Run Stage 3**. Detection runs on every CFR frame and can take several minutes.
5. A passing bundle creates and opens `Assets/Scenes/Stage3ActorPlayback.unity`.
6. Inspect the external `tracking_preview.mp4` and `track_map.png`.
7. Press Unity **Play** to replay the detected car and person trajectories. Select
   **Stage 3 Real Track Playback** in the Hierarchy to use the Inspector Play/Pause,
   Reset, and timeline controls while in Play Mode.

The Stage 3 importer rechecks the Stage 1 report, CFR video, transform-stack, and Stage
2 calibration hashes before modifying Unity. A failed or stale bundle leaves the
previous scene unchanged.

## Stage 4A semantic review workflow

1. Run `setup_stage4_model.py` once from the repository root.
2. Open the Stage 2 metric scene and select **Drone World -> Stage 4 -> Run Stage 4A
   Semantic Mapping...**.
3. Run automatic mode to obtain a proposal, or select a corrected canonical one-channel
   PNG to validate and import operator labels.
4. A structural pass opens `Assets/Scenes/Stage4SemanticReview.unity`.
5. Select the review root, inspect the overlay at full resolution, and accept only an
   actually corrected map.

Stage 4A intentionally generates no geometry. Automatic mode uses OpenEarthMap aerial
semantics plus a road-only specialist; the specialist cannot propose buildings.
`pass` remains only a file/provenance/coverage gate. Acceptance is stored externally
and bound to the exact report and semantic-label hashes.
Changing either artifact invalidates approval. Stage 4B polygonization, height
hypotheses, and final Unity geometry remain future substages.

The editor integration has a batch-mode self-test (`DroneWorld.Stage1.Editor.Stage1UnitySelfTest.Run`)
that imports a temporary passing fixture, verifies the mesh/UV convention, then removes
the fixture and restores the blocked scene.
