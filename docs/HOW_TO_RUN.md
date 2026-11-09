# Beginner guide: install and run the drone-video-to-Unity pipeline

For a simpler numbered walkthrough, start with [howtorun.md](../howtorun.md).
This document keeps the detailed per-stage instructions and reference options.

This is the authoritative setup and run guide for the current staged project. It
covers the existing Stage 1, Stage 2, Stage 3, and Stage 4A-C implementation. The legacy workflow
later in the repository's main README is retained only as design reference and should
not be used with the current Unity project.

## What the project currently produces

The input is one pre-recorded, nearly stationary, top-down drone video. The current
pipeline produces:

1. **Stage 1 — static ground:** a validated, cleaned top-down ground image with cars
   and people removed, plus the camera transforms needed by later stages.
2. **Stage 2 — Unity coordinates:** a mapping from image pixels to Unity `X/Z`
   coordinates. This may use an approximate width when no measurements are available.
3. **Stage 3 — actor playback:** detected cars and people placed on the ground and
   replayed on the source video's exact frame clock.
4. **Stage 4A — semantic review:** a provisional pixel map of roads, pavement,
   buildings, vegetation, terrain, water, and other static content, displayed in a
   Unity review scene and blocked from geometry until explicit operator acceptance.
5. **Stage 4B-C — provisional 3D world:** accepted building regions are vectorized
   and extruded, flat semantic regions remain registered to the Stage 2 ground, and
   optional vegetation placeholders can be generated. Every height is visibly marked
   as assumed.

Stage 4D appearance/prop replacement and Stage 4E final spatial QA remain future work.
Fixed top-down monocular footage does not contain enough parallax to measure building
height, so current height values remain visibly marked as assumed unless another
evidence source is supplied.

## Quick run checklist

If the machine is already configured, the normal run is:

1. Open `UnityProject/DroneWorldRecreation` in Unity `6000.5.7f1`.
2. Select **Drone World → Run Full Pipeline (Stages 1-4A)...**.
3. Select the video, leave the isolated run folder under `runs/`, and enter an
   approximate image width. `40 m` is the default when no measurements exist; it is
   an explicit scale assumption, not an estimate.
4. Click **Run Stages 1-4A**. The command stops at the first failed quality gate and
   imports nothing into Unity unless all four processing stages pass.
5. Inspect all review artifacts. Accept Stage 4A only after labels are corrected.
6. From the accepted review root, open the Stage 4B-C generator and create the
   provisional 3D world.
7. Use `Stage3ActorPlayback.unity` for isolated playback,
   `Stage4SemanticReview.unity` for label review, and `Stage4GeneratedWorld.unity`
   for the composed 3D preview with actor playback.

Each video gets separate artifacts at `runs/<video-name>/stage1` through `stage4`, so
trying another clip does not overwrite the earlier `stage1_out`-style folders. A run
folder is bound to its original video and cannot silently be reused for another one.

The older per-stage menus remain available for diagnosis, manual measured Stage 2
calibration, and rerunning only one expensive stage.

### Optional balanced fast one-click run

For a substantially shorter Stage 3 run without lowering aerial resolution or
disabling tiles/TTA, select **Drone World -> Run Full Pipeline - Fast Mode (Stages
1-4A)...**. Choose a separate folder such as `runs/crossroads_fast`, then click
**Run Stages 1-4A - Fast**.

Fast Mode performs Stage 3 detector inference on frames 0, 2, 4, and so on (and
always the final frame). The tracker still advances on every frame, and the offline
smoother interpolates the skipped positions, so the exported preview and Unity
playback keep the source video's full frame rate and duration. Cars or people visible
for only a single skipped frame can be missed; use the original maximum-accuracy menu
when that edge case matters.

The equivalent command is:

```powershell
.venv\Scripts\python.exe fast_pipeline.py sample_vids\crossroads.mp4 --run-root runs\crossroads_fast --assumed-world-width 40
```

The run-folder marker prevents an accuracy run and a fast run from overwriting each
other. The original `full_pipeline.py` script, its Unity menu, and all per-stage menus
remain available and keep their previous accuracy settings.

If a Fast Mode run has already passed Stages 1-3 and then encounters a Stage 4
processing error, press the fast run button again with the same video and run folder.
It validates and reuses the passing bundles, then resumes at Stage 4. If Python has
already completed the recovery outside Unity, use **Import Existing Passing Fast Run**
to import all four validated bundles without rerunning processing.

The rest of this document explains the one-time installation and each check in detail.

## 1. Use suitable footage

Start with a short 10–30 second test clip. The footage should satisfy all of these
conditions:

- The drone is looking almost straight down.
- The drone mostly hovers in one place. Small hover drift is acceptable.
- Slow, smooth zoom is acceptable.
- There are no cuts, transitions, sudden zoom jumps, or abrupt camera movement.
- The same ground area remains substantially visible throughout the clip.
- Roads and actors are sufficiently sharp to see.
- Use one continuous shot, preferably MP4, MOV, AVI, MKV, or M4V.

The pipeline intentionally rejects unsuitable footage instead of modifying Unity with
an unreliable result.

## 2. Install the prerequisites

The current tested environment is:

- Windows 10 or 11, 64-bit;
- Python 3.12, 64-bit;
- Unity `6000.5.7f1` (Unity 6.5);
- FFmpeg and FFprobe available on `PATH`;
- an NVIDIA CUDA-capable GPU is strongly recommended.

CPU execution is supported, but the accuracy-first full-frame, tiled, test-time-
augmentation detector can be extremely slow on CPU.

### 2.1 Install Python 3.12

Install Python 3.12 (64-bit) from <https://www.python.org/downloads/>. Enable
**Add Python to PATH** in the installer.

Open a new PowerShell window and verify it:

```powershell
py -3.12 --version
```

The command should print `Python 3.12.x`.

### 2.2 Install FFmpeg

With Windows Package Manager:

```powershell
winget install Gyan.FFmpeg
```

Close and reopen PowerShell after installation, then verify both programs:

```powershell
ffmpeg -version
ffprobe -version
```

Both commands must print version information.

### 2.3 Install Unity

1. Install Unity Hub.
2. In Unity Hub, open **Installs** and install Editor version `6000.5.7f1`.
3. Sign in and activate a Unity license if Hub requests it.
4. A platform build-support module is not required merely to run the project in the
   Editor. Install Windows Build Support later if an executable build is needed.

Do not create a blank Unity project. A configured project already exists inside this
repository.

## 3. Set up the Python environment once

Keep the Unity project inside the repository. The Unity tools locate Python by walking
from `UnityProject/DroneWorldRecreation` back to the repository root; moving only the
Unity subfolder will break that connection.

Open PowerShell and move to the repository root. Replace the example path if the
repository is somewhere else:

```powershell
Set-Location "D:\drone-game-world-generator-usin-cv"
```

Create the virtual environment:

```powershell
py -3.12 -m venv .venv
```

You do not have to activate it. The commands below and the Unity menus use its Python
executable directly.

### 3.1 Install PyTorch

For the tested NVIDIA CUDA 12.6 environment:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

If that CUDA build is incompatible with the installed driver, use the command provided
by the official selector at <https://pytorch.org/get-started/locally/>. For a deliberate
CPU-only installation:

```powershell
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 3.2 Install the remaining packages

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3.3 Download and verify the pinned aerial model

```powershell
.\.venv\Scripts\python.exe setup_drone_model.py
```

This downloads `models/visdrone-yolo26x.pt`, verifies its byte size and SHA-256, and
checks its exact VisDrone class map. Future verification does not redownload it:

```powershell
.\.venv\Scripts\python.exe setup_drone_model.py --verify-only
```

Expected final output includes:

```text
Model classes verified: pedestrian, people, bicycle, car, ...
```

### 3.4 Download and verify the Stage 1 completion model

```powershell
.\.venv\Scripts\python.exe setup_inpainting_model.py
.\.venv\Scripts\python.exe setup_inpainting_model.py --verify-only
```

This installs the checksum-pinned OpenCV LaMa ONNX model. Stage 1 uses it only when a
persistent car or person prevents temporal reconstruction and the missing component is
too large for conservative OpenCV inpainting. Its pixels are synthesized estimates,
recorded separately in `learned_inpaint_mask.png`, and must be inspected.

### 3.5 Download and verify the Stage 4 aerial semantic models

```powershell
.\.venv\Scripts\python.exe setup_stage4_model.py
.\.venv\Scripts\python.exe setup_stage4_model.py --verify-only
```

This installs a checksum-pinned OpenEarthMap Mask2Former-Swin model plus a narrowly
scoped road specialist. OpenEarthMap owns building, pavement, vegetation, terrain and
water. The specialist may only change a proposed pixel to `road`; its building and
other predictions are discarded. Both models remain proposal tools, not ground truth.

### 3.6 Verify GPU access

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU mode')"
```

`CUDA: True` means the detector will use the NVIDIA GPU. `CUDA: False` is not a setup
failure, but Stage 1 and especially Stage 3 will be much slower.

### 3.7 Run the automated tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

All tests should finish with `OK`. These tests validate the Python artifact contracts,
camera normalization, calibration rules, tracking, caching, and Unity command setup.

## 4. Open the correct Unity project

1. Open Unity Hub.
2. Select **Projects**.
3. Click **Add** or **Add project from disk**.
4. Select this exact folder:

   ```text
   <repository>\UnityProject\DroneWorldRecreation
   ```

5. Open it with Unity `6000.5.7f1`.
6. Wait for package import and C# compilation to finish.

The selected folder is correct only if it directly contains `Assets`, `Packages`, and
`ProjectSettings`. Do not select the repository root and do not select only the
`UnityProject` parent folder.

An empty or waiting scene on the first opening is normal. The pipeline stages create
the actual scenes. When compilation finishes, Unity's top menu should contain
**Drone World**.

## 5. Run Stage 1: video to validated static ground

Stage 1 is the required first step for every new video.

1. In Unity, remain in Edit Mode; the Play button at the top must not be blue.
2. Select **Drone World → Run Stage 1 Pipeline...**.
3. Next to **Input video**, click **Browse** and select the drone video.
4. Leave **Artifact folder** at the default repository path ending in `stage1_out`.
5. Click **Run Stage 1**.
6. Keep Unity open and wait. The log shows video metadata, camera checks, detector
   setup, registration, coverage, blur, motion, and the final gate result.

On success the status is:

```text
PASS — Stage1Preview.unity was created and opened.
```

Unity imports the passing bundle and opens:

```text
Assets/Scenes/Stage1Preview.unity
```

The important external artifacts are:

```text
stage1_out/report.json
stage1_out/ground.png
stage1_out/detection_preview.png
stage1_out/reference_foreground_mask.png
stage1_out/temporal_variation.png
stage1_out/video_cfr.mp4
stage1_out/frame_transforms_to_reference.npy
```

Before continuing, inspect `ground.png`. The roads and static ground should be clear,
and cars/people should not remain baked into the ground texture. Also inspect
`detection_preview.png` and the masks. A report can enforce measurable rules, but it
cannot guarantee that a detector found every tiny actor.

If Stage 1 fails, it does not replace the Unity scene. Read the red failure reasons in
the Stage 1 window or open `stage1_out/report.json` and inspect `failure_reasons`.

## 6. Run Stage 2: define Unity ground coordinates

Stage 2 defines the size, origin, and orientation of the flat Unity ground. It does not
detect landmarks automatically. There are two modes.

### Recommended beginner mode: approximate scale

Use this mode when no GPS, survey measurements, altitude, or camera calibration is
available.

1. Select **Drone World → Run Stage 2 Metric Calibration...**.
2. The imported Stage 1 manifest should be selected automatically. If it is blank,
   click **Find Imported Stage 1 Manifest**.
3. Enable **Use approximate automatic scale (no measurements required)**.
4. Enter an **Assumed world width**. For example, `40` makes the complete image 40
   Unity units wide. A 16:9 image then becomes approximately 40 × 22.5 units.
5. Leave **Ground Y** at `0` unless the project requires another ground elevation.
6. Leave the output path ending in `stage2_out`.
7. Click **Build Approximate-Scale Scene**.

On success Unity opens:

```text
Assets/Scenes/Stage2MetricWorld.unity
```

The status reports confidence `approximate_assumed_scale`. An RMSE of `0.0000` in this
mode does not prove real-world accuracy; it only means the automatically generated
image-corner mapping is internally exact. Distances, prefab sizes, and speeds remain
approximate.

Inspect:

```text
stage2_out/world_calibration.json
stage2_out/calibration_overlay.png
```

### Optional measured mode

Use this only when matching ground coordinates are genuinely known.

1. Disable **Use approximate automatic scale**.
2. Click at least four well-spread, non-collinear fixed ground landmarks. Six or more
   distributed points are recommended.
3. For every clicked pixel, enter the landmark's separately measured Unity/real-world
   `X` and `Z` coordinates.
4. Every landmark must use its own coordinate. Only one point may be `(0, 0)`.
5. Click **Validate Calibration and Build Metric Scene**.

Poles, road-marking corners, curb corners, or manholes can be landmarks only when their
matching coordinates have actually been measured. Clicking an image point does not
automatically determine a world coordinate.

## 7. Run Stage 3: detect and replay cars and people

Stage 3 consumes the exact passing Stage 1 and Stage 2 bundles. It detects only cars
and people; trucks, vans, buses, bicycles, and motorcycles are excluded by the current
project policy.

1. Make sure `Stage2MetricWorld.unity` is open.
2. Select **Drone World → Stage 3 → Run Stage 3 Tracking Pipeline...**.
3. Confirm that the Stage 2 calibration field is populated.
4. Leave the artifact path ending in `stage3_out`.
5. Click **Run Stage 3**.
6. Wait for all CFR frames to finish. The accuracy-first detector performs full-frame,
   overlapping-tile, and test-time-augmentation inference. Runtime scales roughly with
   video length and can be many minutes even on a GPU.

On success Unity validates the full provenance chain and opens:

```text
Assets/Scenes/Stage3ActorPlayback.unity
```

Review these external artifacts before accepting the scene:

```text
stage3_out/report.json
stage3_out/tracking_preview.mp4
stage3_out/track_map.png
stage3_out/frame_counts.csv
stage3_out/tracks.json
```

In `tracking_preview.mp4`, track numbers are online candidate identities; a number by
itself does not guarantee that the final segment passed confirmation. Very short or
low-confidence observations may be labelled `unassigned`. Only the confirmed segments
in `tracks.json` become Unity actors. In `track_map.png`, trajectories should follow
plausible roads and pedestrian paths without large identity teleports.

### Play the result in Unity

1. Press the main Unity **Play** button.
2. Select **Stage 3 Real Track Playback** in the Hierarchy.
3. In its Inspector use **Play/Pause**, **Reset**, or the timeline slider.
4. Exit Play Mode before changing pipeline setup or rebuilding scenes.

The playback uses one clock derived from the Stage 1 CFR video:

```text
time = source frame number / source FPS
```

### Import an existing passing Stage 3 bundle

If `stage3_out` already contains a passing run, avoid repeating inference:

1. Open `Stage2MetricWorld.unity`.
2. Select **Drone World → Stage 3 → Import Passing Stage 3 Bundle...**.
3. Select the `stage3_out` folder.

Unity rechecks the Stage 1 report, CFR video, frame transforms, Stage 2 calibration,
timeline, and track values before creating the playback scene.

## 8. Run Stage 4A: create and review static semantic labels

Stage 4A uses the Stage 1 clean reference ground and the Stage 2 coordinate mapping.
It does not depend on Stage 3, so Stage 3 and Stage 4A may be rerun independently once
Stages 1 and 2 are fixed.

1. Exit Play Mode and open `Stage2MetricWorld.unity`.
2. Select **Drone World → Stage 4 → Run Stage 4A Semantic Mapping...**.
3. Confirm the Stage 2 calibration and leave the artifact folder ending in
   `stage4_out`.
4. Leave **Corrected labels (optional)** empty for the first automatic proposal.
5. Click **Run Stage 4A**. Full-frame, overlapping-tile, and horizontal-flip consensus
   prioritizes proposal quality over speed.
6. On a structural pass, Unity opens:

   ```text
   Assets/Scenes/Stage4SemanticReview.unity
   ```

7. Select **Drone World - Stage 4A Semantic Review (NO GEOMETRY)** in the Hierarchy.
   Use its Inspector to show/hide the overlay or reveal the full-resolution PNG.
8. Compare every colored area to the original ground. In particular, check road versus
   building, pavements, small vegetation, shadows, roofs, and image borders.

Important: `"pass": true` means only that the output files and coverage are
structurally usable. It does not mean semantic accuracy. The aerial model is a major
domain improvement, but resolution, geography, shadows and rendered/synthetic footage
can still differ from its training distribution. Inspect the complete overlay.

The canonical label image is a one-channel PNG at exactly the Stage 1 dimensions. Each
pixel stores one of these integer IDs:

| ID | Meaning |
|---:|---|
| 0 | unknown / needs review |
| 1 | road |
| 2 | pavement / sidewalk |
| 3 | building |
| 4 | vegetation |
| 5 | terrain / soil |
| 6 | water |
| 7 | other static object |

To use a corrected map, edit/export that indexed value image in a suitable annotation
tool without resizing it. Then choose it in **Corrected labels (optional)** and rerun
Stage 4A. The Python pipeline validates dimensions, label IDs, valid-mask bounds,
provenance, and artifacts before Unity imports it.

Only after the overlay is correct, click **Accept Exact Semantic Map for Stage 4B** in
the review root's Inspector. This writes `stage4_out/semantic_acceptance.json` bound to
the exact SHA-256 of `report.json` and `semantic_labels.png`. Re-running or changing
either file invalidates the acceptance. Acceptance does not certify heights and does
not create geometry.

### 8.1 Generate the accepted provisional 3D world

Do not perform these steps if the automatic colors are visibly wrong. Correct the
canonical PNG and rerun Stage 4A first.

1. Select the accepted **Stage 4A Semantic Review** root in the Hierarchy.
2. In its Inspector, click **Open Stage 4B-C 3D Geometry Generator**. The same window
   is available from **Drone World → Stage 4 → Generate Accepted 3D World
   (Stage 4B-C)...**.
3. Keep the default geometry output inside the run, normally
   `runs/<name>/stage4/geometry`.
4. Choose a height policy:
   - **Fixed** uses the displayed operator-configured height for every building and is
     the safest default when no height evidence exists.
   - **Footprint Prior** varies assumed floor count from roof area. It can look more
     natural but is not more physically accurate.
5. Leave vegetation placeholders disabled unless the accepted vegetation mask really
   represents tree canopy rather than grass or general green space.
6. Click **Generate and Import Provisional 3D World**.
7. On success Unity opens:

   ```text
   Assets/Scenes/Stage4GeneratedWorld.unity
   ```

8. Inspect the scene from the perspective Game camera and in Scene view. Select the
   generated-world root to toggle buildings, vegetation, and the accepted semantic
   overlay. Press Play to run the matching Stage 3 actor playback over the world.

The generator refuses missing, revoked, or stale acceptance; invalid polygons;
escaping or corrupted artifacts; mismatched Stage 2 coordinates; and unsafe output
replacement. Building meshes include roof/wall geometry and mesh colliders. The Stage
2 ground remains the authoritative ground collider.

Current limits are intentional: roof outlines are used as provisional wall
footprints, courtyard holes are filled and warned about, facades are procedural, and
all heights and optional vegetation dimensions are assumptions. See
`STAGE4_GEOMETRY_CONTRACT.md`.

Review these external artifacts:

```text
stage4_out/report.json
stage4_out/semantic_labels.png
stage4_out/semantic_overlay.png
stage4_out/components.json
stage4_out/mask_building.png
stage4_out/mask_road.png
```

To import an existing passing proposal without inference, use **Drone World → Stage 4
→ Import Stage 4A Semantic Proposal...** and select `stage4_out`.

## 9. Correct order for every new video

Run the stages consecutively in this exact order:

```text
new video
   ↓
Stage 1 → stage1_out + Stage1Preview.unity
   ↓
Stage 2 → stage2_out + Stage2MetricWorld.unity
   ↓
Stage 3 → stage3_out + Stage3ActorPlayback.unity
   ↓
Stage 4A → stage4_out + Stage4SemanticReview.unity
   ↓ operator reviews/corrects and accepts exact label hashes
Stage 4B-C → stage4_out/geometry + Stage4GeneratedWorld.unity
```

Changing the source video or regenerating Stage 1 invalidates the old Stage 2 and Stage
3/4 provenance. Re-run all dependent stages afterward. Do not copy a `report.json`,
ground image, calibration, tracks file, or semantic image between unrelated runs;
hashes intentionally prevent mixed bundles from being imported.

The default output directories can safely be reused by their own stage. Publication is
atomic and a stage refuses to replace an unrelated non-empty directory.

## 10. Optional PowerShell commands

The Unity windows are recommended for beginners because they run Python, import only a
passing result, and create the correct scene automatically. The equivalent Python
commands are useful for diagnostics.

Run Stage 1 without importing it into Unity:

```powershell
.\.venv\Scripts\python.exe stage1.py "sample_vids\your_video.mp4" --output stage1_out
```

Run Stage 2 using a control-points file already produced by the Unity Stage 2 window:

```powershell
.\.venv\Scripts\python.exe stage2.py stage1_out stage1_out\stage2_control_points.json --output stage2_out
```

Run Stage 3 without importing it into Unity:

```powershell
.\.venv\Scripts\python.exe stage3.py stage1_out stage2_out --output stage3_out
```

After one complete Stage 3 detector pass, reapply current tracker/boundary settings
without repeating YOLO inference:

```powershell
.\.venv\Scripts\python.exe stage3.py stage1_out stage2_out --output stage3_out --reuse-detections-from stage3_out
```

The reuse command still validates all upstream hashes and decodes the frames again for
appearance and diagnostic generation. It skips only the expensive detector inference.

Run Stage 4A without importing it into Unity:

```powershell
.\.venv\Scripts\python.exe stage4.py stage1_out stage2_out --output stage4_out
```

Revalidate and publish an operator-corrected canonical label image without automatic
inference:

```powershell
.\.venv\Scripts\python.exe stage4.py stage1_out stage2_out --output stage4_out --semantic-labels path\to\corrected_labels.png
```

After Unity has written a valid acceptance record, generate the Stage 4B-C geometry
bundle without importing it:

```powershell
.\.venv\Scripts\python.exe stage4b.py stage4_out --output stage4_geometry_out --building-height 8
```

Python exit codes are:

- `0`: success and passing quality gate;
- `1`: setup, input, or processing error;
- `2`: completed processing but rejected by the stage's quality gate.

## 11. Troubleshooting

### Unity Hub says there is no project

Add this exact folder from disk:

```text
<repository>\UnityProject\DroneWorldRecreation
```

It must directly contain `Assets`, `Packages`, and `ProjectSettings`.

### Unity opens but the scene looks empty

That is expected before Stage 1 passes. Wait for compilation, then use
**Drone World → Run Stage 1 Pipeline...**. The stage creates and opens its scene.

### The Drone World menu is missing

Wait for script compilation to finish and open **Window → General → Console**. Resolve
any red C# compilation errors first. Confirm that the project was opened with Unity
`6000.5.7f1` and that the exact project folder was selected.

### Unity reports that `.venv` or Python is missing

Create `.venv` at the repository root, not inside the Unity project. Do not move
`UnityProject/DroneWorldRecreation` away from the repository.

### `ffmpeg` or `ffprobe` is not recognized

Install FFmpeg, close every PowerShell/terminal window, open a new one, and repeat the
version checks from section 2.2.

### The drone model is missing or has the wrong hash

Run:

```powershell
.\.venv\Scripts\python.exe setup_drone_model.py --verify-only
```

If verification fails, redownload the pinned file:

```powershell
.\.venv\Scripts\python.exe setup_drone_model.py --force
```

### CUDA is unavailable

The pipeline falls back to CPU. To restore GPU execution, update the NVIDIA driver and
install a PyTorch CUDA build compatible with that driver. Verify with section 3.5.

### Stage 1 rejects a clip

Open `stage1_out/report.json` and read `failure_reasons`. Typical causes are a cut,
sudden zoom, excessive translation/rotation, insufficient overlap, blur, or failed
registration. Use a shorter, steadier, more top-down continuous shot.

### Stage 2 says control points have duplicate world coordinates

Clicked pixel locations are not world measurements. Either enable approximate automatic
scale, or enter a distinct measured `X/Z` location for every landmark. Only one point
may use `(0, 0)`.

### Stage 2 passes with RMSE 0 but says the scale is approximate

This is expected in approximate mode. The generated image-corner mapping fits itself
exactly, but no real measurement was supplied. Treat positions and speeds as Unity
units, not verified metres.

### Stage 3 appears stuck

Check the pipeline log. Progress lines such as `frames 125/240` mean it is working.
Full-frame plus tiled TTA intentionally favors accuracy over speed. Stage 3 processes
every video frame, so longer footage takes proportionally longer.

### Stage 3 import says the bundle is stale

One of the upstream Stage 1/2 files changed after Stage 3 was generated. Re-run the
stages in order. Do not edit generated JSON or image artifacts manually.

### Stage 4A says PASS but the labels look wrong

That is an expected distinction: the automatic gate checks file integrity, provenance,
valid label IDs, coverage, and fragmentation; it cannot prove the semantic meaning of
each pixel. Do not click acceptance. Correct the one-channel label PNG using the IDs in
section 8, rerun Stage 4A with that file, and inspect the new overlay.

### Stage 4 model is missing

Run:

```powershell
.\.venv\Scripts\python.exe setup_stage4_model.py
```

If using a corrected label PNG, the automatic model is not needed for that run.

### Unity reports that scene setup cannot run during Play Mode

Exit Play Mode, run the menu command, save the scene if prompted, and then enter Play
Mode again.

## 12. Final acceptance checklist

Before treating a run as successful, verify all of the following:

- `stage1_out/report.json` has `"pass": true`.
- `stage1_out/ground.png` is aligned and does not visibly bake moving actors into the
  ground.
- `stage2_out/world_calibration.json` has `"pass": true`.
- `calibration_overlay.png` matches the intended ground orientation and size.
- If approximate Stage 2 was used, nobody claims the output is physically metric.
- `stage3_out/report.json` has `"pass": true`.
- `tracking_preview.mp4` detects the intended cars and people.
- `track_map.png` has plausible, continuous paths without identity teleports.
- `Stage3ActorPlayback.unity` enters Play Mode without Console errors.
- Cars and people move on the ground at the correct video timing.
- `stage4_out/report.json` has `"pass": true` and `"ready_for_geometry": false`
  until separate operator acceptance.
- `semantic_overlay.png` was checked against the original ground at full resolution.
- Incorrect automatic labels were corrected and Stage 4A was rerun.
- `semantic_acceptance.json` exists only if the exact current map was consciously
  accepted for polygonization.
- `stage4_out/geometry/report.json` is passing and its source hashes match the accepted
  Stage 4A bundle.
- `Stage4GeneratedWorld.unity` contains plausible building footprints and matching
  Stage 3 playback, without actors passing through accepted buildings.
- Every current Stage 4 height is described as assumed, never measured.

## 13. Licensing before distribution

Local setup and testing do not remove distribution obligations. The pinned VisDrone
checkpoint and the Ultralytics implementation used by this project are AGPL-3.0
licensed. Review the model, code, and eventual product distribution requirements before
shipping a Unity build or service. The Stage 4 OpenEarthMap model card declares MIT,
but OpenEarthMap source-image/label licenses vary by region; the road specialist also
uses the Ultralytics runtime. Perform a separate distribution review for all weights and
training-data obligations.

For implementation details and quality contracts, see:

- [`STAGE1_CONTRACT.md`](STAGE1_CONTRACT.md)
- [`STAGE2_CONTRACT.md`](STAGE2_CONTRACT.md)
- [`STAGE3_CONTRACT.md`](STAGE3_CONTRACT.md)
- [`STAGE3_IMPLEMENTATION_PLAN.md`](STAGE3_IMPLEMENTATION_PLAN.md)
- [`STAGE4_CONTRACT.md`](STAGE4_CONTRACT.md)
- [`STAGE4_GEOMETRY_CONTRACT.md`](STAGE4_GEOMETRY_CONTRACT.md)
- [`STAGE4_IMPLEMENTATION_PLAN.md`](STAGE4_IMPLEMENTATION_PLAN.md)
