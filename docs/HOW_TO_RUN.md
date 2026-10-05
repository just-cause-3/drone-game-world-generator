# Beginner guide: install and run the drone-video-to-Unity pipeline

This is the authoritative setup and run guide for the current staged project. It
covers the existing Stage 1, Stage 2, and Stage 3 implementation. The legacy workflow
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

Stage 4, including building height, depth, and 3D building generation, is not yet part
of this run procedure. At the end of this guide the Unity scene contains a textured
flat ground and animated sample car/person models.

## Quick run checklist

If the machine is already configured, the normal run is:

1. Open `UnityProject/DroneWorldRecreation` in Unity `6000.5.7f1`.
2. Run **Drone World → Run Stage 1 Pipeline...** and select the video.
3. Run **Drone World → Run Stage 2 Metric Calibration...** and use approximate scale
   when no measurements exist.
4. Run **Drone World → Stage 3 → Run Stage 3 Tracking Pipeline...**.
5. Inspect the three stages' review artifacts.
6. In `Stage3ActorPlayback.unity`, press Play and select the playback root for controls.

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

### 3.4 Verify GPU access

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU mode')"
```

`CUDA: True` means the detector will use the NVIDIA GPU. `CUDA: False` is not a setup
failure, but Stage 1 and especially Stage 3 will be much slower.

### 3.5 Run the automated tests

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

## 8. Correct order for every new video

Run the stages consecutively in this exact order:

```text
new video
   ↓
Stage 1 → stage1_out + Stage1Preview.unity
   ↓
Stage 2 → stage2_out + Stage2MetricWorld.unity
   ↓
Stage 3 → stage3_out + Stage3ActorPlayback.unity
```

Changing the source video or regenerating Stage 1 invalidates the old Stage 2 and Stage
3 provenance. Re-run Stage 2 and Stage 3 afterward. Do not copy a `report.json`, ground
image, calibration, or tracks file between unrelated runs; hashes intentionally prevent
mixed bundles from being imported.

The default output directories can safely be reused by their own stage. Publication is
atomic and a stage refuses to replace an unrelated non-empty directory.
