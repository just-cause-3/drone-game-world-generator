# Drone Video → Unity Recreation Pipeline

Turn a **pre-recorded drone video** into a schematic Unity world: a flat ground plane
textured with a clean top-down image stitched from your footage, plus every detected
person and vehicle replayed as a moving marker on their real-world path.

**You give it:** one video file.
**It gives you:** `ground.png` (top-down ground texture, moving objects erased),
`tracks_world.json` (every person/vehicle's path in meters, Unity axes), and
`report.json` (a quality report that refuses to deploy garbage).

The pipeline runs on **Windows with an NVIDIA GPU (CUDA)**. It also works on CPU or
Apple Silicon, just slower.

---

## Table of contents

1. [What you need before starting](#1-what-you-need-before-starting)
2. [Folder layout](#2-folder-layout)
3. [One-time setup on Windows](#3-one-time-setup-on-windows)
4. [Choosing and adding your video](#4-choosing-and-adding-your-video)
5. [First run (fully automatic)](#5-first-run-fully-automatic)
6. [Understanding the outputs](#6-understanding-the-outputs)
7. [If the gate FAILs](#7-if-the-gate-fails)
8. [Metric calibration (optional but recommended)](#8-metric-calibration-optional-but-recommended)
9. [Re-running: what is cached and when to use --force](#9-re-running-what-is-cached-and-when-to-use---force)
10. [Every config option explained](#10-every-config-option-explained)
11. [Phase 6 - build the Unity scene (ground and agents)](#11-phase-6---build-the-unity-scene-ground-and-agents)
12. [Phase 7 - zones and the detections CSV log](#12-phase-7---zones-and-the-detections-csv-log)
13. [Phase 8 - optional one-command headless build](#13-phase-8---optional-one-command-headless-build)
14. [Phase 9 - validate the result](#14-phase-9---validate-the-result)
15. [Tuning guide (symptom → knob)](#15-tuning-guide-symptom--knob)
16. [Troubleshooting](#16-troubleshooting)
17. [How the pipeline works internally](#17-how-the-pipeline-works-internally)

---

## 1. What you need before starting

- A **Windows PC with an NVIDIA GPU** and a reasonably recent driver
  (check: open a terminal, run `nvidia-smi` — if it prints a table, you're fine).
- **~6 GB free disk**: Python packages (~4 GB, mostly PyTorch), the YOLO model
  (~40 MB, auto-downloaded on first run), plus room for extracted frames.
- **Internet access** for the one-time setup and the first run (model download).
- A **drone video file** (see section 4 for what makes a good one).
- This project folder copied onto the Windows machine (any location, e.g.
  `C:\work\tempProject`). No git needed — it's plain files.

---

## 2. Folder layout

What the folder contains **before** any run:

```text
tempProject/
  run.py                  <- the one command you run
  config.yaml             <- all settings (section 10)
  requirements.txt        <- Python dependencies
  README.md               <- this file
  video/                  <- PUT YOUR VIDEO HERE (video/drone.mp4 by default)
  scripts/
    common.py             <- shared helpers (paths, ffprobe, the frame-index rule)
    ingest.py             <- Phase 1: clean the clip, extract frames
    track_agents.py       <- Phase 2: YOLO detection + tracking + mover masks
    calibrate.py          <- Phase 3a: scale/orientation (auto + click tool)
    geometry.py           <- Phase 3b: per-frame registration (the geometric core)
    build_mosaic.py       <- Phase 4: the ground texture
    map_to_world.py       <- Phase 5: pixel tracks -> world-meter tracks
    selfcheck.py          <- 2-second math sanity check, no video needed
  unity_scripts/          <- ready-made Unity C# scripts (Phases 6-8); copy into Unity (section 11)
    AgentPlayback.cs      <- replays tracks_world.json as moving prefabs
    ZoneVolume.cs         <- trigger-box zone definition
    DetectionLogger.cs    <- writes detections.csv with video-time stamps
    AgentZoneReporter.cs  <- per-agent zone membership -> logger
    Editor/
      BuildRunner.cs      <- headless scene/player builder (Phase 8)
```

What gets **created** by running it:

```text
  out/
    shot_clean.mp4        <- your clip: trimmed to one shot, cropped, constant fps
    ingest_meta.json      <- fps + stride marker (guards the cache)
    tracks_image.json     <- raw pixel-space tracks from YOLO
    calibration.json      <- only if you do metric calibration (section 8)
    H_ref_to_world.npy    <- pixel -> meters mapping for the reference frame
    H_per_frame.npy       <- pixel -> meters mapping for EVERY frame
    inliers.npy           <- per-frame registration quality
    report.json           <- the quality gate verdict (section 6)
    ground.png            <- the ground texture
    world_bounds.json     <- where the world rectangle sits, in meters
    tracks_world.json     <- the final agent paths
  frames/                 <- extracted registration frames (rebuilt each ingest)
  masks/                  <- moving-object masks (rebuilt each tracking run)
  UnityProject/
    Assets/Generated/     <- ground.png + tracks_world.json + scene_meta.json, ready for Unity
```

---

## 3. One-time setup on Windows

Do these in order. Each step ends with a check so you know it worked.

### 3.1 Install Python 3.12

1. Download **Python 3.12 (64-bit)** from <https://www.python.org/downloads/>.
   Avoid 3.13/3.14 — PyTorch wheels lag behind the newest Python.
2. In the installer, tick **"Add python.exe to PATH"**, then Install.
3. **Check:** open a new terminal (Windows key → type `cmd` → Enter) and run:

   ```bat
   py -3.12 --version
   ```

   Expected output: `Python 3.12.x`

### 3.2 Install ffmpeg

1. In the same terminal:

   ```bat
   winget install Gyan.FFmpeg
   ```

   (If winget is unavailable: download the "release full" build from
   <https://www.gyan.dev/ffmpeg/builds/>, unzip it, and add its `bin` folder to your
   PATH via *System Properties → Environment Variables*.)
2. **Close the terminal and open a new one** — PATH changes only apply to new terminals.
3. **Check:**

   ```bat
   ffmpeg -version
   ffprobe -version
   ```

   Both must print version info. If either says "not recognized", the PATH step failed.

### 3.3 Create the virtual environment

All commands from here on are run **inside the project folder**:

```bat
cd C:\work\tempProject
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
```

After activation your prompt starts with `(.venv)`.

- **PowerShell users:** use `.venv\Scripts\Activate.ps1` instead. If PowerShell refuses
  ("running scripts is disabled"), run once:
  `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, answer `Y`, then retry.
- **Every future session**: re-run the `activate` line before using the pipeline.

### 3.4 Install PyTorch with CUDA (order matters — do this BEFORE requirements.txt)

If you install `requirements.txt` first, pip pulls the **CPU-only** torch and tracking
will be ~20x slower. So:

```bat
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

`cu126` works with any reasonably recent NVIDIA driver. If it errors, open
<https://pytorch.org/get-started/locally/>, pick *Stable / Windows / Pip / CUDA*, and
copy the exact command it shows.

**Check:**

```bat
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected: something like `2.x.x+cu126 True NVIDIA GeForce RTX ...`
If it prints `False`, see [Troubleshooting](#16-troubleshooting).

### 3.5 Install the remaining dependencies

```bat
pip install -r requirements.txt
```

### 3.6 Verify the pipeline math (no video needed)

```bat
python -m scripts.selfcheck
```

Expected output: `selfcheck OK`. This exercises the coordinate/interpolation/stitching
math with synthetic data — if it passes, the environment and the core logic are sound.

Setup is done. You never repeat section 3 (except `activate` each new terminal).

---
