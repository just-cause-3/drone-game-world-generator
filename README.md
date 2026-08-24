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

## 4. Choosing and adding your video

### 4.1 What footage works

Best → worst, based on how the geometry behaves:

1. **High-altitude, near top-down, slow orbit/pan over flat ground** (parking lot,
   yard, intersection) — ideal.
2. **Hovering overhead shot** — ideal and even simpler.
3. **Moderate oblique (~30-50 degrees down), slow motion, flat ground filling the
   frame** — good.
4. **Fast/low FPV footage, horizon in frame, mostly sky/water, heavy motion blur** —
   expect the quality gate to reject it.

Also required: subjects large enough to detect (people should be more than a few
pixels — 1080p or better helps), and ground with visible texture (registration
matches features; bare uniform tarmac/water/snow gives it nothing to grab).

Cuts/edits are handled automatically: the pipeline detects scene cuts and keeps the
**longest continuous shot**. A burned-in HUD/watermark should be cropped via config
(section 10, `ingest.crop`).

### 4.2 Add the file

Copy your clip to:

```text
tempProject\video\drone.mp4
```

Any container ffmpeg reads is fine (`.mp4`, `.mov`, `.mkv`...). If your file has a
different name or location, open `config.yaml` and change the first line, e.g.:

```yaml
input: video/DJI_0042.MP4
```

Paths in `config.yaml` are relative to the project folder.

---

## 5. First run (fully automatic)

From the project folder, with the venv active:

```bat
python run.py
```

What happens, stage by stage (you'll see matching `[stage]` log lines):

| Stage | What it does | Rough time (70 s clip, decent GPU) |
|---|---|---|
| `ingest` | reads true fps, finds the longest continuous shot, crops HUD + 10% borders, re-encodes to constant fps, extracts every 6th frame | 1–3 min |
| `track` | YOLO11m detects + tracks people/vehicles on every frame at `imgsz 1920`; writes mover masks. **First run downloads the model (~40 MB)** | 2–10 min |
| `calib` | no clicked points yet, so it auto-estimates scale from detected car sizes (falls back to "unscaled" if too few cars) | seconds |
| `geometry` | registers every extracted frame to the reference frame (chained matching + drift re-anchoring) | 1–5 min |
| `gate` | computes quality metrics and **stops here if the clip is unusable** | seconds |
| `mosaic` | builds the ground texture: median of all frames with movers masked out | 1–3 min |
| `map` | converts pixel tracks to world meters, splits/smooths/stitches them | seconds |
| `deploy` | copies the results into `UnityProject/Assets/Generated/` | instant |

A successful run ends with a line like:

```text
[done] scale=auto_scaled. For metric scale: python -m scripts.calibrate, then re-run.
```

and exit code 0. A gate rejection ends with `FAIL: clip unsuitable` and exit code 2 —
see section 7.

---

## 6. Understanding the outputs

### 6.1 `out/report.json` — read this first

```json
{
  "frames_registered_pct": 84.2,
  "tracks": 17,
  "blur_pct": 6.1,
  "pass": true,
  "scale_status": "auto_scaled",
  "ref_index": 141
}
```

| Field | Meaning | Healthy value |
|---|---|---|
| `frames_registered_pct` | % of frames whose geometry was solved directly (the rest are interpolated) | above 60 (gate threshold), ideally 80+ |
| `tracks` | number of people/vehicles found | whatever matches reality |
| `blur_pct` | % of frames that are blurry | below 40 (gate threshold) |
| `pass` | did the gate accept the clip | `true` |
| `scale_status` | `metric` (you calibrated) / `auto_scaled` (car-size estimate) / `auto_scaled_lowconf` (estimate, don't trust it) / `unscaled` (consistent but arbitrary units) | `metric` after section 8 |
| `ref_index` | which extracted frame is the reference (used by the calibration tool) | — |

### 6.2 The three deliverables

- **`out/ground.png`** — top-down ground image in the world frame. Moving objects are
  erased (per-pixel median over time). Already vertically flipped to match Unity's
  texture orientation — do not flip it again.
- **`out/tracks_world.json`** — `{video_fps, tracks: [{id, class, pts: [{t, x, z}]}]}`.
  `t` is seconds from clip start; `x`/`z` are meters in Unity axes (X = right/east,
  Z = forward/north). `class` is `person` or `vehicle`.
- **`out/world_bounds.json`** — `{x0, z0, W_m, H_m, ppm}`: the texture covers the world
  rectangle from `(x0, z0)` to `(x0+W_m, z0+H_m)` at `ppm` pixels per meter.

All three (bounds merged into `scene_meta.json`) are also copied to
`UnityProject/Assets/Generated/`.

---

## 7. If the gate FAILs

The pipeline refuses to emit a broken world rather than producing garbage silently.
Match the failing metric in `out/report.json`:

| Failing metric | Cause | Fix, in order of preference |
|---|---|---|
| `frames_registered_pct` low | featureless ground, fast motion, too little overlap between frames | 1) lower `ingest.stride` to 3 or 4 (more overlap); 2) lower `geometry.reanchor_every` to 10; 3) pick a steadier / more textured clip |
| `tracks` = 0 | subjects too small/blurry, or wrong classes | raise `detect.imgsz` (2560), lower `detect.conf` (0.15); if people are single-digit pixels, no detector saves the clip — use lower-altitude footage |
| `blur_pct` high | motion blur / low light | slower clip, or (if the metric seems oversensitive on your footage) raise `gates.blur_threshold` |

After changing anything under `ingest:` or `detect:`, rerun with `--force`
(see section 9). Changes elsewhere just need a plain rerun.

---
