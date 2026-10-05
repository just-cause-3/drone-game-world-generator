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
