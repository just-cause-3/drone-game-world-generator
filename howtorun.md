# How to run the project

This guide takes you from a video file to a Unity scene with car/person playback
and provisional 3D buildings where buildings are present.

**Already installed everything? Start at [Step 4](#4-open-the-unity-project).**

The complete order is:

**Install once -> open Unity -> run Stages 1-4A -> review labels -> generate Stage 4B-C -> press Play.**

The full-pipeline button stops for your review before generating buildings.
Building heights and the default world scale are assumptions, not measurements.

## 1. Install the required software — once

Use a Windows PC with:

- **Python 3.12, 64-bit.** Download it from [Python](https://www.python.org/downloads/).
  Enable **Add Python to PATH** during installation.
- **Unity Hub and Unity Editor `6000.5.7f1`.** This is the version recorded by this
  project. Sign in to Hub and activate your Editor license before continuing.
- **FFmpeg**, for reading and preparing videos.
- An NVIDIA GPU is strongly recommended. CPU processing works but can take much longer.

To install FFmpeg, open **PowerShell** and run:

```powershell
winget install Gyan.FFmpeg
```

Close PowerShell, open it again, and check:

```powershell
py -3.12 --version
ffmpeg -version
ffprobe -version
```

All three commands should print version information. If a command is not found,
finish that installation before continuing.

## 2. Prepare Python — once

Run these commands in **PowerShell**, not the Unity Console. Run one line at a time
and wait for it to finish. If it reports an error, resolve that error before the next line.

Open the repository folder:

```powershell
Set-Location "D:\drone-game-world-generator-usin-cv"
```

If your repository is elsewhere, use its actual location throughout this guide.

For a new installation, create the Python environment and install the packages:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

You do not need to activate `.venv`. Unity uses this exact Python environment.
Keep an existing working environment; you do not need to reinstall it for each run.

The CUDA command above follows this repository's setup. If it is incompatible with
your driver, use the [official PyTorch installation selector](https://pytorch.org/get-started/locally/)
for Windows, Pip, Python, and a supported CUDA build. Run the resulting installation
through `.\.venv\Scripts\python.exe -m pip` so it installs into this project.

Check GPU access:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU mode')"
```

**Expected for your RTX 3050: `CUDA: True` and the GPU name.** Having an NVIDIA card
alone does not guarantee that Python can use it. If this prints `CPU mode`, processing
will use the CPU. See [Troubleshooting](#troubleshooting) before starting a long run.

## 3. Download the models — once

In the same PowerShell window, run:

```powershell
.\.venv\Scripts\python.exe setup_drone_model.py
.\.venv\Scripts\python.exe setup_inpainting_model.py
.\.venv\Scripts\python.exe setup_stage4_model.py
```

Wait for each download and verification to finish. Internet access is needed for
the initial downloads. These commands reuse existing verified model files.

Check the installation:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

The final result should be **`OK`**. Some tests intentionally print failed quality
gates while checking rejection behavior; the final test summary is what matters.

## 4. Open the Unity project

1. Open **Unity Hub -> Projects**.
2. Choose **Add / Add project from disk**.
3. Select this folder:

   ```text
   D:\drone-game-world-generator-usin-cv\UnityProject\DroneWorldRecreation
   ```

4. Open it with Unity `6000.5.7f1`.
5. Wait for importing and compilation to finish.
6. Confirm that **Drone World** appears in Unity's top menu.

Select the folder containing `Assets`, `Packages`, and `ProjectSettings`.
Do not select the repository root or create another blank project. Keep the Unity
project in its current repository location so it can find Python and the models.

An empty scene or a **STAGE 1 WAITING** platform is normal before processing.
