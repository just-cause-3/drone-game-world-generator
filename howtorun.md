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

## 5. Run the video through Stages 1-4A

Use a short test video first. It should look almost straight down, remain mostly
stationary, and contain one continuous shot. Small drift and smooth bounded zoom
are supported. Avoid cuts, sudden zoom jumps, and strongly angled footage.

1. In Unity, exit **Play Mode** if it is running.
2. Open **Drone World -> Run Full Pipeline (Stages 1-4A)...**.
3. Click **Browse** beside **Input video** and select your video. For example:

   ```text
   D:\drone-game-world-generator-usin-cv\sample_vids\A_perfectly_top_down_degre.mp4
   ```

4. In **Isolated run folder**, type a new folder path:

   ```text
   D:\drone-game-world-generator-usin-cv\runs\perfect_top_down_v2_full
   ```

   **The folder does not have to exist. The pipeline creates it.** Use the text field
   when entering a new path; the folder picker is for existing folders.
5. Set **Approx. image width (m)** to `40` for a first test without measurements.
   This makes the whole ground image approximately 40 Unity units wide; it is not
   the width of one road or a detected measurement.
6. Click **Run Stages 1-4A** once.
7. Keep Unity open and wait for the final status. Do not start another pipeline or
   enter Play Mode while it is processing.

The pipeline automatically creates the ground, assigns coordinates, tracks cars and
people, and proposes the static labels. No manual ground markers are needed in this mode.

**Success:** the status begins with `PASS`, and `Stage4SemanticReview.unity` opens.
The colored overlay is a review map, not the finished 3D world.

**Failure:** read the failed stage and reason in the pipeline window. The automated
run stops; do not continue to geometry as if it passed.

### Want a faster run?

Use **Drone World -> Run Full Pipeline - Fast Mode (Stages 1-4A)...** instead.
Choose a separate folder, such as `runs\perfect_top_down_v2_fast`, and click
**Run Stages 1-4A - Fast**.

Fast mode detects actors on alternate frames and keeps full-rate playback. It can
miss very brief appearances. The normal mode detects every frame. Do not use the
same run folder for both modes.

## 6. Review and accept Stage 4A

First open your run folder in File Explorer and check:

| File inside the run folder | What to check |
| --- | --- |
| `stage1\ground.png` | Ground looks aligned; cars and people are not left painted into it. |
| `stage3\tracking_preview.mp4` | Detection boxes and candidate IDs follow the visible actors. |
| `stage4\semantic_overlay.png` | Roads, sidewalks, buildings, and vegetation are labelled sensibly. |

Then return to Unity:

1. In the **Hierarchy** panel, select **Drone World - Stage 4A Semantic Review (NO GEOMETRY)**.
2. Look at its **Inspector**. Use **Hide Semantic Overlay / Show Semantic Overlay**
   to compare the labels with the ground image.
3. If the labels are correct, click **Accept Exact Semantic Map for Stage 4B**.
4. Click **Accept** in the confirmation dialog.

Select the **object in the Hierarchy**, not the scene file in the Project panel.
Selecting a scene file only shows its scene-asset Inspector and an Open button.

If labels are wrong, correct them before accepting. Open
**Drone World -> Stage 4 -> Run Stage 4A Semantic Mapping...**, choose a corrected
label PNG in **Corrected labels (optional)**, and rerun only Stage 4A using the
matching calibration and the same run's `stage4` folder. Then review again.
The correction must be a single-channel image containing class IDs, not the colored
overlay. See [the detailed label-correction instructions](docs/HOW_TO_RUN.md#8-run-stage-4a-create-and-review-static-semantic-labels).

## 7. Generate the 3D world — Stage 4B-C

1. On the accepted review object's Inspector, click
   **Open Stage 4B-C 3D Geometry Generator**.
2. Confirm **Accepted Stage 4A map** is filled in.
3. Leave **Geometry output** at the selected run's `stage4\geometry` folder.
4. For a first run, use **Height policy: Fixed**, **Fixed building height (m): 8**,
   and leave the other settings unchanged. Leave **Vegetation placeholders** off.
5. Click **Generate and Import Provisional 3D World**.
6. Wait for `PASS` and `Stage4GeneratedWorld.unity` to open.

The value `8` is your chosen building-height assumption. If the accepted map contains
no buildings, generating zero buildings is correct; an intersection-only video may
still look mostly flat.

## 8. Play the finished scene

1. If necessary, double-click **Assets -> Scenes -> Stage4GeneratedWorld** in Unity's
   **Project** panel to open the scene.
2. Click Unity's main **Play** button at the top.
3. Open the **Game** tab to see the camera view.
4. Select **Stage 3 Real Track Playback** in the **Hierarchy**.
5. Use its Inspector's **Pause / Play**, **Reset**, and **Timeline (seconds)** controls.

You should see confirmed cars and people replaying their paths on the generated
world. Actors appear only during their supported time intervals. Stationary actors
may be hidden by the playback settings.

To view actor playback without the geometry, open
**Assets -> Scenes -> Stage3ActorPlayback** and press Play instead.

Exit Play Mode before rebuilding or importing anything. Changes made only during
Play Mode are generally not saved as scene edits.

## Run another video later

Repeat **Steps 5-8**. Select the new video and use a new run folder.
You do not need to reinstall Python, download models again, or make a new Unity project.

Separate run folders preserve the external results. Unity reuses the standard
generated scene names, so reimport the matching run when switching between videos.

## Reopen completed results without processing again

If the scenes are already imported, simply open `Stage4GeneratedWorld.unity` and press Play.

If you need to import a completed run again:

1. Exit Play Mode.
2. Open **Drone World -> Run Full Pipeline - Fast Mode (Stages 1-4A)...**.
3. Set the run-folder field to the existing completed run's **parent folder**,
   such as `runs\perfect_top_down_v2_full`, not its `stage1` or `stage4` subfolder.
4. Click **Import Existing Passing Fast Run**. Despite its name, the current import
   action accepts a passing normal or fast run. It imports files without inference.
5. Continue from Step 6. Existing acceptance remains usable only if the exact map
   and report still match. In the geometry window, **Import Existing Passing Geometry
   Bundle** lets you reuse a matching completed `stage4\geometry` bundle.

Do not click a **Run** button just to reopen results: it can repeat processing.

## Troubleshooting

| Problem | Simple next step |
| --- | --- |
| No **Drone World** menu | Wait for compilation, then check Unity's Console for red compile errors. Confirm you opened `UnityProject\DroneWorldRecreation`. |
| Missing model or Python environment | Complete Steps 2-3 in the repository folder. |
| CUDA is false / the run is very slow | Run `nvidia-smi` in PowerShell. If it cannot see the GPU, check the NVIDIA driver. If it can, check the project's PyTorch CUDA installation using the official selector linked in Step 2. A CUDA-enabled package still needs a working compatible driver. |
| A run folder is missing | Type the full new path into the run-folder text field. The pipeline creates it. |
| The log has not changed for a while | One detector pass can be slow, especially on CPU. Check Task Manager for the Python process and its CPU/GPU activity before starting another run. |
| Stage 1 failed | Read `failure_reasons` in the run's `stage1\report.json`. Check footage suitability; do not bypass the quality gate. |
| Fast mode passed Stages 1-3 but failed Stage 4 | Retry with the same video, run folder, and width. It resumes Stage 4 only if its saved inputs and artifacts still validate. |
| Import says files are stale or mismatched | Use stages from the same run. If an upstream stage changed, rebuild the stages depending on it. |
| Only one sample car/person, or no movement | Open `Stage3ActorPlayback` or `Stage4GeneratedWorld`, then press Play. Sample-model placement is not real tracking playback. Check `stage3\report.json` for confirmed tracks. |
| No 3D buildings | Finish Steps 6-7 and check whether the accepted map actually has building regions. |
| Unity reports no valid license | Open Unity Hub, sign in, and activate an eligible Editor license, then reopen the project. |

For individual-stage commands and advanced options, see [the detailed guide](docs/HOW_TO_RUN.md).
For recent fixes and remaining accuracy limits, see [the robustness audit](docs/ROBUSTNESS_AUDIT.md).
