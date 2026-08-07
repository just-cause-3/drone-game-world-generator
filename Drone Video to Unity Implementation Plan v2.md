# Schematic Drone-Video → Unity Recreation — Implementation Plan **v3 (Pre-Recorded Footage, Hardened Edition)**

> **Goal:** take a **pre-recorded** drone video and produce a simple Unity recreation of the real place — a flat ground textured from the footage, with the people and vehicles re-created as moving stand-in shapes, plus zone labels and a log. **Not detailed. As light as possible, but robust to real footage.**

**What changed from v1 → v2 (read this):** v1 assumed you might shoot your own footage and that the camera could hover (one fixed mapping for the whole clip). You're using **pre-recorded shots**, and pre-recorded drone footage is *almost always filmed with a moving camera* (orbiting, flying forward, panning, changing altitude). That single fact reshapes the whole conversion: instead of one image→ground mapping, you generally need **per-frame registration**. v2 therefore:

1. Adds a deep section on **which pre-recorded clips work** and how to triage/prep a clip you already have (Part 1).
2. Adds a large, in-depth **"How the video becomes a Unity world"** section explaining the conversion conceptually + mathematically (Part 2) — this is the centerpiece you asked for.
3. Makes the **moving-camera pipeline** (per-frame homography + mosaic) the default path, with full code (Phase 3B / Phase 5).
4. Expands every phase, plus a worked example, an accuracy/validation phase, a coordinate cheat-sheet, and bigger troubleshooting.

Read **Part 0**, **Part 1**, and **Part 2** before doing anything — they decide everything else.

**What changed from v2 → v3 (hardening pass):** v3 fixes the implementation-level gaps found in review (see the companion *"Issues and Fixes"* doc for the full reasoning). The load-bearing changes:

1. **One clock, one frame index (Part 2.8).** Every artifact is keyed by the video frame number `n`; registration frames are an integer-stride subset (`select=mod(n,STRIDE)`, not `fps=4`), masks are named by `n`, and no stage hardcodes a frame rate. This makes the frame↔mask and detection↔homography pairings exact *by construction*.
2. **Homographies are interpolated over time** when mapping full-fps detections through stride-sampled registration (Phase 5) — nearest-neighbor lookup produced stair-step tracks.
3. **Chained registration with re-anchoring** replaces direct-to-reference matching (Phase 3B). Failed frames are marked and interpolated — never given a stale copy of the previous `H`. Per-frame inlier counts are saved for the gate.
4. **Two distinct mirror bugs, two distinct fixes** (Phase 4, Phase 9, Appendix B): the OpenCV-vs-Unity texture flip is baked in once, and a probe scene verifies it separately from the calibration-axis flip.
5. **Manual calibration is bound to its exact reference frame** by filename + hash (Phase 3B, 8.7) — `ref_frame: auto` can no longer silently invalidate clicked points.
6. Plus: lens-distortion handling (Phase 1.3b), `imgsz`/GMC for small objects and moving cameras (Phase 2), foot-point shift for oblique views, masked-median mosaic with **auto-computed world bounds** (Phase 4B), gap-aware smoothing + an actually-implemented track stitcher (Phase 5), and a time-correct zone logger (Phase 7). Assumed hardware: an NVIDIA CUDA GPU for detection.

---

## Table of Contents

- **Part 0 — The mental model (what you're building)**
- **Part 1 — Choosing your video: what type is preferred (in depth)**
- **Part 2 — HOW THE VIDEO BECOMES A UNITY WORLD (the conversion, in depth)**
- **Part 3 — Tools & project layout**
- **Phase 0 — Validate the Unity side first (fake data)**
- **Phase 1 — Ingest & pre-process the pre-recorded video**
- **Phase 2 — Detect & track people/vehicles (+ masks)**
- **Phase 3 — Establish the geometry (the heart of the conversion)**
- **Phase 4 — Build the static ground (rectified frame or mosaic)**
- **Phase 5 — Map image tracks → world tracks (+ smoothing & gaps)**
- **Phase 6 — Spawn & animate the agents in Unity**
- **Phase 7 — Zone System + logging (the surveillance layer)**
- **Phase 8 — Automate the pipeline (one command)**
- **Phase 9 — Validate & measure accuracy**
- **Phase 10 — Optional upgrades**
- **Worked example (end-to-end)**
- **Appendix A — Troubleshooting (expanded)**
- **Appendix B — Coordinate-system cheat sheet**
- **Appendix C — Glossary**
- **Appendix D — Tools & references**

---

## Part 0 — The mental model (what you're building)

### 0.1 The idea in one picture

```
                         ┌──────────────────────────────────────────┐
                         │  STATIC LAYER (built once)                 │
  pre-recorded ────────▶ │  • a flat ground plane in Unity            │
  drone video            │  • textured with a top-down image          │
                         │    (a rectified frame, or a stitched        │
                         │     mosaic) derived from the footage        │
                         └──────────────────────────────────────────┘
                                            +
                         ┌──────────────────────────────────────────┐
  pre-recorded ────────▶ │  DYNAMIC LAYER                             │
  drone video            │  • detect + track people/vehicles          │
                         │  • map each one's ground position per       │
                         │    frame into the world frame               │
                         │  • spawn capsule (person) / box (vehicle)    │
                         │    that moves along its real trajectory      │
                         │  • zone labels + CSV log                     │
                         └──────────────────────────────────────────┘
```

You rebuild the **ground as a flat map**, and the **moving things as simple shapes that move the way they did in the video.** Nothing more.

### 0.2 The two-layer principle (why it's split this way)

Reconstructing a static scene and tracking moving things are *different problems that interfere with each other*. So the world is two independent layers:

- **Static layer:** the ground/buildings, which don't move. Built once.
- **Dynamic layer:** people/vehicles, which move. Rebuilt as animated markers.

Keeping them separate is what makes a moving-camera, surveillance-style video tractable.

### 0.3 What this approach does well / does not do

**Well:** fast; no heavy GPU reconstruction; robust to footage that 3D reconstruction chokes on; gives positions, zones, and a who-was-where-when log.

**Not:** no real 3D terrain/buildings (ground is flat; structures are optional plain boxes); everything is positioned *on the ground plane*; positions are approximate, not survey-grade.

### 0.4 The pre-recorded constraint

Because you can't re-shoot:
- You **select and triage** clips instead of capturing ideal ones (Part 1).
- You must **handle whatever camera motion** the clip has (Part 2 / Phase 3).
- You usually have **no camera calibration, no GPS/telemetry, possibly edits and overlays** — all handled in Phase 1.

---

## Part 1 — Choosing your video: what type is preferred (in depth)

You asked specifically what video is preferred. The short answer and the reasoning:

### 1.1 The deciding factor is **camera motion**, then **angle**, then **content**

A homography (image→flat-ground mapping) is the engine of the schematic method. It is exact only for a **flat ground**, and a *single* homography is valid only for a **non-moving camera**. So clips sort into three **motion regimes**, each needing a different amount of work (full details in Part 2):

| Regime | Camera motion | What it needs | Difficulty |
|---|---|---|---|
| **A — Static** | Locked-off hover, no movement | **One** homography for the whole clip | Easiest (rare in pre-recorded) |
| **B — Moving, planar** | Orbit / pan / fly-over of a roughly **flat** area | **Per-frame** homography + a stitched ground | **Default for pre-recorded** |
| **C — Moving, 3D** | Big parallax around tall structures / strong terrain | Camera **pose (SfM) + depth** | Advanced upgrade |

> **Expect Regime B.** Most pre-recorded drone "shots" move. Plan for per-frame registration; treat A as a lucky simplification and C as an optional upgrade.

### 1.2 Ranked preference for **pre-recorded** clips

**Preferred (pick these):**
1. **High-altitude, near top-down (nadir), slow steady orbit or slow lateral pan over a flat area** (parking lot, yard, field, base, intersection, rooftop). The ground dominates the frame, motion is smooth, the flat-ground assumption holds → Regime B works beautifully, and a clean mosaic is easy.
2. **Locked-off / hovering overhead shot** → Regime A, the simplest possible case. Uncommon but ideal when you find one.
3. **Moderate oblique (≈30–50° down), slow pan, flat-ish ground filling most of the frame** → Regime B with mild foreshortening; acceptable.

**Acceptable with extra work:**
4. **Steady forward fly-over at altitude, near-nadir** → mosaic/registration works, but long clips accumulate drift; process in segments and re-anchor (Part 2.3-B).
5. **Slow orbit around a building/object** → this contains real 3D (parallax). Ground-level objects still map okay on the flat plane, but anything off the ground is misplaced → consider Regime C.

**Poor fits (avoid or expect failure):**
6. **Low-altitude FPV / fast cinematic / racing-drone footage** → fast motion, motion blur, large perspective change frame-to-frame → registration and detection both suffer.
7. **Low-oblique / horizon-in-frame shots** → the flat-ground assumption breaks badly for distant ground; far objects compress to nonsense.
8. **Heavily edited clips** (hard cuts, transitions, speed ramps, picture-in-picture, burned-in text/HUD) → cuts break tracking and registration; overlays confuse features and detection. Must be split/cleaned first (Phase 1).

### 1.3 The content rules (independent of motion)

- **Ground should fill most of the frame and be roughly flat.** The method positions everything *on a plane*.
- **Resolution and clarity good enough for YOLO** to detect people/vehicles. 1080p+ is comfortable; tiny, blurry, or heavily compressed subjects detect poorly.
- **Avoid scenes that are mostly sky, water, or featureless tarmac** — registration (feature matching) needs texture on the ground.

### 1.4 Pre-recorded-specific gotchas (and fixes — see Phase 1)

- **Editing cuts** → split into continuous single-shot segments first (PySceneDetect / manual trim).
- **Variable frame rate (VFR) or re-encoded files** → normalize to constant fps; read the true fps with `ffprobe`.
- **Burned-in overlays / watermarks / HUD / logos** → crop or mask them so they don't pollute feature matching, the mosaic, or detection.
- **No GPS/telemetry, no camera calibration** → you won't get metric scale automatically. You calibrate scale manually (Part 2.6): from known object sizes, or by measuring the same location on a satellite map if it's identifiable.
- **Unknown intrinsics** → full calibration is not needed for the homography path, **but lens distortion is not optional**: a homography is exact only for a *pinhole* camera, and typical wide-FOV drone/action cams have real barrel distortion — RANSAC will keep center matches and quietly reject the frame edges, so registration "succeeds" while edge positions and mosaic borders are bent. Handle it in Phase 1.3b (border crop, or a one-parameter undistortion). Full intrinsics are only needed for the Regime-C pose+depth upgrade (estimated via SfM).

### 1.5 Triage checklist for a clip you already have

Run through this before committing to a clip:

1. Is it **one continuous shot**? If not → split (Phase 1).
2. Does the **ground fill the frame and look roughly flat**? If not (lots of sky/buildings/horizon) → poor fit or Regime C.
3. Is the camera **static, slow-moving, or fast/erratic**? Static→A, slow→B, fast→risky.
4. Are **people/vehicles clearly visible** (not 3-pixel specks)? If too small → detection will fail; pick a lower-altitude or higher-res clip.
5. Any **overlays/HUD/text**? → crop/mask.
6. Do you have **any way to get real scale** (known object, identifiable location on a map)? If not, you'll have an *unscaled* but still usable world.

If a clip passes 1–4, it's a good candidate. 5–6 are fixable.

---

## Part 2 — HOW THE VIDEO BECOMES A UNITY WORLD (the conversion, in depth)

This is the core of your request. We'll build the concept from the ground up: the problem, the coordinate frames, the three geometric regimes (with math + intuition), the unifying data flow, why moving objects are separated, and how scale/orientation land in Unity.

### 2.1 The core problem stated precisely

A video is a sequence of 2D images `I_0, I_1, … I_N`, each captured by the drone camera at a *different position and orientation* (because it's moving). A point on the real ground appears at **different pixel locations in different frames.** To place anything into a single Unity world, you need a **fixed world coordinate frame** and, for **every frame**, a rule that converts "pixel in this frame" → "position in the world frame."

Formally, for each frame `t` you want a function:

```
   world_position  =  G_t( pixel )
```

- If the camera is **static**, `G_t` is the **same** for all `t` → one mapping.
- If the camera **moves over flat ground**, `G_t` is a **per-frame homography** (a 3×3 matrix that changes each frame).
- If the camera **moves through real 3D**, `G_t` requires the camera's **pose** plus **per-pixel depth**.

Everything else (detection, tracking, ground texture) feeds into or consumes this `G_t`.

### 2.2 The coordinate systems you must keep straight

Four frames are in play. Mixing them up is the #1 source of bugs.

1. **Image / pixel frame** — 2D, origin top-left, `(u, v)` in pixels, `u` right, `v` **down**. This is where detections live.
2. **World / ground frame** — 2D metric ground plane, `(X, Z)` in **meters**. This is the fixed reference everything maps into.
3. **Unity world frame** — 3D, **left-handed**, `+X` right, `+Y` **up**, `+Z` forward, 1 unit = 1 meter. Agents go at `(X, groundY, Z)` and the ground plane lies in the `X–Z` plane at `y = groundY`.
4. **(Regime C only) Camera 3D frame** — 3D camera-centered coordinates, used when unprojecting with depth.

**Key alignment rule:** define your world/ground `(X, Z)` *directly in Unity's convention* (e.g., `+X` = east/right, `+Z` = north/away, looking down the `-Y` axis). Then the homography output is already in Unity coordinates and no axis-swapping is needed. If the result looks **mirrored** in Unity, flip the sign of one axis in your calibration points (Appendix B).

### 2.3 The three geometric regimes (math + intuition + when)

#### Regime A — Static camera → a single homography

**Intuition:** if the camera never moves, the relationship between the image and the flat ground is fixed. A **homography** `H` (3×3) relates a ground point to its pixel via projective geometry:

```
   [u]           [X]
   [v]  ~   H_w  [Z]            (homogeneous; "~" means up to scale)
   [1]           [1]
```

Invert it to go pixel→ground. In practice you compute the pixel→ground homography `H` directly from **4+ correspondences** (4 ground points whose pixel `(u,v)` *and* metric `(X,Z)` you know):

```python
H, _ = cv2.findHomography(image_pts, world_pts)         # maps (u,v) -> (X,Z)
XZ   = cv2.perspectiveTransform([[ (u,v) ]], H)          # apply to any pixel
```

**When:** locked-off hover only. **Cost:** trivial. **Limitation:** flat-ground assumption (objects must be on the ground plane).

#### Regime B — Moving camera + flat ground → per-frame homography (the workhorse)

**Intuition:** the camera moves, so the image↔ground relationship is **different in every frame**. But because the ground is (approximately) **flat**, the relationship between **any two frames** of that ground is *also* a homography (this is the key theorem: images of a plane are related by homographies). So we:

1. **Calibrate once:** pick a **reference frame** `R` and compute the reference→world homography `H_{R→W}` from 4 known ground points (as in Regime A, but only for the reference frame).
2. **Register every frame to the reference:** for each frame `t`, compute the frame→reference homography `H_{t→R}` by matching image features.
3. **Compose:** the full per-frame mapping is the matrix product

```
   H_{t→W}  =  H_{R→W} · H_{t→R}
```

   So any pixel in frame `t` maps to world meters via `XZ = perspectiveTransform(pixel, H_{t→W})`.

**How `H_{t→R}` is computed (feature registration):**

```python
# ORB features + RANSAC homography from frame t to reference R
orb = cv2.ORB_create(4000)
kpR, desR = orb.detectAndCompute(ref_gray, refMask)      # refMask hides movers/overlays
kpT, desT = orb.detectAndCompute(frame_gray, frameMask)
matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desT, desR, k=2)
good = [m for m, n in matches if m.distance < 0.75 * n.distance]   # Lowe's ratio test
src = np.float32([kpT[m.queryIdx].pt for m in good]).reshape(-1,1,2)
dst = np.float32([kpR[m.trainIdx].pt for m in good]).reshape(-1,1,2)
H_t_to_R, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)   # robust to outliers
```

**Drift & long clips:** matching a far-away frame *directly* to a single reference fails when the views barely overlap. Two robust remedies:
- **Sequential chaining with re-anchoring:** register frame `t` to frame `t-1` (always high overlap), and keep a running product to the reference; periodically re-anchor to a real reference/mosaic to limit accumulated error.
- **Incremental mosaic registration:** register each frame to the *growing mosaic* (below), which always contains the overlapping region. More robust for big sweeps.

> **v3:** Phase 3B's code implements **chaining + periodic re-anchoring as the default** — direct-to-reference matching alone fails as soon as an orbit or sweep changes the view too much, and a frame that fails to register must be *marked and interpolated*, never given a stale copy of the previous homography (that silently freezes the geometry while the camera keeps moving, then snaps).

**The mosaic (your ground texture, and a clean static plate):** warp every (masked) frame into the **world frame** with `H_{t→W}` and blend them onto one big canvas. Taking the **per-pixel temporal median** across frames makes **moving objects disappear** (they're only in a few frames; the median is the static background) — giving a clean ground map. (Part 2.5 explains why this matters.)

**When:** the common pre-recorded case — orbits, pans, fly-overs of flat-ish areas. **Cost:** moderate (feature matching per frame). **Limitation:** still flat-ground; off-ground objects get misplaced; fast motion/blur degrades matching.

#### Regime C — Moving camera + real 3D → camera pose + depth (upgrade)

**Intuition:** when the scene has real height variation/parallax (orbit around a building, hilly terrain), a flat homography can't be correct for everything. You instead recover the **camera's 3D pose** for each frame and a **per-pixel depth**, then back-project pixels into 3D world space.

1. **Poses + intrinsics** from Structure-from-Motion (COLMAP) or a feed-forward model (VGGT). You get camera intrinsics `K` and, per frame, rotation `R_t` and translation `t_t` (note conventions below).
2. **Per-pixel depth** `d(u,v)` from a monocular depth model (Depth Anything V2, or Video Depth Anything for temporal consistency / metric scale).
3. **Back-project** a detection foot pixel:

```
   X_cam   = d(u,v) · K⁻¹ · [u, v, 1]ᵀ          # camera-space 3D point
   X_world = R_tᵀ · (X_cam − t_t)               # if convention is  X_cam = R_t·X_world + t_t  (COLMAP-style)
```

**Conventions matter:** COLMAP stores **world-to-camera** `[R|t]` (i.e., `X_cam = R·X_world + t`), so world = `Rᵀ(X_cam − t)`. Always verify on a known point; sign/handedness mistakes here are common.

**Scale:** monocular SfM scale is arbitrary; fix it with a known distance or a **metric** depth model (Depth Pro / metric Video Depth Anything).

**When:** off-ground structures you care about, or terrain with relief. **Cost:** high (SfM + depth + a GPU). **Limitation:** more moving parts; still imperfect on thin/reflective surfaces.

### 2.4 The unifying data-flow diagram

```
                    pre-recorded drone video (one continuous shot)
                                     │
                         ┌───────────┴────────────┐
                         │  Phase 1: ffmpeg →      │
                         │  frames + true fps      │
                         └───────────┬────────────┘
              ┌────────────────────── │ ──────────────────────┐
              ▼                       ▼                        ▼
   ┌────────────────────┐  ┌──────────────────────┐  ┌─────────────────────┐
   │ Phase 2: DETECT+    │  │ Phase 3: GEOMETRY     │  │ Phase 4: STATIC      │
   │ TRACK (YOLO)        │  │  pick regime A/B/C    │  │ GROUND               │
   │ → image-space       │  │  → G_t (per-frame      │  │ → rectified frame    │
   │   tracks {id,cls,    │  │     mapping) + masks   │  │   or median mosaic   │
   │   foot (u,v), t}     │  │                        │  │   (uses G_t)         │
   └─────────┬──────────┘  └───────────┬───────────┘  └──────────┬──────────┘
             │                         │                         │
             └──────────┬──────────────┘                         │
                        ▼                                         │
            ┌──────────────────────────┐                         │
            │ Phase 5: MAP image tracks │                         │
            │ → WORLD tracks via G_t,    │                         │
            │ smooth, stitch gaps        │                         │
            │ → tracks_world.json        │                         │
            └─────────────┬─────────────┘                         │
                          │                                       │
                          ▼                                       ▼
            ┌─────────────────────────────────────────────────────────────┐
            │ Phase 6–7: UNITY                                              │
            │  ground plane (texture from Phase 4)                          │
            │  + agents animated along tracks_world.json (foot → position)  │
            │  + zone volumes + CSV log                                     │
            └─────────────────────────────────────────────────────────────┘
```

Three independent producers — **detection/tracking**, **geometry**, **ground texture** — meet at the world frame and assemble in Unity. The **geometry** (`G_t`) is the linchpin both for the ground texture and for placing agents, which is why Part 2 and Phase 3 dwell on it.

### 2.5 Why moving objects are handled separately

If you stitched raw frames into the mosaic, moving people/cars would smear into **ghosts** (they're in different places each frame). Two reasons to separate them:

1. **For the static ground:** mask movers out (YOLO masks, Phase 2) and/or take the **temporal median** across registered frames — moving things vanish, leaving a clean background plate.
2. **For the dynamic layer:** the same detections that you masked are *exactly* the things you re-inject as moving agents (Phase 5–6).

So one detection pass serves both: it cleans the ground **and** drives the agents. This is the elegant core of the two-layer design.

### 2.6 Scale & orientation: getting meters and aligning to Unity

- **Scale (meters):** pre-recorded clips usually have no telemetry, so set scale from one known real distance:
  - A known object: car ≈ 4.5 m long, parking space ≈ 2.5 m wide, road lane ≈ 3.5 m, soccer goal ≈ 7.32 m.
  - If the location is identifiable, measure a distance on **Google/Bing Maps** and use those coordinates for your 4 calibration points.
  - If you truly can't, pick arbitrary but consistent meters; the world will be self-consistent, just not real-scale.
- **Orientation:** assign your 4 calibration world points in **Unity's axes** (`+X` right/east, `+Z` forward/north). Verify with a known landmark; if mirrored, flip one axis sign (Appendix B).
- **Origin:** put one corner of your area at world `(0,0)` so the Unity plane, the mosaic texture, and the agent coordinates all share an origin (prevents misalignment and large-coordinate jitter).

### 2.7 How each regime degrades (so you can predict results)

- **Regime A on a slightly moving camera** → everything slowly slides; positions drift. Fix: switch to B.
- **Regime B on non-flat ground / off-ground objects** → things *on* the ground are fine; things *above* it (rooftops, drone-height objects) are pushed outward from the camera. Acceptable for "ground agents only."
- **Regime B on fast/blurred motion** → feature matching fails on some frames → gaps/jumps. Fix: lower fps, segment the clip, or upgrade to C.
- **Regime C with wrong pose convention/scale** → mirrored or wrongly-scaled world. Fix: verify on known points; use metric depth.

### 2.8 One clock, one frame index (the convention that prevents silent misalignment)

The pipeline has exactly one timeline: the constant-fps video produced in Phase 1.3. Every artifact is keyed by the **video frame number `n`** (0-based, at `VIDEO_FPS`):

- **Registration frames** are the subset `n % STRIDE == 0` (Phase 1.4). Extracted file `i` (0-based) *is* video frame `n = i · STRIDE` — an identity by construction, not a lookup.
- **Masks** are written for every video frame and named `masks/frame_{n:06d}.png`. The mask for extracted frame `i` is fetched at `n = i · STRIDE` — **never** by zipping two sorted directory listings (they'd be on different timelines).
- **Timestamps** are always `t = n / VIDEO_FPS`.
- **Homographies** exist per extracted frame; a detection at video frame `n` uses the value **interpolated** at `n / STRIDE` (Phase 5) — the camera moves between registration samples.
- `VIDEO_FPS` is **measured** (`ffprobe`), `STRIDE` lives **only in the config**. No script declares its own frame rate.

Everything in Phases 1–5 follows this convention. Break it and frames get paired with the wrong masks and detections with the wrong homographies — silently, with no error raised anywhere.

---

## Part 3 — Tools & project layout

### 3.1 Tools (lightweight)

- **Unity Hub + Unity 6 LTS** (or 2022.3 LTS), **URP** template.
- **Python 3.10+** venv:
  ```bash
  python -m venv .venv
  # Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
  pip install --upgrade pip
  pip install ultralytics opencv-contrib-python numpy scenedetect
  # opencv-contrib gives SIFT/extra features; scenedetect splits edited clips
  # NVIDIA GPU: install CUDA torch from pytorch.org for faster YOLO
  ```
- **ffmpeg / ffprobe** — ingest, fps, cropping, splitting.
- **Optional (Regime C only):** COLMAP, and a depth model (Depth Anything V2 / Video Depth Anything). A GPU helps a lot here.

### 3.2 Suggested project layout

```
project/
  video/           drone.mp4
  frames/          frame_000001.jpg ...      (every STRIDE-th video frame — Part 2.8)
  masks/           frame_000000.png ...      (mover/overlay masks, named by VIDEO frame number)
  out/
    tracks_image.json
    calibration.json          (ref-frame binding: filename + sha1 + clicked points)
    H_ref_to_world.npy        (calibration)
    H_per_frame.npy           (Regime B: H_{t→W} stack, one per extracted frame)
    inliers.npy               (per-frame registration quality — the gate reads this)
    ground.png                (median mosaic, vertically flipped for Unity UV)
    world_bounds.json         (auto-computed x0, z0, W_m, H_m, ppm)
    tracks_world.json
  scripts/         ingest.py, track_agents.py, geometry.py, calibrate.py, build_mosaic.py, map_to_world.py, run.py
  UnityProject/    Assets/Generated/ (ground.png + tracks_world.json + scene_meta.json copied here)
```

---
