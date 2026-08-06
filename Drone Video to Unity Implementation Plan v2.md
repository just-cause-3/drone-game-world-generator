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
