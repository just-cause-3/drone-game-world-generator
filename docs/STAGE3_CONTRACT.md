# Stage 3 contract: validated video to Unity actor playback

Stage 3 consumes the exact passing Stage 1 and Stage 2 bundles. It detects only cars
and people, associates them in the stabilized Stage 1 reference plane, maps them into
the Stage 2 Unity X/Z coordinate system, and emits playback data.

## Required upstream inputs

- `stage1_out/report.json` must be `stage1-ground-v1` with `pass: true`.
- The Stage 1 CFR video metadata must exactly match the report.
- `frame_transforms_to_reference.npy` must be finite, nonsingular, and shaped
  `(frame_count, 3, 3)`.
- `stage2_out/world_calibration.json` must be `stage2-calibration-v1` with `pass: true`.
- Its Stage 1 report, ground, source-video, and dimensions bindings must still match.

Stage 3 records new SHA-256 bindings for the Stage 1 report, CFR video, per-frame
transform stack, and Stage 2 calibration. Unity repeats these checks before importing.

## Detector policy

The initial production detector is the checksum-pinned YOLO26x checkpoint fine-tuned
for VisDrone. Inference runs on every CFR frame and combines:

- one complete-frame inference pass;
- overlapping 640-pixel crops;
- 1280-pixel inference resolution;
- test-time augmentation;
- class-specific duplicate suppression across all passes.

Only these labels are consumed:

| Detector label | Stage 3 kind |
| --- | --- |
| `car` | `car` |
| `pedestrian` | `person` |
| `people` | `person` |

Vans, trucks, buses, bicycles, motorcycles, tricycles, and every other class are
ignored. Class names are resolved from model metadata; numeric IDs are never assumed.

## Coordinate and association rule

For a detection centre `[u, v, 1]` in CFR frame `n`:

```text
reference = frame_transforms_to_reference[n] @ [u, v, 1]
reference /= reference[2]

world = reference_pixel_to_world_xz @ reference
world /= world[2]

unity = (world[0], ground_y, world[1])
```

Tracking association occurs after the first transform. Accepted camera zoom and hover
jitter therefore do not appear as actor motion. The anchor is the box centre because
the Stage 1 input contract is near-nadir; the reverted oblique-camera `foot_shift`
heuristic is not used.

Detections whose stabilized anchors lie beyond the finite reference ground (apart
from a small registration tolerance) are excluded. A tolerated sub-pixel overshoot is
clamped to the edge, so Unity never receives an actor position outside its ground.

The tracker uses high/low confidence association: high-confidence observations
can create and confirm identities, while low-confidence observations can only recover
an existing identity. Matching is class-specific and gated by stabilized predicted
position, transformed-box overlap, size consistency, and a central-crop HSV appearance
histogram. Recovery is limited to three frames so an expired identity cannot jump to a
different actor. Short gaps are interpolated and explicitly marked `observed: false`.
A constant-velocity Rauch-Tung-Striebel pass smooths confirmed observations before
playback samples are written. A segment needs at least five observed frames before it
is allowed into playback, preventing very short detector flicker from spawning an
actor.

## Artifact bundle (`stage3-tracks-v1`)

Stage 3 publishes atomically and replaces only an existing directory containing its
private `.stage3-tracks-output` marker.

| Artifact | Contract |
| --- | --- |
| `report.json` | Pass/fail, upstream/model hashes, exact options, coordinate policy, detector/tracker metadata, warnings, limitations, and quality metrics |
| `detections.jsonl` | One audit record per merged detection, including source box/anchor, stabilized reference box/anchor, world anchor, score, detector class, inference passes, and assigned identity when available |
| `tracks.json` | Unity runtime contract: exact FPS/frame count/duration and ordered car/person tracks with frame, derived timestamp, X/Z, heading, speed, observed flag, and confidence |
| `tracking_preview.mp4` | Annotated video on the exact CFR timeline |
| `track_map.png` | Confirmed stabilized trajectories drawn over the Stage 1 ground |
| `frame_counts.csv` | Per-frame detections, assignments, and active-track count |

An empty scene may pass with a warning. Stage 3 must not invent an actor to satisfy a
minimum-track gate. Short unconfirmed fragments are discarded and reported.

## Safe retracking without repeated inference

Tracker and boundary settings can be re-evaluated from an existing Stage 3 bundle:

```powershell
.venv\Scripts\python.exe stage3.py stage1_out stage2_out --output stage3_out `
  --reuse-detections-from stage3_out
```

The cache is accepted only when its Stage 1 report, CFR video, transform stack, Stage 2
calibration, frame count, and FPS still match. Frames are decoded again to recompute
appearance descriptors and diagnostics; YOLO inference alone is skipped. Loading is
completed before atomic replacement, so using the same input/output directory is safe.

## Timing

There is one clock:

```text
timestamp_seconds = CFR_frame_index / CFR_fps
```

Timestamps are derived rather than accumulated. Unity validates every timestamp against
its frame number and interpolates between adjacent playback samples using this clock.

## Approximate Stage 2 scale

When Stage 2 confidence is `approximate_assumed_scale`, actor placement is consistent
with the selected Unity ground size, but distances, prefab dimensions, and speeds are
not physical measurements. Reports and UI must use “units” rather than claim surveyed
metres.

## Running

From PowerShell:

```powershell
.venv\Scripts\python.exe stage3.py stage1_out stage2_out --output stage3_out
```

From Unity:

1. Open/import the Stage 2 metric scene.
2. Select **Drone World -> Stage 3 -> Run Stage 3 Tracking Pipeline...**.
3. Confirm the Stage 2 calibration and output folder.
4. Press **Run Stage 3** and wait for the detector to process every frame.
5. On a pass, Unity validates the bundle and creates
   `Assets/Scenes/Stage3ActorPlayback.unity`.
6. Inspect `tracking_preview.mp4` and `track_map.png`, then press Unity Play.

The full technology evaluation and later accuracy work are in
`STAGE3_IMPLEMENTATION_PLAN.md`. Forward/backward reconciliation, tracker benchmarking,
project-specific ReID, and held-out annotation metrics remain planned improvements and
are not falsely reported as part of this initial implementation.
