# Stage 3 implementation plan: cars and pedestrians to Unity playback

## Goal and boundary

Stage 3 consumes the exact, passing Stage 1 and Stage 2 bundles and produces a
time-indexed recreation of only two actor kinds:

- `car`
- `person` (`pedestrian` and `people` detector labels are normalized to this kind)

It detects actors in the Stage 1 constant-frame-rate video, maintains identities,
maps every observation through the validated image transforms into Unity X/Z, derives
stable motion, and imports a passing bundle into the Stage 2 scene. Trucks, buses,
vans, bicycles, motorcycles, and every other class are intentionally excluded.

Stage 3 does not infer buildings or depth. That remains Stage 4. It also does not
claim physically accurate speed when Stage 2 uses `approximate_assumed_scale`.

## Non-negotiable design decisions

1. **Use the Stage 1 CFR video and frame numbers.** No stage may reopen the original
   input and silently create a different timeline.
2. **Stabilize before association.** A detection at frame `n` is mapped through
   `frame_transforms_to_reference[n]` before it is offered to the tracker. This removes
   the accepted zoom and hover jitter from the motion model.
3. **Map a nadir footprint centre, not an oblique-view box foot.** For the accepted
   near-nadir footage, the actor ground anchor is the centre of its detection/instance
   mask. The legacy `foot_shift` rule is for oblique footage and is not reused.
4. **Preserve observations and estimates separately.** Short gaps may be predicted or
   interpolated, but generated samples are marked and never presented as detections.
5. **Keep cars that do not move.** Stage 1 removes cars from the ground plate, so a
   consistently detected parked car is still an actor and must be recreated.
6. **An empty clip can pass.** “No actors present” is valid. Stage 3 must not invent an
   actor merely to satisfy a minimum-track count.
7. **Fail closed on provenance or geometry.** A changed Stage 1 report, CFR video,
   transform stack, ground image, or Stage 2 calibration invalidates the bundle.
8. **Require visual review.** A summary map and annotated preview video are contract
   artifacts, not optional decoration.

## Technology evaluation

### Detection

| Candidate | Fit for this project | Decision |
| --- | --- | --- |
| Pinned YOLO26x VisDrone checkpoint | Already verified and used by Stage 1; aerial class map contains `pedestrian`, `people`, and `car`; supports high-resolution and tiled inference | **Initial production detector** |
| SAHI-style sliced inference | Directly targets small objects in high-resolution aerial imagery; compatible with the existing full-frame plus overlapping-tile approach | **Use the strategy, retain our deterministic merger** |
| RF-DETR | Strong modern detector and attractive Apache-licensed base stack, but its COCO score does not establish accuracy on this camera or VisDrone | Benchmark only after fine-tuning on the same labelled set |
| Generic COCO YOLO/RT-DETR/D-FINE models | Easy to run, but their label domain and tiny-aerial-object recall are a worse starting point than the pinned VisDrone model | Do not use as the default |
| SAM 2 video segmentation | Can propagate prompted masks through video, but it is not an automatic car/person detector and tiny crowded actors are a difficult input | Optional later mask refinement, not primary tracking |

The initial detector therefore reuses the exact pinned model SHA-256 and model class
map from Stage 1. Every frame receives a whole-frame pass plus overlapping tiles. Tile
and whole-frame results are merged in source-frame coordinates. The merger must retain
the detector score and source provenance and must not merge different canonical kinds.

Accuracy cannot be selected from COCO or VisDrone headline scores alone. The final
detector decision will be made on a small project-specific, held-out annotation set.
Fine-tuning on frames from the actual camera, altitude, compression, geography, and
lighting is the largest available accuracy improvement once those labels exist.

### Multi-object association

| Candidate | Strength | Limitation here | Use |
| --- | --- | --- | --- |
| ByteTrack | Recovers useful low-confidence detections and is a strong simple baseline | Primarily online and appearance-free | Required benchmark |
| BoT-SORT + ReID | Fuses motion, IoU, appearance, and camera-motion compensation | Generic ReID is weak for tiny top-down people/cars; Stage 1 already supplies better camera geometry | Required benchmark, not blindly assumed winner |
| OC-SORT | Handles observation gaps and nonlinear motion well | Appearance-free identity recovery through dense crossings is limited | Required benchmark |
| Deep OC-SORT | Adaptively combines OC-SORT motion and ReID | Extra dependency/model and ReID domain mismatch | Optional benchmark |
| TrackTrack | Modern online association available through current Ultralytics tracking | Newer, with less project-specific evidence | Optional benchmark |
| Custom stabilized-plane tracker | Uses the exact known camera transform and can perform offline forward/backward reconciliation | More implementation and testing work | **Production architecture** |

The production tracker is not a fork of the legacy `track_agents.py`. It will associate
canonical detections in the stable reference plane, using a class-specific combination
of predicted-centre distance, transformed-box IoU, size consistency, detector score,
and optional appearance cosine distance. It will use two confidence bands in the style
of ByteTrack: high-confidence observations start/confirm tracks; plausible low-score
observations may recover an existing track but cannot create one.

Because input is prerecorded, Stage 3 can improve on a strictly online tracker:

1. run association forward;
2. run it backward;
3. reconcile agreeing tracklets;
4. solve short-gap joins with motion, time, class, size, and appearance gates;
5. retain ambiguous joins as separate identities rather than forcing one.

BoT-SORT, ByteTrack, OC-SORT, and the custom tracker will all consume the same cached
detections during evaluation. This separates detector quality from association quality.

### Motion refinement

- A confidence- and size-aware constant-velocity Kalman filter provides online state.
- An offline Rauch-Tung-Striebel pass smooths confirmed segments without averaging
  across gaps.
- Short missing intervals may use optical flow as supporting evidence. Optical flow is
  never allowed to create a new class/identity by itself.
- Cars receive acceleration, turn-rate, and reverse-motion plausibility gates.
- Pedestrians receive looser direction and acceleration gates.
- Heading comes from the smoothed X/Z velocity only above a movement threshold. At a
  stop, the last reliable heading is held. It is not inferred from an axis-aligned box.

## Exact coordinate chain

For source-frame anchor `p = [u, v, 1]` at CFR frame `n`:

```text
p_reference = frame_transforms_to_reference[n] @ p
p_reference /= p_reference[2]

p_world = reference_pixel_to_world_xz @ p_reference
p_world /= p_world[2]

unity_position = (p_world[0], ground_y, p_world[1])
```

Association operates primarily on `p_reference`; world coordinates are also retained
for validation and playback. This avoids applying Stage 2 twice and makes it possible
to diagnose whether a fault came from detection, registration, or calibration.
