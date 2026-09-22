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

## Stage 3 artifact contract (`stage3-tracks-v1`)

The publisher writes to a temporary sibling directory and atomically replaces the old
bundle only after completion.

| Artifact | Purpose |
| --- | --- |
| `report.json` | Pass/fail, exact options, input and model hashes, counts, timing, quality metrics, warnings, failures, and relative output paths |
| `detections.jsonl` | Audit record for every merged detection: frame, source bbox, canonical kind, score, reference anchor, and world anchor |
| `tracks.json` | Compact Unity contract containing track metadata and ordered playback samples |
| `tracking_preview.mp4` | CFR video annotated with detection, identity, observed/predicted state, and confidence |
| `track_map.png` | All world paths over the Stage 1 ground image, with start/end and track IDs |
| `frame_counts.csv` | Per-frame car/person detection and active-track counts for discontinuity review |

Proposed `tracks.json` shape:

```json
{
  "schema_version": "stage3-tracks-v1",
  "fps": 24.0,
  "frame_count": 240,
  "duration_s": 10.0,
  "scale_confidence": "approximate_assumed_scale",
  "tracks": [
    {
      "id": 1,
      "kind": "car",
      "first_frame": 12,
      "last_frame": 239,
      "mean_confidence": 0.86,
      "samples": [
        {
          "frame": 12,
          "t": 0.5,
          "x": -3.42,
          "z": 1.91,
          "heading_deg": 88.2,
          "speed_units_per_s": 2.74,
          "observed": true,
          "confidence": 0.91
        }
      ]
    }
  ]
}
```

Timestamps are derived, never accumulated: `t = frame / fps`. Samples are sorted and
unique by frame. No NaN/Infinity values are permitted.

## Quality gates and diagnostics

Hard failures:

- either upstream stage is missing, failed, stale, or hash-inconsistent;
- CFR metadata does not match Stage 1 exactly;
- transform stack shape is not `(frame_count, 3, 3)` or contains invalid matrices;
- detector weights or class map differ from the declared configuration;
- output time is not monotonic or a track has duplicate frames;
- a mapped observed point is non-finite;
- preview video frame count/FPS differs from the source timeline;
- an internal exception occurs before atomic publication.

Warnings/review metrics rather than universal hard failures:

- zero actors;
- high low-confidence recovery ratio;
- short-track/fragment count;
- identity-join ambiguity count;
- long hidden intervals;
- p50/p95 stationary jitter;
- implausible speed, acceleration, or turn-rate samples;
- sudden per-frame count changes away from image boundaries;
- tracks born/dying in the interior;
- detections excluded at the finite reconstructed-world boundary;
- large percentage of detector observations left unassigned.

Approximate Stage 2 scale requires the report and Unity UI to say `units/s`, not claim
measured `m/s`. Physical speed gates become approximate and should also have a
reference-pixel fallback based on actor size.

## Accuracy evaluation plan

Before calling Stage 3 accurate, label representative frames from the actual footage:

1. sample dense difficult intervals, not just evenly spaced easy frames;
2. label cars and individual people, including partial occlusions and boundaries;
3. preserve identities through consecutive frames for a smaller MOT subset;
4. keep train/tune/test clips or intervals separate;
5. evaluate detection AP/precision/recall by kind and object pixel size;
6. evaluate tracking HOTA, DetA, AssA, IDF1, ID switches, fragmentation, and mostly
   tracked/lost with TrackEval;
7. choose detector, tile settings, confidence thresholds, tracker, and ReID only from
   held-out results;
8. archive the annotations, exact command, hashes, and metrics with the chosen model.

The first target should be recall at useful precision for tiny people, then identity
association. A sophisticated tracker cannot recover actors that were never detected.

## Unity architecture

Stage 3 Unity integration will mirror the previous fail-closed stages:

1. `Stage3PipelineWindow` selects Stage 1, Stage 2, model, and output directory and runs
   Python asynchronously.
2. `Stage3BundleImporter` validates schema, pass state, hashes, dimensions, FPS, track
   ordering, coordinates, and kind names before modifying the scene.
3. `Stage3TrackAsset` stores the validated runtime data; imported JSON remains available
   as an audit artifact.
4. `Stage3PlaybackController` owns one authoritative simulation clock and supports
   play, pause, seek, speed, loop, and reset.
5. Actors spawn only between their first and last visible sample. Short estimated gaps
   remain visible but can be tinted in an optional diagnostics mode.
6. Position is linearly interpolated between adjacent samples. Heading uses wrapped
   angular interpolation. Teleports across non-joined gaps are forbidden.
7. Cars and people use separate prefab slots and optional pooling. A missing custom
   prefab falls back to the sample prefabs.
8. Actors are placed at `calibration.GroundYMetres`; prefab pivots are at ground level.

Initial visual dimensions are 4.2 x 1.8 x 1.5 units for cars and 1.75 units high for
people. They are placeholders for readability and collision testing, not measurements
of individual detected actors. Stage 4 may replace them without changing the track
contract.

## Implementation sequence

### 3A — contracts and fixtures

- define dataclasses/options and strict JSON schemas;
- implement upstream provenance validation and exact frame clock;
- compose the Stage 1 and Stage 2 transforms;
- create synthetic car/person paths with zoom/jitter, crossings, stops, and gaps;
- unit-test coordinate mapping, frame indexing, atomic failure, and serialization.

### 3B — cached high-recall detections

- extract detector adapter shared with (but not coupled to) Stage 1 masking;
- run full-frame plus tile inference on every CFR frame;
- normalize only `car`, `pedestrian`, and `people`;
- merge duplicates deterministically and cache `detections.jsonl`;
- create the first annotated preview and detector-only metrics.

### 3C — stabilized tracking

- transform anchors/boxes into reference coordinates before association;
- implement high/low confidence association and track lifecycle;
- use appearance histograms as a conservative identity cue and strictly bound gaps;
- add forward/backward reconciliation and conservative gap joining;
- add confidence-aware filtering/smoothing and heading derivation;
- generate `tracks.json`, track map, frame counts, and report.

### 3D — benchmark and tune

- add MOTChallenge import/export for the labelled subset;
- run ByteTrack, BoT-SORT, OC-SORT, optional Deep OC-SORT/TrackTrack, and custom tracker
  against the exact cached detections;
- choose per-kind settings from held-out HOTA/AssA/IDF1 and detection recall;
- fine-tune the detector on project footage if detector recall is the bottleneck.

### 3E — Unity playback

- implement importer, ScriptableObject, clock, spawner, interpolation, diagnostics UI,
  and scene builder;
- use the sample car/person prefabs as defaults;
- add an editor self-test with synthetic tracks before importing real output;
- verify first/last frames, seek, loop, pause, heading, ground contact, and scale warning.

### 3F — acceptance

- run Stage 1 -> Stage 2 -> Stage 3 consecutively from the editor;
- inspect `tracking_preview.mp4` and `track_map.png`;
- overlay Unity top-down output against the source at multiple frames;
- record limitations and only then declare Stage 3 complete.

## Primary references

- VisDrone dataset and aerial MOT benchmark: https://github.com/VisDrone/VisDrone-Dataset
- Ultralytics tracking modes/configuration: https://docs.ultralytics.com/modes/track/
- ByteTrack: https://arxiv.org/abs/2110.06864
- BoT-SORT: https://arxiv.org/abs/2206.14651
- OC-SORT: https://openaccess.thecvf.com/content/CVPR2023/html/Cao_Observation-Centric_SORT_Rethinking_SORT_for_Robust_Multi-Object_Tracking_CVPR_2023_paper.html
- Deep OC-SORT: https://arxiv.org/abs/2302.11813
- TrackTrack: https://openaccess.thecvf.com/content/CVPR2025/papers/Shim_Focusing_on_Tracks_for_Online_Multi-Object_Tracking_CVPR_2025_paper.pdf
- SAHI: https://arxiv.org/abs/2202.06934
- TrackEval/HOTA reference implementation: https://github.com/JonathonLuiten/TrackEval
- RF-DETR (benchmark candidate): https://github.com/roboflow/rf-detr
- SAM 2 (optional mask propagation): https://github.com/facebookresearch/sam2
