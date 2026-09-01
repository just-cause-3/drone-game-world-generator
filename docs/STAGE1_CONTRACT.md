# Stage 1 contract: stationary top-down video to static ground

This document is authoritative for the staged rebuild. The older root documentation and
Unity files are retained only as implementation reference.

## Input contract

One run accepts one continuous, pre-recorded shot. The operator must confirm that:

- the drone is hovering in one position, with only minimal stabilization jitter;
- the camera is aimed approximately straight down (nadir);
- zoom may change gradually, but total registered scale variation must stay within 18%;
- the clip contains no edits or scene cuts;
- the visible area is approximately planar; and
- the video is the highest-quality original available, preferably already corrected for
  lens distortion by the camera.

The software checks timing, cuts, blur, registration, overlap, rotation, scale change,
coverage, foreground exclusions, and repair size. Registration can measure stability,
but it cannot prove that the view is nadir. Passing the gate therefore also requires a
visual review of the ground image and diagnostic masks.

Zoom and translation are evaluated separately. Motion of the transformed image center
must remain within 3% of the image diagonal, while corner displacement may reach 12% to
accommodate centered zoom. At least 75% frame overlap must remain after normalization.

Every consecutive frame pair is also audited: at least 99% must produce trusted geometry,
and no pair may exceed 1% scale change, 0.5% center movement, or 0.25 degrees rotation.
Scene cuts remain an independent hard failure.

Accepted samples are normalized directly into the temporal-center reference image with
their measured similarity transforms. Color frames use Lanczos4 resampling; validity and
foreground masks use nearest-neighbor resampling. `registration.json` reports source and
post-normalization Laplacian sharpness plus retention percentage for every accepted
sample. `report.json` includes minimum/median normalized sharpness and retention so
resampling softness is visible even when all other gates pass. If more than 25% of
accepted normalized samples fall below the sharpness threshold, the stage fails.

## Versioned artifact bundle

`report.json` has `schema_version: stage1-ground-v1`. Paths in its `outputs` object are
relative to the directory containing the report, allowing the complete bundle to move
without rewriting its manifest. Consumers must stop unless `pass` is `true`.

| Artifact | Contract |
| --- | --- |
| `report.json` | Provenance, exact options, metrics, assumptions, limitations, gate result, and relative artifact paths. A processing exception produces a report-only failed bundle. |
| `video_cfr.*` | Exact video timeline all later stages must consume. It is a copy when the source is already constant-frame-rate, otherwise an H.264 CFR normalization. |
| `ground.png` | BGRA/RGBA-compatible static plate in reference-image pixel space. Pixel origin is top-left; alpha 255 means supported/repaired and alpha 0 means unresolved. |
| `valid_mask.png` | The alpha/validity channel as a separate 8-bit mask. |
| `frame_transforms_to_reference.npy` | Float64 array shaped `(frame_count, 3, 3)`. Each matrix maps a homogeneous pixel from that CFR video frame into reference-image pixels. Only consume it when the report passes. |
| `sample_transforms_to_reference.npy` | Float64 array shaped `(sample_count, 3, 3)`. Rejected samples contain `NaN`; accepted transforms are directly estimated. |
| `registration.json` | Per-sample matches, inliers, error, overlap, motion, rotation, scale, acceptance, and rejection reason. |
| `frame_geometry.json` | Every consecutive-frame similarity estimate and its frame indices, scale step, center motion, rotation, overlap, inliers, and trust result. |
| `detection_preview.png` | Reference frame annotated with the merged aerial detections. This is a required human-review artifact, not decoration. |
| `reference_foreground_mask.png` | Exact padded/dilated binary actor mask used for the reference sample. |
| `foreground_frequency.png` | How often a selected foreground class covered each pixel in accepted samples. |
| `semantic_exclusion_mask.png` | Diagnostic candidate regions repeatedly classified as actors. It does not erase clean actor-free observations; use it to review high-occupancy areas. |
| `inpaint_mask.png` | Small exclusion regions repaired conservatively. Large connected regions remain invalid and fail the gate. |
| `temporal_variation.png` | Robust temporal color-change diagnostic after registration and masking. |

Outputs are staged in a temporary sibling directory and replace the previous bundle as
one publication operation. This prevents a later stage from reading a mixture of old and
new artifacts.

## Foreground definition

The default static world keeps roads, markings, pavements, vegetation, structures, and
non-car vehicle types. It excludes only pedestrians (`pedestrian` and `people`) and
`car`. The `person` alias is accepted when an explicitly selected COCO checkpoint is
used. Bicycles, motorcycles, vans, trucks, buses, tricycles, and `others` are not masked.
Class names are matched by name rather than numeric ID so a COCO class map cannot
silently be applied to VisDrone weights.

Inference combines a whole-frame pass with overlapping 640 px crops resized to 1280 px,
uses test-time augmentation, and performs global class-agnostic NMS before the masks are
created. The mask threshold is intentionally recall-oriented (`0.08`); the final static
plate residual gate uses the more precision-oriented `0.25`. Box padding and dilation
cover the full actor silhouette and small registration errors. These defaults trade
runtime for small-object recall.

If the final detector finds a small residual car or pedestrian, Stage 1 may perform up
to two additional conservative repair passes. Those pixels count toward the same 2%
total inpainting limit, large components remain unresolved, and the final plate must be
checked again by the detector before it can pass.

This is a project policy, not an inherent truth: decide before the next stage whether
parked vehicles should be part of the world or removed as actors. No detector guarantees
perfect detection, especially for a new altitude, camera, geography, or lighting. The
overlay, masks, temporal frequency, and final plate therefore remain required review
artifacts. The report records the exact weight SHA-256, model class map, thresholds, and
inference strategy for reproducibility.

Install/verify the pinned checkpoint with:

```powershell
.venv\Scripts\python.exe setup_drone_model.py
```

`train_drone_model.py` can reproduce a VisDrone fine-tune from an official Ultralytics
base model. A production detector should ultimately be fine-tuned on annotated frames
from the actual camera, altitude, and target environment, then evaluated against a
held-out project-specific set before replacing the pinned baseline.

The pinned checkpoint and Ultralytics implementation are AGPL-3.0 licensed. Resolve
distribution/licensing requirements before the generated application is shipped.

## What must be defined for metric Unity coordinates

Stage 1 deliberately stops in pixels. Before building Unity geometry, define:

1. At least four well-spread, non-collinear ground control points. For every point,
   record its reference-image pixel `(u, v)` and matching real ground coordinate
   `(X, Z)` in meters. Surveyed points are best; known corners or road-marking
   intersections are practical alternatives.
2. A world origin, such as one chosen control point at `(0, 0)`.
3. Axis directions. Recommended: Unity `+X` is east/right, `+Y` is up, and `+Z` is
   north/forward. Image `v` points downward, so it must not be copied directly to `Z`.
4. Ground elevation. Use `Y = 0` for the first flat-world version.
5. The intended static-world policy, especially whether stationary vehicles, people,
   temporary objects, trees, and shadows belong in the generated world.

The next calibration artifact should be `world_calibration.json`, versioned separately,
and contain the control points, a 3x3 `reference_pixel_to_world_xz` homography, reprojection
RMSE, world bounds, axis convention, and units. Four or more points allow slight residual
perspective to be represented and independently checked. One known distance can estimate
meters per pixel, but it cannot validate skew or perspective and is not recommended for
the final reconstruction.

For a point `p = [u, v, 1]` observed in CFR frame `n`, the later pipeline composes:

```text
p_reference = frame_transforms_to_reference[n] @ p
p_world_xz   = reference_pixel_to_world_xz @ p_reference
p_world_xz  /= p_world_xz[2]
unity        = (p_world_xz[0], 0, p_world_xz[1])
```

This single, explicit transform chain prevents the vertical flips and mixed coordinate
conventions present in the legacy implementation.
