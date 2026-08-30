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
