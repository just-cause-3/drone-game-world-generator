# Workflow robustness and accuracy audit

Audit date: 2026-09-30.

## Scope and workflow

The active workflow is `full_pipeline.py` or `fast_pipeline.py`, followed by Unity
import and explicit semantic acceptance. Stage 1 validates camera motion and creates
the reference ground. Stage 2 maps reference pixels to world coordinates. Stage 3
detects and tracks cars and people. Stage 4A proposes semantic labels for review;
accepted labels can enter Stage 4B-C for provisional geometry.

The older `run.py` workflow and `unity_scripts/` are reference implementations. They
are not interchangeable with the staged artifact contracts. Existing documentation,
completed runs, and unrelated working-tree changes were preserved during this audit.

## Corrected issues

| Area | Problem | Correction |
| --- | --- | --- |
| Stage 1 registration | Features on detected moving actors could influence camera alignment. | Foreground masks now exclude those features in both the source sample and reference before registration. Reference detection is reused. |
| Stage 2 / full-run input | NaN, infinity, or invalid world widths could reach processing before rejection. | Require a finite positive width before creating a run or running inference. |
| Numeric options | Non-finite thresholds could bypass comparisons or produce invalid geometry. | Reject non-finite numeric options across all staged processors. |
| Stage 3 detections | Class confidence filtering removed weak pedestrian candidates before low-confidence association. | Keep candidates down to the low threshold; apply class-specific thresholds when starting identities. Weak candidates only continue existing tracks through the association gates. |
| Stage 3 continuity | The lost-frame limit counted elapsed frames, making a limit of zero break even consecutive detections. | Count missing frames between observations. Reject inference strides that exceed the interpolation budget. |
| Stage 3 smoothing | Fixed world-unit noise changed smoothing behavior with the assumed scale and flattened turns. | Smooth in reference pixels and frame units, then project into world coordinates. Keep smoothed anchors within the reference rectangle. |
| Stage 3 heading | Stopped cars snapped to image-axis headings; smoothing rebound could flip their direction. | Preserve the last motion-supported heading. Transform a never-moving car's ambiguous axis prior through calibration. |
| Python / Unity gaps | Python omitted long gaps but Unity interpolated across them. | Split smoothing across unsupported gaps; Unity hides actors during omitted intervals. |
| Fast resume | Old Stage 2/3 results could be reused after changing the requested world width, or with altered artifacts. | Verify requested and stored widths, actual upstream provenance, cached cadence, required files, and available artifact hashes before resume. |
| Detection cache | Edited detections or a different inference cadence could be replayed as if unchanged. | Add detection/output hashes and validate cache cadence. Preserve compatibility with older reports while warning about their missing weak detections. |
| Semantic input | Converting a 16-bit label PNG to 8-bit before validation could turn ID 257 into road ID 1. | Validate original label values before conversion. |
| Geometry area | A global median pixel scale could discard a region whose locally calibrated area is large enough. | Apply minimum-area checks to projected world polygons. Ignore degenerate contours; report building simplification failures. |
| Bundle publication | Stages 2/3 removed previous results before promotion; Stage 4 reused a predictable backup directory. | Use unique backups, restore the previous bundle if promotion fails, and only clean up this publication's backup. Publish the full-run JSON through file replacement. |

No quality gate thresholds were relaxed. Detector resolution, tiles, augmentation,
model checkpoint, and the mandatory semantic review gate remain as configured.
