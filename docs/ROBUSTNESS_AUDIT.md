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

## Verification

The original suite contained 69 passing tests; the expanded suite passes all 85.
Regression coverage now includes
scale-invariant turning trajectories, stopped-car headings, missing-frame budgets,
weak pedestrian recovery, the production detector adapter, cache tampering, changed
world widths, foreground feature masks, invalid 16-bit labels, calibrated region
area, and failed publication rollback across downstream stages.

Reprocessed all 240 frames of the saved `perfect_top_down_full` run using its
existing detections, with new outputs under `.tmp/stage3_robustness_top_down`.
The same 3,184 detections produced 103 started tracks versus 107 previously,
and 62 confirmed tracks versus 63. Confirmed pedestrian tracks remained at 22.
These are continuity diagnostics, not precision/recall or identity-accuracy scores.
No held-out annotations were available to establish those scores.

Also reprocessed all 144 frames of `crossroads_6s_fast` at its original inference
stride of two. It passed with 1,682 cached detections and 25 confirmed tracks;
the separate review output is `.tmp/stage3_robustness_crossroads`.

A fresh production YOLO pass on the first frame, using the normal full-frame,
640-pixel tile, and TTA settings, produced 19 candidates: nine cars and ten people.
Four pedestrian candidates below 0.60 were retained for association. A retained
candidate is not automatically a confirmed actor.

Both Unity runtime and editor sources, including the new playback self-test,
compiled successfully with the installed Unity compiler/reference assemblies.
An isolated Unity batch launch could not execute the runtime test because the
licensing service reported no valid Editor license (exit 198). This is not a verified
Play Mode result. After activating Unity, the runtime check can be run with
`-executeMethod DroneWorld.Stage3.Editor.Stage3UnitySelfTest.Run`.

## Applying the improvements

Let Unity compile the changed scripts. Its gap-handling fix applies to existing
track assets immediately. Other processing changes take effect when their stages
are rerun; already generated artifacts are not silently rewritten.

For a quick check of tracking changes using the saved detections:

```powershell
.venv\Scripts\python.exe stage3.py runs\perfect_top_down_full\stage1 runs\perfect_top_down_full\stage2 --reuse-detections-from runs\perfect_top_down_full\stage3 --output runs\perfect_top_down_retracking
```

Import that output through **Drone World -> Stage 3 -> Import Passing Stage 3
Bundle...** with the matching Stage 2 scene open. For fast-mode caches, also pass
`--detector-frame-stride 2`. Cache reuse cannot recover candidates the earlier model
adapter discarded.

To include foreground-aware Stage 1 registration and fresh low-confidence detector
candidates, run the normal full pipeline into a new folder. Review the ground,
tracking preview, and semantic overlay before accepting geometry.

## Remaining accuracy limits

- The available Python environment reports `torch.cuda.is_available() == False`.
  Full inference currently falls back to CPU. CUDA availability must be resolved
  before expecting RTX-class processing time; this audit did not change drivers or
  the Python package environment.
- Assumed world width is not a measured scale. Sample prefab sizes and heights
  remain priors, not recovered physical dimensions.
- Near-nadir, approximately planar footage remains the input contract. Oblique
  footage, building parallax, hidden roads, and camera cuts require separate handling.
- Per-frame stabilization still interpolates sampled reference transforms. Very
  short camera jitter can therefore survive between those samples.
- Stationary-person filtering and Unity's stationary-actor visibility settings can
  suppress real people who stand still. They are deliberate precision policies,
  not proof that all suppressed detections are false.
- Learned inpainting is synthesized background. Semantic labels over it, aerial
  segmentation mistakes, and street-view road-specialist overrides require review.
- Building roofs are provisional extrusions. Courtyard holes are currently filled
  with a warning; roof-to-ground footprints and heights are not reconstructed.
- More model computation alone cannot establish accuracy. A representative set of
  manually annotated clips is still needed to measure detection recall, identity
  switches, trajectory error, and semantic overlap before tuning thresholds further.
