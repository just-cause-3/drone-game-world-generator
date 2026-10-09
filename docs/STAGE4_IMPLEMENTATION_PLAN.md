# Stage 4 implementation plan: semantic ground to an editable 3D world

## Goal and truth boundary

Stage 4 converts the accepted Stage 1 reference plate and Stage 2 pixel-to-world
mapping into editable static Unity geometry. It does not claim that a stationary,
near-nadir monocular video contains measured 3D. With almost no camera baseline there
is no useful parallax, roof pixels are not guaranteed to be wall-footprint pixels, and
absolute height is unobservable without another cue.

The output is therefore split into three confidence levels:

1. **Observed:** image-space semantic labels and boundaries visible in the Stage 1
   plate.
2. **Mapped:** 2D vertices transformed through the exact Stage 2 homography into the
   selected Unity X/Z coordinate system.
3. **Assumed:** curb offsets, building heights, roof shapes, tree dimensions, facade
   appearance, and any other quantity not measured by the input.

Every artifact and Unity component must preserve this distinction. An approximate
Stage 2 calibration also means all generated horizontal dimensions are approximate
Unity units, even when JSON field names retain `_m` for schema compatibility.

## Input constraints inherited from Stages 1-3

- One continuous, cut-free, nearly stationary, near-nadir shot.
- A passing `stage1-ground-v1` bundle is the only image source. Stage 4 must not reopen
  the original video and choose a different frame.
- A passing `stage2-calibration-v1` bundle supplies the only pixel-to-world transform.
- Image origin is top-left; `u` points right and `v` points down.
- Unity uses `(X, Y, Z)` with ground at Stage 2's configured `Y`.
- The valid mask limits all analysis. Unsupported Stage 1 pixels cannot become geometry.
- Cars and people are dynamic Stage 3 actors, not static Stage 4 geometry.
- Model weights, reports, masks, and calibrations are bound by SHA-256.
- A failed or stale stage must not modify a later Unity scene.
- Generated bundles are published atomically and may only replace a directory bearing
  that stage's private marker.
- The unmeasured workflow is supported, but no UI or report may describe assumed scale
  or height as surveyed truth.

## Technology decision

### Semantic proposal

The default adapter uses a pinned Mask2Former-Swin checkpoint fine-tuned on
OpenEarthMap's eight aerial/satellite land-cover classes. Padded whole-image context is
combined with overlapping square crops and horizontal-flip TTA. The previous
Cityscapes checkpoint is retained only as a conservative road specialist: it may
promote a pixel to `road`, but none of its building or other class predictions are
allowed into the fused output. On the current intersection this reduced the proposed
building coverage from roughly 62% to roughly 8% while recovering the central road.

The durable interface remains backend-independent. It also accepts a corrected canonical
single-channel label PNG. A future project-trained aerial semantic model can replace the
bootstrap adapter without changing the Stage 4 artifact or Unity contracts.

Preferred accuracy path:

1. Bootstrap labels with an aerial land-cover model trained on OpenEarthMap-like
   categories (road, developed surface, building, vegetation, terrain, water).
2. Correct a diverse project-specific set of actual drone plates.
3. Fine-tune and validate a dense semantic model using video/location-level splits.
4. Optionally use SAM 2 to refine operator-selected boundaries. SAM 2 is promptable and
   class-agnostic, so it cannot replace the semantic classifier by itself.
5. Treat SAM 3 concept prompting as an optional adapter only. Its checkpoints are gated,
   its environment is currently CUDA-specific, and it uses a separate SAM license.

OpenEarthMap is a suitable aerial taxonomy reference: it contains 5,000 high-resolution
aerial/satellite images, eight land-cover classes, and global regions, but individual
source-data licenses vary. Dataset/model licensing must be resolved before redistribution.

Primary references:

- OpenEarthMap: <https://arxiv.org/abs/2210.10732>
- Ultralytics semantic segmentation: <https://docs.ultralytics.com/tasks/semantic/>
- SAM 2: <https://github.com/facebookresearch/sam2>
- SAM 3: <https://github.com/facebookresearch/sam3>
- Default OpenEarthMap Mask2Former checkpoint:
  <https://huggingface.co/mfaytin/mask2former-satellite>

The checkpoint is pinned to repository revision
`558c4959866ddd8dca42c13d24e49ca3e12bb71e` and its safe-tensor SHA-256 is recorded in
the setup script and every report. Its configuration exposes eight output logits in
the original OpenEarthMap order. The accompanying model card displays a shifted
nine-class table including background; the adapter deliberately follows the verified
checkpoint tensor schema and original eight-class dataset taxonomy instead.

### Boundary refinement and vectorization

Raster labels remain authoritative until explicitly accepted. For each accepted class:

- connected components remove speckles below a configurable image/world area;
- contours use hierarchy-aware extraction so courtyards and islands remain holes;
- simplification uses a bounded pixel/world error rather than an arbitrary vertex cap;
- building edges may be line-fitted and snapped to dominant near-orthogonal directions,
  but only when the fit error improves;
- every simplified pixel vertex is mapped individually through the Stage 2 homography;
- polygons are checked for finite coordinates, winding, duplicate vertices,
  self-intersection, degenerate area, and containment of holes;
- original raster masks and overlay remain available for comparison.

Roads and pavements remain area polygons in the first version. Lane centreline topology
is a later optional product; it must not be inferred merely from the outline of a large
intersection.

### Height and roof policy

The first 3D generator uses an explicit `stage4-height-policy-v1` file. Suggested
defaults are deliberately editable, for example:

- pavement/curb offset: 0.12 Unity units;
- one-storey building: 3.2 units;
- assumed storeys when unknown: 2;
- flat roof unless an operator chooses another roof type;
- no generated poles/trees until their placement is reviewed.

Possible later evidence sources:

- known building/storey metadata;
- operator-entered per-building height;
- shadow length plus known sun elevation/direction (`height = shadow_length *
  tan(sun_elevation)`), only after the shadow belongs to the correct structure;
- relative monocular depth as a boundary/ranking hint, never an absolute height source.

Depth Anything V2 predicts robust monocular relative depth, but the Small weights are
Apache-2.0 while larger weights are CC-BY-NC-4.0, and a generic model cannot create
metric truth from this fixed top-down shot: <https://github.com/DepthAnything/Depth-Anything-V2>.

### Unity geometry

- Ground remains the validated Stage 2 mesh and texture.
- Road/terrain overlays are triangulated 2D meshes with small, documented Y offsets.
- Building polygons are extruded into wall and roof submeshes with outward-facing
  winding, normals, UVs, materials, and mesh colliders.
- Roof textures may sample the Stage 1 image; facades use procedural materials because
  facades are not visible in a nadir plate.
- Objects are grouped under generated roots by class and retain stable source IDs.
- Generated components record semantic confidence, source hashes, height source,
  assumed/measured status, and original polygon ID.
- Stage 3 playback is composed into the final scene only after static geometry passes.

## Staged implementation

### Stage 4A - semantic proposal and review (current milestone)

Inputs: passing Stage 1 + Stage 2, plus either an automatic backend or corrected label
PNG.

Outputs:

- canonical `semantic_labels.png`;
- `semantic_overlay.png`;
- one binary class mask per canonical class;
- connected-component audit JSON;
- provenance/quality `report.json`;
- Unity `Stage4SemanticReview.unity` scene.

Passing Stage 4A means the artifact is structurally valid, not that the labels are
correct. The operator must inspect the overlay and create a hash-bound acceptance record.

### Stage 4B - accepted raster to validated polygons

- Require `stage4-semantic-acceptance-v1` matching the exact Stage 4A report and labels.
- Extract, simplify, regularize, map, and validate polygons.
- Emit pixel and Unity X/Z coordinates plus comparison overlays.
- Do not generate meshes on failed topology or stale acceptance.

### Stage 4C - height policy and building generation

- Require an explicit height policy.
- Create extruded building/curb/static-region meshes.
- Record every assumed height and roof type.
- Validate mesh manifoldness, winding, bounds, collider complexity, and overlaps.

### Stage 4D - optional objects and appearance

- Detect/review trees, poles, signals, lamps, and other static props.
- Replace placeholders with configured prefabs.
- Generate roof/facade materials and LOD/collider policies.

### Stage 4E - composition and final QA

- Compose Stage 4 geometry with Stage 3 actor playback.
- Check actors against ground, buildings, and world boundaries over the full timeline.
- Produce a final scene manifest with the complete Stage 1-4 provenance chain.

## Stage 4A quality gate

Hard failures:

- stale or failed Stage 1/2 input;
- missing, unreadable, escaping, or dimensionally inconsistent artifacts;
- unknown canonical label IDs;
- non-zero labels outside the Stage 1 valid mask;
- excessive unlabeled valid area;
- corrupt output images or mismatched output hashes;
- unsafe output replacement;
- provider/model failure or non-finite metadata.

Warnings requiring review:

- generic/non-aerial model domain;
- unusually high building, unknown, or other-static coverage;
- excessive component fragmentation;
- road not reaching an image boundary;
- no buildings or vegetation (valid for an intersection-only scene);
- approximate Stage 2 scale.

Operator acceptance is a separate gate. Stage 4B must reject an absent, revoked, or stale
acceptance record even if the Stage 4A structural gate passed.

## Testing strategy

- Unit fixtures for upstream hash and dimension validation.
- Synthetic canonical maps covering holes, multiple components, invalid labels,
  unsupported pixels, and fragmentation cleanup.
- Atomic-publication and unrelated-directory refusal tests.
- Fake semantic provider tests; unit tests never download model weights.
- Real sample smoke test producing review artifacts from `stage1_out`.
- Unity compilation/import tests for stale hashes, texture dimensions, acceptance, and
  scene preservation after rejection.
- Later Stage 4B/4C tests for polygon topology, homography mapping, triangulation,
  extrusion winding, bounds, and colliders.

## Completion criteria

Stage 4 is complete only when an operator can run Stage 1 through Stage 4 on unseen
footage, review/correct the semantic map, produce valid editable geometry, see Stage 3
actors replay over it, and trace every object back to exact source artifacts and either
an observation or an explicit assumption.
