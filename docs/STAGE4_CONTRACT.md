# Stage 4A contract: static semantic proposal and operator review

Stage 4A consumes the exact passing Stage 1 and Stage 2 bundles. It proposes a canonical
static semantic map in Stage 1 reference-image pixels and creates review artifacts. It
does not generate 3D geometry and does not infer height.

## Canonical label image

`semantic_labels.png` is an 8-bit, one-channel PNG with the same dimensions and
top-left origin as `ground.png`.

| ID | Name | Meaning |
|---:|---|---|
| 0 | `unknown` | unsupported, dynamic, ambiguous, or not classified |
| 1 | `road` | drivable road surface including painted markings |
| 2 | `pavement` | sidewalk, pedestrian paved area, curb island |
| 3 | `building` | visible roof/structure candidate, not yet a footprint |
| 4 | `vegetation` | trees, shrubs, or dense plant canopy |
| 5 | `terrain` | grass, soil, or other non-paved ground |
| 6 | `water` | visible water surface |
| 7 | `other_static` | walls, fences, poles, signs, and unresolved static objects |

Pixels outside the Stage 1 valid mask must be `unknown`. Cars and people are never
static labels; they remain Stage 3 actors.

## Inputs and provenance

- `stage1_out/report.json`: `stage1-ground-v1`, stage 1, `pass: true`.
- Stage 1 `ground.png` and `valid_mask.png` at the declared dimensions.
- `stage2_out/world_calibration.json`: `stage2-calibration-v1`, stage 2, `pass: true`.
- Stage 2 hashes must still match the Stage 1 report, ground, source video, and image
  dimensions.
- Automatic mode additionally requires the pinned semantic checkpoint.
- Corrected/external mode requires a canonical label PNG satisfying this document.

## Bundle (`stage4-semantics-v1`)

| Artifact | Contract |
|---|---|
| `report.json` | Gate result, upstream/provider hashes, options, coverage, warnings, limitations, output paths and hashes |
| `semantic_labels.png` | Canonical label IDs |
| `semantic_overlay.png` | Ground image blended with canonical colors and legend |
| `mask_<class>.png` | Binary 0/255 mask for each non-unknown class |
| `components.json` | Per-class connected-component areas, bounds, centroids, and border contact |

The bundle is staged in a sibling temporary directory and replaces only an existing
directory containing `.stage4-semantics-output`.

## Gate semantics

`report.json.pass: true` means the inputs, provider output, label schema, dimensions,
coverage, and artifact hashes are structurally valid. It does not mean semantic accuracy
has been established. The report therefore always contains:

```json
{
  "requires_human_review": true,
  "ready_for_geometry": false
}
```

Unity may create `Stage4SemanticReview.unity` from a passing proposal. It must not create
3D geometry until the operator records acceptance.

## Acceptance record

After inspecting the ground and overlay, Unity may write
`semantic_acceptance.json` with schema `stage4-semantic-acceptance-v1`. It contains the
exact Stage 4 report SHA-256, semantic-label SHA-256, acceptance state, and UTC time.

Stage 4B must recompute both hashes and reject missing, revoked, or stale acceptance.
Regenerating Stage 4A invalidates the old acceptance automatically.

## Automatic-provider limitation

The default provider uses a pinned Mask2Former-Swin checkpoint fine-tuned on the eight
OpenEarthMap aerial/satellite classes. It combines padded global context with
overlapping square crops and flip TTA. A pinned Cityscapes model is used only as a road
specialist: its output can promote a pixel to canonical `road`, but its building and all
other predictions are discarded. This conservative fusion fixes the observed
developed-space/road ambiguity without restoring the old false-building failure.

OpenEarthMap imagery is typically 0.25–0.5 m GSD. Different ground resolution,
geography, shadows, seasons, camera rendering, or synthetic footage can still fail.
The overlay must be reviewed, and a corrected canonical PNG remains the authoritative
path when boundaries are wrong.
