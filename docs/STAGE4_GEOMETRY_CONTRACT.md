# Stage 4B-C contract: accepted semantics to provisional Unity geometry

Stage 4B-C consumes an operator-accepted Stage 4A semantic bundle. It vectorizes
accepted regions in Stage 1 reference pixels, maps them through the exact Stage 2
homography, assigns an explicit assumed-height policy, and creates editable Unity
geometry. It never treats monocular height as measured truth.

## Mandatory acceptance gate

The input directory must contain:

- passing `stage4-semantics-v1` `report.json`;
- `semantic_labels.png` whose SHA-256 matches that report;
- accepted `stage4-semantic-acceptance-v1` `semantic_acceptance.json`;
- acceptance hashes matching the exact report and label bytes;
- unchanged passing Stage 1 and Stage 2 bundles referenced by Stage 4A.

Missing, revoked, malformed, or stale acceptance is a hard failure. Regenerating the
semantic proposal invalidates geometry authorization.

## Geometry bundle

The output marker is `.stage4-geometry-output`, and the schema is
`stage4-geometry-v1`.

| Artifact | Purpose |
|---|---|
| `report.json` | Gate status, source hashes, options, counts, warnings, limitations, and artifact hashes |
| `geometry.json` | World-X/Z building footprints, roof triangles, assumed heights, surface boundaries, and optional vegetation proxies |
| `geometry_overlay.png` | Source ground with vectorized building outlines, stable IDs, and proxy positions |

Publication is atomic. A non-empty output without the geometry marker is never
replaced.

## Horizontal geometry

- Accepted class ID `3` is treated as visible roof/structure regions.
- Small regions below the configured world-area threshold are omitted.
- Boundaries are simplified with a tolerance expressed in Stage 2 world units.
- Each simple outer building polygon is mapped through
  `reference_pixel_to_world_xz` and ear-clipped into roof triangles.
- Invalid, non-finite, out-of-bounds, or non-triangulable building polygons stop the
  gate rather than producing a broken mesh.
- Interior roof holes are recorded and warned about. The provisional mesh currently
  fills them; the operator must inspect courtyard buildings.
- Non-building class boundaries are retained as surface-region references. The
  validated Stage 2 mesh remains the authoritative ground and collision surface.

## Height policy

Every generated height has `height_is_assumed: true`.

- `fixed`: an operator-configured height is applied to each building. This is the
  safest default when no height evidence exists.
- `footprint_prior`: an assumed floor count is varied from roof area and multiplied by
  the configured floor height. This improves visual variety only; it is not more
  physically accurate.

Stationary top-down monocular video has no stereoscopic or motion-parallax baseline.
Generic relative-depth networks cannot turn this input into metric height. A later
version may replace an individual assumed height only when explicit evidence is
recorded, such as surveyed data, known storeys, LiDAR, or a validated shadow/sun model.

## Vegetation policy

Vegetation proxies are disabled by default. When enabled, distance-transform maxima
inside accepted vegetation regions provide spacing-based canopy placeholders with
bounded counts and assumed dimensions. They are not detected trunks and should only
be enabled after confirming the accepted vegetation mask represents canopy rather
than grass or generic green space.

## Unity output

The importer rechecks every source and output hash before modifying Unity. A passing
bundle creates `Assets/Scenes/Stage4GeneratedWorld.unity` with:

- Stage 2 textured ground and collider;
- matching Stage 3 car/person playback when available;
- extruded building roof/wall meshes and mesh colliders;
- optional trunk/canopy placeholders;
- an angled perspective preview camera and directional light;
- a generated-world inspector for building, vegetation, and semantic-overlay
  visibility.

Generated meshes and metadata remain editable. The scene name and hierarchy visibly
mark provisional and assumed content.

## Required human checks

Before using the world:

1. compare every extrusion with `geometry_overlay.png` and the source ground;
2. check roof overhangs, joined buildings, courtyards, and shadows;
3. confirm that the selected assumed height is visually reasonable but never report it
   as measured;
4. play the full Stage 3 timeline and check actors against buildings and world bounds;
5. keep vegetation disabled unless canopy placement is credible.
