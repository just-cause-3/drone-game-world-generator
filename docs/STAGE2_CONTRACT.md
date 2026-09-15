# Stage 2 contract: reference pixels to metric Unity coordinates

Stage 2 consumes one passing `stage1-ground-v1` bundle and operator-supplied,
real-world ground-control points. It emits a separately versioned
`stage2-calibration-v1` artifact. It does not detect objects, infer dimensions from
cars, or invent geographic coordinates.

When metric accuracy is not required, the Unity tool also provides
`approximate_image_extent` mode. It assigns the complete image a chosen world width,
preserves the image aspect ratio, and generates four corner correspondences
automatically. The resulting scale is explicitly labelled `approximate_assumed_scale`;
distances, object sizes, and speeds derived from it are approximate rather than measured.

## Required input

Each ground-control point binds:

- a pixel `(u, v)` in the exact Stage 1 reference image; and
- the same physical point's ground coordinate `(X, Z)` in metres.

The input schema is `stage2-control-points-v1`; see
`STAGE2_CONTROL_POINTS.example.json`.

Four non-collinear correspondences are the mathematical minimum for a planar
homography. Four points have zero independent residual degrees of freedom, however,
so Stage 2 labels that result `minimum_unverified`. Use at least six surveyed points
distributed around the image for a well-constrained result. Points clustered in one
small region are rejected even if their algebraic fit is exact.

Recommended Unity convention:

- one Unity unit equals one metre;
- `+X` is east/right;
- `+Y` is vertical;
- `+Z` is north/forward;
- the first flat-world ground elevation is `Y = 0`.

Image `v` increases downward. The fitted transform handles this explicitly; downstream
code must not copy image `v` directly into Unity `Z`.

## Output

`stage2_out/world_calibration.json` records:

- SHA-256 bindings to the Stage 1 report and ground texture;
- all supplied and reprojected control points;
- the 3x3 `reference_pixel_to_world_xz` homography;
- metric coordinates of all four image corners;
- metric world bounds;
- reprojection RMSE and maximum error;
- control-point hull coverage;
- local scale, anisotropy, handedness, and projective-validity checks;
- confidence, warnings, gate failures, options, axes, units, and ground height.

`calibration_overlay.png` is mandatory visual evidence. It labels every clicked point,
its metric coordinate, and its residual error. `control_points_used.json` is a
normalized copy of the exact input.
