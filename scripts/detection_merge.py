"""Conservative removal of crop-edge fragments before cross-tile NMS."""
from __future__ import annotations


def suppress_tile_fragments(candidates: list[dict], image_shape: tuple[int, int],
                            tile_size: int, *, margin: float = 8.0) -> list[dict]:
    """Keep a fragment when it is the only evidence; prefer a covering whole box.

    IoU alone misses a clipped quarter-car inside a whole-car box. Require a
    matching canonical class, an actual internal crop boundary, >=85% coverage
    of the smaller box, and a substantially larger box from another pass.
    Do NOT use containment alone on crowds or independently overlapping cars.
    Accept both raw and cached inference-pass metadata.
    """
    if tile_size <= 0:
        return list(candidates)
    height, width = image_shape[:2]

    def sources(item):
        return set(item.get("inference_passes", []) or [item.get("inference_pass", "")])

    def kind(item):
        value = item.get("kind", item.get("class", ""))
        return "person" if value in {"pedestrian", "people", "person"} else value

    def clipped(item):
        passes = sources(item)
        if not passes or any(not p.startswith("tile:") for p in passes):
            return False
        x1, y1, x2, y2 = item["bbox_xyxy"]
        for name in passes:
            try:
                x0, y0 = map(int, name[5:].split(","))
            except ValueError:
                return False
            right, bottom = x0 + min(tile_size, width), y0 + min(tile_size, height)
            if not ((x0 > 0 and abs(x1 - x0) <= margin) or
                    (y0 > 0 and abs(y1 - y0) <= margin) or
                    (right < width and abs(x2 - right) <= margin) or
                    (bottom < height and abs(y2 - bottom) <= margin)):
                return False
        return True

    areas = [max(0., b[2] - b[0]) * max(0., b[3] - b[1])
             for b in (item["bbox_xyxy"] for item in candidates)]
    kept = []
    for index, candidate in enumerate(candidates):
        remove = False
        if areas[index] > 0 and clipped(candidate):
            a = candidate["bbox_xyxy"]
            for other_index, other in enumerate(candidates):
                if (kind(candidate) != kind(other) or areas[other_index] < areas[index] * 1.5 or
                        other["confidence"] < candidate["confidence"] * 0.75 or
                        not (sources(other) - sources(candidate))):
                    continue
                b = other["bbox_xyxy"]
                intersection = (max(0., min(a[2], b[2]) - max(a[0], b[0])) *
                                max(0., min(a[3], b[3]) - max(a[1], b[1])))
                if intersection / areas[index] >= 0.85:
                    remove = True
                    break
        if not remove:
            kept.append(candidate)
    return kept
