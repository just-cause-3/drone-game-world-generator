import unittest

from scripts.detection_merge import suppress_tile_fragments


def box(bounds, source, score=.9, kind="car"):
    return dict(bbox_xyxy=bounds, inference_pass=source, confidence=score, kind=kind)


class TileFragmentTests(unittest.TestCase):
    def test_internal_edge_fragment_prefers_complete_car_even_with_higher_score(self):
        full = box([600, 270, 735, 337], "full", .89)
        clipped = box([606, 276, 640, 338], "tile:0,0", .94)
        self.assertEqual(suppress_tile_fragments([clipped, full], (720, 1280), 640), [full])

    def test_real_cached_bottom_edge_fragment(self):
        full = box([519.99, 575.32, 586.18, 720], "full")
        clipped = box([519.48, 575.47, 585.07, 639.85], "tile:0,0")
        clipped.pop("inference_pass")
        clipped["inference_passes"] = ["tile:0,0", "tile:480,0"]
        self.assertEqual(suppress_tile_fragments([clipped, full], (720, 1280), 640), [full])

    def test_only_evidence_is_preserved(self):
        fragment = box([610, 270, 640, 330], "tile:0,0")
        self.assertEqual(suppress_tile_fragments([fragment], (720, 1280), 640), [fragment])

    def test_non_boundary_overlap_and_other_class_are_preserved(self):
        complete = box([500, 200, 700, 350], "full")
        smaller = box([520, 220, 590, 300], "tile:0,0")
        person = box([610, 270, 640, 330], "tile:0,0", kind="person")
        items = [complete, smaller, person]
        self.assertEqual(suppress_tile_fragments(items, (720, 1280), 640), items)
