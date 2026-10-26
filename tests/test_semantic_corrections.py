import unittest

import numpy as np

from scripts.semantic_corrections import remove_class_to_nearest
from scripts.stage4_semantics import CANONICAL_IDS


class SemanticCorrectionTests(unittest.TestCase):
    def test_absent_building_is_filled_from_nearest_road_or_pavement(self):
        labels = np.full((7, 11), CANONICAL_IDS["road"], np.uint8)
        labels[:, 7:] = CANONICAL_IDS["pavement"]
        labels[2:5, 3:9] = CANONICAL_IDS["building"]

        corrected = remove_class_to_nearest(
            labels, "building", ("road", "pavement"))

        self.assertFalse(np.any(corrected == CANONICAL_IDS["building"]))
        self.assertEqual(corrected[3, 3], CANONICAL_IDS["road"])
        self.assertEqual(corrected[3, 8], CANONICAL_IDS["pavement"])

    def test_other_classes_are_unchanged(self):
        labels = np.array([[CANONICAL_IDS["vegetation"],
                            CANONICAL_IDS["terrain"]]], dtype=np.uint8)
        corrected = remove_class_to_nearest(
            labels, "building", ("road", "pavement"))
        np.testing.assert_array_equal(corrected, labels)

    def test_requires_existing_replacement_class(self):
        labels = np.full((3, 3), CANONICAL_IDS["building"], np.uint8)
        with self.assertRaisesRegex(ValueError, "none"):
            remove_class_to_nearest(
                labels, "building", ("road", "pavement"))


if __name__ == "__main__":
    unittest.main()
