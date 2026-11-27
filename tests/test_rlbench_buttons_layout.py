import unittest
from collections import namedtuple
from types import SimpleNamespace

from rlbench_buttons.layout import restrict_to_center


class ButtonLayoutTest(unittest.TestCase):
    def setUp(self):
        box_type = namedtuple('Box', 'min_x max_x min_y max_y min_z max_z')
        self.box = box_type(-0.2, 0.6, -0.5, 0.5, 0.1, 0.2)
        self.region = SimpleNamespace(_boundary_bbox=self.box)
        self.boundary = SimpleNamespace(_boundaries=[self.region])

    def test_scales_axes_around_original_center(self):
        restrict_to_center(self.boundary, 0.7)
        bounds = self.region._boundary_bbox
        for actual, expected in zip(bounds, (-0.08, 0.48, -0.35, 0.35, 0.1, 0.2)):
            self.assertAlmostEqual(actual, expected)

    def test_default_preserves_original_box(self):
        restrict_to_center(self.boundary, 1.0)
        self.assertIs(self.region._boundary_bbox, self.box)

    def test_rejects_invalid_scale(self):
        for scale in (0, -0.5, 1.01, float('nan'), float('inf')):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                restrict_to_center(self.boundary, scale)


if __name__ == '__main__':
    unittest.main()
