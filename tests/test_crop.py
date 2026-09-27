"""Unit tests for square patch extraction and bounding box logic."""

import unittest
from PIL import Image
from scripts.fetch_real_data import extract_square_crop


class TestPatchExtraction(unittest.TestCase):
    """Test geometric polygon bounding-box cropping and image border clamping."""

    def setUp(self):
        # Create a 200x200 canvas
        self.img = Image.new("RGB", (200, 200), color=(50, 100, 150))

    def test_extract_square_crop_dimensions_and_aspect_ratio(self):
        """Cropped patch must be square and centered on the polygon vertices."""
        # A polygon centered at (100, 100) spanning 20x40 px
        pts = [[90, 80], [110, 80], [110, 120], [90, 120]]
        crop, box = extract_square_crop(self.img, pts, margin=1.0)

        # Width and height of the box should be equal
        x0, y0, x1, y1 = box
        self.assertEqual(x1 - x0, y1 - y0)
        self.assertEqual(crop.width, crop.height)
        self.assertEqual(crop.width, 40)

    def test_extract_square_crop_border_clamping(self):
        """Cropping near the image edge must clamp cleanly without throwing IndexError."""
        # Polygon right against top-left corner
        pts = [[2, 2], [10, 2], [10, 10], [2, 10]]
        crop, box = extract_square_crop(self.img, pts, margin=2.0)

        x0, y0, x1, y1 = box
        self.assertGreaterEqual(x0, 0)
        self.assertGreaterEqual(y0, 0)
        self.assertLessEqual(x1, self.img.width)
        self.assertLessEqual(y1, self.img.height)


if __name__ == "__main__":
    unittest.main()
