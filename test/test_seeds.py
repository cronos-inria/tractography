import tempfile
import unittest
from collections import namedtuple
from pathlib import Path

import numpy as np

import tractography as tg
import test


class TestFromSurface(unittest.TestCase):
    def test_simple(self):
        """Test the simplest use-case"""

        Surface = namedtuple("Surface", ["vertices", "triangles"])
        vertices = np.array([[0, 0, 0], [0, 0.5, 0], [1, 0, 0]])
        triangles = np.array([[0, 1, 2]])
        surface = Surface(vertices, triangles)

        seeds = tg.seeds.from_surface(surface, 10)

        # The correct number of seeds.
        self.assertEqual(len(seeds), 10)

        # The normal is unit length.
        lengths = np.linalg.norm([s.orientation for s in seeds], axis=1)
        np.testing.assert_almost_equal(lengths, np.ones((10,)))

        # Test simple saving and loading to .tck.
        with tempfile.TemporaryDirectory() as d:
            tg.seeds.save(Path(d) / "cross-seeds-from-surface.txt", seeds)
            loaded = tg.seeds.load(Path(d) / "cross-seeds-from-surface.txt")

        for ell, s in zip(seeds, loaded):
            np.testing.assert_array_almost_equal(s.orientation, ell.orientation)
            np.testing.assert_array_almost_equal(s.location, ell.location)


class TestFromODF(unittest.TestCase):
    def test_simple(self):
        """Test the simplest use-case"""

        fod = test.data.cross()
        affine = np.eye(4)
        wm = fod[..., 0] > 0
        seeds = tg.seeds.from_mask(wm, affine, 1000)

        fod_data = tg.utils.normalize_odf(fod)
        seeds = tg.seeds.from_fod(fod_data, affine, 100000)
        self.assertEqual(len(seeds), 100000)

        # Test simple saving and loading to .tck.
        with tempfile.TemporaryDirectory() as d:
            tg.seeds.save(Path(d) / "cross-seeds-from-fod.tck", seeds)
            loaded = tg.seeds.load(Path(d) / "cross-seeds-from-fod.tck")

        for ell, s in zip(seeds, loaded):
            np.testing.assert_array_almost_equal(s.orientation, ell.orientation)
            np.testing.assert_array_almost_equal(s.location, ell.location)
