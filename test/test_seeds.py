import unittest
from collections import namedtuple
from pathlib import Path
from time import time

import nibabel as nib
import numpy as np

import tractography as tg


_DATA_DIR = Path(__file__).parents[0] / "data"


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
        tg.seeds.save(_DATA_DIR / "cross-seeds-from-surface.txt", seeds)
        loaded = tg.seeds.load(_DATA_DIR / "cross-seeds-from-surface.txt")

        for ell, s in zip(seeds, loaded):
            np.testing.assert_array_almost_equal(s.orientation, ell.orientation)
            np.testing.assert_array_almost_equal(s.location, ell.location)


class TestFromODF(unittest.TestCase):
    def test_simple(self):
        """Test the simplest use-case"""

        fod = nib.load(_DATA_DIR / "cross-fod.nii.gz")
        wm_mask = nib.load(_DATA_DIR / "cross-wm.nii.gz")

        fod_data = tg.core.apply_mask(
            fod.get_fdata(), fod.affine, wm_mask.get_fdata(), wm_mask.affine
        )
        fod_data = tg.utils.normalize_odf(fod_data)

        seeds = tg.seeds.from_odf(fod_data, fod.affine, 100000)
        self.assertEqual(len(seeds), 100000)

        # Test simple saving and loading to .tck.
        tg.seeds.save(_DATA_DIR / "cross-seeds-from-odf.tck", seeds)
        loaded = tg.seeds.load(_DATA_DIR / "cross-seeds-from-odf.tck")

        for ell, s in zip(seeds, loaded):
            np.testing.assert_array_almost_equal(s.orientation, ell.orientation)
            np.testing.assert_array_almost_equal(s.location, ell.location)

