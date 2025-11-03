import tempfile
import unittest
from collections import namedtuple
from pathlib import Path

import nibabel as nib
import numpy as np

import tractography as tg
import test

_TEST_RESULTS_DIR = Path(__file__).parents[1] / "test-results" / "seeds"


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

        seeds = tg.seeds.from_surface(surface, 10, cone_angle=25)

        # The correct number of seeds.
        self.assertEqual(len(seeds), 10)

        # The normal is unit length.
        lengths = np.linalg.norm([s.orientation for s in seeds], axis=1)
        np.testing.assert_almost_equal(lengths, np.ones((10,)))

        # The angle is respected.
        for seed in seeds[1:]:
            angle = np.rad2deg(np.acos(np.dot(seeds[0].orientation, seed.orientation)))
            self.assertTrue(angle <= 50)


class TestFromODF(unittest.TestCase):
    def test_simple(self):
        """Test the simplest use-case"""

        fod = test.data.cross()
        affine = np.eye(4)
        wm = fod[..., 0] > 0
        seeds = tg.seeds.from_mask(wm, affine, 1000)

        fod_data = tg.utils.normalize_odf(fod)
        seeds = tg.seeds.from_fod(fod_data, affine, 10000, use_opencl=False)
        self.assertEqual(len(seeds), 10000)

        # Test simple saving and loading to .tck.
        with tempfile.TemporaryDirectory() as d:
            tg.seeds.save(Path(d) / "cross-seeds-from-fod.tck", seeds)
            loaded = tg.seeds.load(Path(d) / "cross-seeds-from-fod.tck")

        for ell, s in zip(seeds, loaded):
            np.testing.assert_array_almost_equal(s.orientation, ell.orientation)
            np.testing.assert_array_almost_equal(s.location, ell.location)

        # Returning as an array should change nothing.
        seeds = tg.seeds.from_fod(
            fod_data, affine, 10000, as_array=True, use_opencl=False
        )
        self.assertEqual(seeds.shape, (10000, 8))

        # Saving and loading in an unknow filetype raises and error.
        self.assertRaises(
            ValueError, lambda: tg.seeds.save(Path(d) / "oups.tyk", seeds)
        )
        self.assertRaises(ValueError, lambda: tg.seeds.load(Path(d) / "oups.tyk"))


class TestOpenCLFromFOD(unittest.TestCase):
    """Test generating seeds on using the OpenCL implementation"""

    def setUp(self):
        _TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def test_simple(self):
        """Test the simple case of generating seeds using OpenCL"""

        # Generate fake data.
        n_seeds = 100000
        fod = test.data.cross()
        mask = fod[..., 0] > 0
        affine = np.eye(4)
        seeds = tg.seeds.from_fod(fod, affine, n_seeds, as_array=True, use_opencl=True)

        # All points must be in the grid.
        self.assertTrue(np.all(seeds[:, :3] > -0.5))
        self.assertTrue(np.all(seeds[:, :3] < 9.5))
        self.assertTrue(np.allclose(np.linalg.norm(seeds[:, 4:], axis=1), 1))

        # Save the seeds for QA.
        nib.save(
            nib.Nifti1Image(fod, affine), _TEST_RESULTS_DIR / "gpu-simple-fod.nii.gz"
        )
        nib.save(
            nib.Nifti1Image(mask.astype(np.uint8), affine),
            _TEST_RESULTS_DIR / "gpu-simple-wm.nii.gz",
        )
        tg.seeds.save(
            _TEST_RESULTS_DIR / "gpu-simple-seeds.tck", tg.seeds.from_array(seeds)
        )
