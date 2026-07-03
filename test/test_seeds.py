import tempfile
import unittest
from collections import namedtuple
from pathlib import Path

import nibabel as nib
import numpy as np

import tractography as tg
import test
import test.data.tensor

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

        fod, wm, _ = test.data.cross()
        seeds = tg.seeds.from_mask(wm, 1000)

        seeds = tg.seeds.from_fod(fod, 10000, use_opencl=False)
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
            fod, 10000, as_array=True, use_opencl=False
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
        fod, mask, _ = test.data.cross()
        fod = tg.nifti.multiply(fod, mask)
        seeds = tg.seeds.from_fod(
            fod,
            n_seeds,
            as_array=True,
            use_opencl=True,
        )

        # All points must be in the grid.
        self.assertTrue(np.all(seeds[:, :3] >= -0.5))
        self.assertTrue(np.all(seeds[:, :3] <= 9.5))
        self.assertTrue(np.allclose(np.linalg.norm(seeds[:, 4:], axis=1), 1))

        # Save the seeds for QA.
        nib.save(fod, _TEST_RESULTS_DIR / "gpu-simple-fod.nii.gz")
        nib.save(mask, _TEST_RESULTS_DIR / "gpu-simple-wm.nii.gz")
        tg.seeds.save(
            _TEST_RESULTS_DIR / "gpu-simple-seeds.tck", tg.seeds.from_array(seeds)
        )

    @staticmethod
    def _orientation_second_moment(orientations, weights=None):
        if weights is None:
            weights = np.full(len(orientations), 1.0 / len(orientations))
        return np.einsum("n,ni,nj->ij", weights, orientations, orientations)

    @staticmethod
    def _sh_probabilities(coefficients, directions):
        azimuths, colatitudes, _ = tg.core.cart2sph(*directions.T)
        ishtmtx, _ = tg.core.ishtmtx(azimuths, colatitudes, 45)
        values = np.dot(ishtmtx, coefficients)

        scaled = 100.0 * values
        weights = np.where(
            scaled < 30.0,
            np.log1p(np.exp(np.clip(scaled, -700.0, 30.0))) / 100.0,
            values,
        )
        probabilities = np.maximum(weights, 0.0)
        return probabilities / np.sum(probabilities)

    @staticmethod
    def _dti_probabilities(coefficients, directions):
        tensor = np.array(
            [
                [coefficients[0], coefficients[3], coefficients[4]],
                [coefficients[3], coefficients[1], coefficients[5]],
                [coefficients[4], coefficients[5], coefficients[2]],
            ]
        )
        values = np.einsum("ni,ij,nj->n", directions, tensor, directions)
        probabilities = np.maximum(values, 0.0)
        return probabilities / np.sum(probabilities)

    def test_orientation_distribution_matches_sh_fod(self):
        n_seeds = 60000
        directions = tg.core.fibonacci_sphere(1000)

        sh_fod, _, _ = test.data.cross()
        coefficients = sh_fod.get_fdata()[5, 1, 0]
        fod = np.zeros((1, 1, 1, 45), dtype=np.float32)
        fod[0, 0, 0] = coefficients
        nii = nib.nifti1.Nifti1Image(fod, sh_fod.affine)

        seeds = tg.seeds.from_fod(nii, n_seeds, as_array=True, use_opencl=True)
        orientations = seeds[:, 4:7]

        expected = self._orientation_second_moment(
            directions, self._sh_probabilities(coefficients, directions)
        )
        empirical = self._orientation_second_moment(orientations)

        np.testing.assert_allclose(empirical, expected, atol=2e-2)

    def test_orientation_distribution_matches_dti_fod(self):
        n_seeds = 60000
        directions = tg.core.fibonacci_sphere(1000)

        dti_fod, _, _ = test.data.tensor.cross()
        coefficients = dti_fod.get_fdata()[10, 1, 0]
        fod = np.zeros((1, 1, 1, 6), dtype=np.float32)
        fod[0, 0, 0] = coefficients
        nii = nib.nifti1.Nifti1Image(fod, dti_fod.affine)

        seeds = tg.seeds.from_fod(nii, n_seeds, as_array=True, use_opencl=True)
        orientations = seeds[:, 4:7]

        expected = self._orientation_second_moment(
            directions, self._dti_probabilities(coefficients, directions)
        )
        empirical = self._orientation_second_moment(orientations)

        np.testing.assert_allclose(empirical, expected, atol=2e-2)
