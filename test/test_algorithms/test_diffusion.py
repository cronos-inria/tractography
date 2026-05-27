import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

import tractography as tg
import test.data.tensor


class TestDiffusionHistogram(unittest.TestCase):

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
        weights = np.maximum(values, 0.0)
        return weights / np.sum(weights)

    @staticmethod
    def _sh_probabilities(coefficients, directions):
        azimuths, colatitudes, _ = tg.core.cart2sph(*directions.T)
        ishtmtx, _ = tg.core.ishtmtx(azimuths, colatitudes, 45)
        values = np.dot(ishtmtx, coefficients)
        weights = np.maximum(values, 0.0)
        return weights / np.sum(weights)

    @staticmethod
    def _orientation_second_moment(directions, probabilities):
        return np.einsum("n,ni,nj->ij", probabilities, directions, directions)

    def test_cross(self):

        fod, wm, _ = test.data.cross()
        fod = tg.nifti.multiply(fod, wm)
        config = tg.configuration.load(tg.Algorithm.DIFFUSION)
        config.noise_variance = 2.0
        
        histogram = tg.algorithms.diffusion.histogram(
            fod.get_fdata(), fod.affine, fod.get_fdata(), fod.affine, 10000, config
        )

        # For a noise variance of 2.0, the histogram and the FOD should match.
        self.assertTrue(
            np.linalg.norm(histogram - fod.get_fdata())
            / wm.get_fdata().size < 0.005
        )

    def test_cross_dti(self):
        """Test histogram generation using DTI model on the cross dataset"""

        tensor, wm, _ = test.data.tensor.cross()
        tensor = tg.nifti.multiply(tensor, wm)
        config = tg.configuration.load(tg.Algorithm.DIFFUSION)

        histogram = tg.algorithms.diffusion.histogram(
            tensor.get_fdata(),
            tensor.affine,
            tensor.get_fdata(),
            tensor.affine,
            1000,
            config,
        )

        # Histogram should have same shape as input tensor and be finite.
        self.assertEqual(histogram.shape[:3], tensor.get_fdata().shape[:3])
        self.assertTrue(np.isfinite(histogram).all())

        # Compare local directional distributions in a common basis.
        directions = tg.core.fibonacci_sphere(1000)
        voxel = (5, 1, 0)
        dti_coefficients = tensor.get_fdata()[voxel]
        hist_coefficients = histogram[voxel]

        expected = self._orientation_second_moment(
            directions, self._dti_probabilities(dti_coefficients, directions)
        )
        observed = self._orientation_second_moment(
            directions, self._sh_probabilities(hist_coefficients, directions)
        )

        np.testing.assert_allclose(observed, expected, atol=3e-2)


class TestTractogram(unittest.TestCase):
    """Test the OpenCL implementation of Diffusion tractography"""

    def test_uniform_isotropic(self):
        """Test diffusion tractography on a uniform isotropic fOD field"""

        # Prepare the data.
        fod = test.data.uniform_isotropic()
        affine = np.eye(4)
        seeds = tg.seeds.from_fod(fod, affine, 1000)

        # Generate the tractogram.
        config = tg.configuration.load(tg.Algorithm.DIFFUSION)
        tracto, _ = tg.algorithms.diffusion.tractogram(
            nib.Nifti1Image(fod, affine), seeds, config
        )
        streamlines = tracto.streamlines

        # Verify the number of streamline produced.
        self.assertEqual(len(streamlines), len(seeds))

    def test_cross_dti(self):
        """Test tractography on the DTI cross dataset"""

        # Prepare the data.
        fod, wm, _ = test.data.tensor.cross()
        seeds = tg.seeds.from_mask(wm.get_fdata(), wm.affine, 1000)

        # Generate the tractogram.
        config = tg.configuration.load(tg.Algorithm.DIFFUSION)
        config.inverse_curvature = 5.0
        config.noise_variance = 0.05
        tracto, _ = tg.algorithms.diffusion.tractogram(fod, seeds, config)
        streamlines = tracto.streamlines

        # Verify the number of streamline produced.
        self.assertEqual(len(streamlines), len(seeds))

    def test_cross(self):
        """Test tractography on the cross dataset"""

        # Prepare the data.
        fod, wm, _ = test.data.cross()
        seeds = tg.seeds.from_mask(wm.get_fdata(), wm.affine, 1000)

        # Generate the tractogram.
        config = tg.configuration.load(tg.Algorithm.DIFFUSION)
        tracto, _ = tg.algorithms.diffusion.tractogram(fod, seeds, config)
        streamlines = tracto.streamlines

        # Verify the number of streamline produced.
        self.assertEqual(len(streamlines), len(seeds))

    def test_circle_dti(self):
        """Test diffusion tractography on the circle dataset, with DTI data"""

        # Prepare the data.
        tensor, wm, _ = test.data.tensor.circle((20, 20, 1), radius=5, width=2)
        seeds = [tg.seeds.Seed([19.0, 22.5, 0.0], [-1.0, 0.0, 0.0])] * 10

        # Generate the tractogram.
        config = tg.configuration.load(tg.Algorithm.DIFFUSION)
        config.inverse_curvature = 10.0
        config.noise_variance = 0.01
        config.streamline.length.maximum = 40.0
        tracto, _ = tg.algorithms.diffusion.tractogram(tensor, seeds, config)
        streamlines = tracto.streamlines

        # The streamlines should run until the maximum length is reached.
        for streamline in streamlines:
            length = len(streamline) * config.save_at
            self.assertAlmostEqual(length, config.streamline.length.maximum)

    def test_circle(self):
        """Test tractography on the circle dataset"""

        # Prepare the data.
        shape = (10, 10, 1)
        radius = 2
        fod = test.data.circle(shape=shape, radius=radius)
        affine = np.eye(4)
        wm = fod[..., 0] > 0
        seeds = [
            tg.seeds.Seed(
                [(shape[0] - 1) / 2, (shape[1] - 1) / 2 + radius, 0.0], [1.0, 0.0, 0.0]
            )
            for _ in range(100)
        ]

        # Generate the tractogram. We set a few specific parameters due to the
        # small size of the circle.
        config = tg.configuration.load(tg.Algorithm.DIFFUSION)
        config.streamline.length.maximum = 100
        config.step_size = 1e-3
        config.inverse_curvature = 50.0
        config.noise_variance = 0.1
        tracto, _ = tg.algorithms.diffusion.tractogram(
            nib.Nifti1Image(fod, affine), seeds, config
        )
        streamlines = tracto.streamlines

        # The streamlines should run until the maximum lenght is reached.
        for streamline in streamlines:
            length = len(streamline) * config.save_at
            self.assertAlmostEqual(length, config.streamline.length.maximum)