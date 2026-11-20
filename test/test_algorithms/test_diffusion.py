import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

import tractography as tg
from tractography.opencl import OpenCLRuntime
import test.data.tensor
from test.test_algorithms import run_tractogram
from test.test_algorithms.exact_solutions import (
    u0u3_coefficients,
    u0u3_reference,
)


class TestExactSolutions(unittest.TestCase):
    def test_exact_zero_noise_solution(self):
        """With zero noise, diffusion follows the analytic cross-term drift."""
        epsilon = 0.3
        n_coefficients = 45
        coefficients = u0u3_coefficients(epsilon, n_coefficients)
        fod_data = np.broadcast_to(
            coefficients, (1, 1, 1, n_coefficients)
        ).copy()
        affine = np.diag((1000.0, 1000.0, 1000.0, 1.0))
        affine[:3, 3] = -0.5
        fod = nib.Nifti1Image(fod_data, affine)

        theta_0 = 0.12
        phi_0 = 0.5
        seed_position = np.asarray([10.0, 10.0, 10.0])
        orientation = np.asarray(tg.core.sph2cart(phi_0, theta_0, 1.0))
        seeds = [tg.seeds.Seed(seed_position, orientation)]

        step_size = 0.001
        maximum_length = 1.0
        algorithm = tg.algorithm.Diffusion(
            tg.local.RealSymmetricSphericalHarmonics(n_coefficients),
            step_size=step_size,
            gamma=1.0,
            noise_variance=0.0,
            seed=0,
        )
        streamlines = tg.tractogram(
            fod,
            seeds,
            algorithm=algorithm,
            save_at=step_size,
            maximum_streamline_length=maximum_length,
            minimum_streamline_length=0.0,
            batch_size=1,
        ).streamlines
        streamline = np.asarray(streamlines[0])

        reference_orientations = u0u3_reference(
            step_size,
            1.0,
            theta_0,
            phi_0,
            epsilon,
            len(streamline),
        )
        expected = np.empty_like(streamline)
        expected[0] = seed_position
        for index in range(1, len(streamline)):
            expected[index] = (
                expected[index - 1] + step_size * reference_orientations[index]
            )

        np.testing.assert_array_almost_equal(streamline, expected, decimal=3)


class TestTractogram(unittest.TestCase):
    """Test the OpenCL implementation of Diffusion tractography"""

    def test_uniform_isotropic(self):
        """Test diffusion tractography on a uniform isotropic fOD field"""

        # Prepare the data.
        fod = test.data.uniform_isotropic()
        affine = np.eye(4)
        nii = nib.nifti1.Nifti1Image(fod, affine)
        seeds = tg.seeds.from_fod(nii, 1000, runtime=OpenCLRuntime())

        # Generate the tractogram.
        tracto = run_tractogram(nii, seeds, tg.algorithm.Diffusion)
        streamlines = tracto.streamlines

        # Verify the number of streamline produced.
        self.assertEqual(len(streamlines), len(seeds))

    def test_cross_dti(self):
        """Test tractography on the DTI cross dataset"""

        # Prepare the data.
        tensor, wm, _ = test.data.tensor.cross()
        nii = nib.nifti1.Nifti1Image(tensor, tensor.affine)
        seeds = tg.seeds.from_mask(wm, 1000)

        # Generate the tractogram.
        tracto = run_tractogram(
            tensor,
            seeds,
            tg.algorithm.Diffusion,
            algorithm_options={"gamma": 5.0, "noise_variance": 0.05},
        )
        streamlines = tracto.streamlines

        # Verify the number of streamline produced.
        self.assertEqual(len(streamlines), len(seeds))

    def test_cross(self):
        """Test tractography on the cross dataset"""

        # Prepare the data.
        fod, wm, _ = test.data.cross()
        seeds = tg.seeds.from_mask(wm, 1000)

        # Generate the tractogram.
        tracto = run_tractogram(fod, seeds, tg.algorithm.Diffusion)
        streamlines = tracto.streamlines

        # Verify the number of streamline produced.
        self.assertEqual(len(streamlines), len(seeds))

    def test_circle_dti(self):
        """Test diffusion tractography on the circle dataset, with DTI data"""

        # Prepare the data.
        tensor, wm, _ = test.data.tensor.circle((20, 20, 1), radius=5, width=2)
        seeds = [tg.seeds.Seed([19.0, 22.5, 0.0], [-1.0, 0.0, 0.0])] * 10

        # Generate the tractogram.
        maximum_length = 40.0
        tracto = run_tractogram(
            tensor,
            seeds,
            tg.algorithm.Diffusion,
            algorithm_options={"gamma": 10.0, "noise_variance": 0.01},
            maximum_streamline_length=maximum_length,
        )
        streamlines = tracto.streamlines

        # The streamlines should run until the maximum length is reached.
        for streamline in streamlines:
            length = len(streamline) * 0.25
            self.assertAlmostEqual(length, maximum_length)

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
        maximum_length = 100.0
        tracto = run_tractogram(
            nib.Nifti1Image(fod, affine),
            seeds,
            tg.algorithm.Diffusion,
            algorithm_options={
                "step_size": 1e-3,
                "gamma": 50.0,
                "noise_variance": 0.1,
            },
            maximum_streamline_length=maximum_length,
        )
        streamlines = tracto.streamlines

        # The streamlines should run until the maximum lenght is reached.
        for streamline in streamlines:
            length = len(streamline) * 0.25
            self.assertAlmostEqual(length, maximum_length)