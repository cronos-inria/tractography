import nibabel as nib
import numpy as np

import tractography as tg
from tractography.opencl import OpenCLRuntime
import test.data
from test.test_algorithms import run_tractogram
from test.test_algorithms.exact_solutions import u0u3_coefficients


class TestExactSolutions:
    def test_exact_zero_angle_limit(self):
        """A zero-radius sampling cap leaves a unit orientation unchanged."""
        algorithm = tg.algorithm.Probabilistic(
            tg.local.RealSymmetricSphericalHarmonics(45),
            max_angle=0.0,
            seed=0,
        )
        initial = np.asarray([tg.core.sph2cart(0.5, 0.12, 1.0)])
        coefficients = u0u3_coefficients()[np.newaxis, :]

        result = tg.algorithm.update_orientation(algorithm, coefficients, initial)

        np.testing.assert_allclose(result, initial, atol=1e-6)


class TestProbabilistic:
    """Test the OpenCL implementation of Probabilistic tractography"""

    def test_uniform_isotropic(self):
        """Test probabilistic tractography on a uniform isotropic fOD field"""

        # Prepare the data.
        fod = test.data.uniform_isotropic()
        affine = np.eye(4)
        nii = nib.nifti1.Nifti1Image(fod, affine)
        seeds = tg.seeds.from_fod(nii, 1000, runtime=OpenCLRuntime())
        # Generate the tractogram.
        run_tractogram(nii, seeds, tg.algorithm.Probabilistic)

    def test_cross(self):
        """Test tractography on the cross dataset"""

        # Prepare the data.
        fod, wm, _ = test.data.cross()
        seeds = tg.seeds.from_mask(wm, 1000)

        # Generate the tractogram.
        run_tractogram(fod, seeds, tg.algorithm.Probabilistic)
