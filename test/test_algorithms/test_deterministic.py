import nibabel as nib
import numpy as np

import tractography as tg
from tractography.opencl import OpenCLRuntime
import test
from test.test_algorithms import run_tractogram
from test.test_algorithms.exact_solutions import u0u3_coefficients


class TestExactSolutions:
    def test_exact_cross_term_maximum(self):
        """The OpenCL update converges to the analytic maximum of 1 + eps*x*z."""
        algorithm = tg.algorithm.Deterministic(
            tg.local.RealSymmetricSphericalHarmonics(45),
            max_iterations=30,
            tolerance=1e-6,
            max_step_size=0.5,
        )
        initial = np.asarray([tg.core.sph2cart(0.5, 0.12, 1.0)])
        coefficients = u0u3_coefficients()[np.newaxis, :]

        result = tg.algorithm.update_orientation(algorithm, coefficients, initial)

        expected = np.asarray([1.0, 0.0, 1.0]) / np.sqrt(2.0)
        np.testing.assert_allclose(result[0], expected, atol=1e-3)


class TestDeterministic:
    """Test the OpenCL implementation of Deterministic tractography"""

    def test_uniform_isotropic(self):
        """Test deterministic tractography on a uniform isotropic fOD field"""

        # Prepare the data.
        fod = test.data.uniform_isotropic()
        affine = np.eye(4)
        nii = nib.nifti1.Nifti1Image(fod, affine)
        seeds = tg.seeds.from_fod(nii, 1000, runtime=OpenCLRuntime())
        # Generate the tractogram.
        run_tractogram(nii, seeds, tg.algorithm.Deterministic)

    def test_cross(self):
        """Test tractography on the cross dataset"""

        # Prepare the data.
        fod, wm, _ = test.data.cross()
        seeds = tg.seeds.from_mask(wm, 1000)

        # Generate the tractogram.
        run_tractogram(fod, seeds, tg.algorithm.Deterministic)
