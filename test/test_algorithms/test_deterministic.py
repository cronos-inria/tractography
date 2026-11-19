import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

import tractography as tg
import test
from test.test_algorithms.exact_solutions import u0u3_coefficients


_TEST_RESULTS_DIR = (
    Path(__file__).parents[2] / "test-results" / "algorithms" / "deterministic"
)


class TestExactSolutions(unittest.TestCase):
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


class TestDeterministic(unittest.TestCase):
    """Test the OpenCL implementation of Deterministic tractography"""

    def setUp(self):
        _TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def test_uniform_isotropic(self):
        """Test deterministic tractography on a uniform isotropic fOD field"""

        # Prepare the data.
        fod = test.data.uniform_isotropic()
        affine = np.eye(4)
        nii = nib.nifti1.Nifti1Image(fod, affine)
        seeds = tg.seeds.from_fod(nii, 1000)
        nib.save(nii, _TEST_RESULTS_DIR / "uniform-fod.nii.gz")

        # Generate the tractogram.
        config = tg.configuration.load(tg.Algorithm.DETERMINISTIC)
        tracto, _ = tg.algorithms.deterministic.tractogram(
            nii, seeds, config
        )
        streamlines = tracto.streamlines

        # Save the streamlines for QA.
        tractogram = nib.streamlines.Tractogram(streamlines, affine_to_rasmm=np.eye(4))
        tck = nib.streamlines.TckFile(tractogram)
        tck.save(_TEST_RESULTS_DIR / "uniform-streamlines.tck")

    def test_cross(self):
        """Test tractography on the cross dataset"""

        # Prepare the data.
        fod, wm, _ = test.data.cross()
        nib.save(fod, _TEST_RESULTS_DIR / "cross-fod.nii.gz")
        nib.save(wm, _TEST_RESULTS_DIR / "cross-wm.nii.gz")
        seeds = tg.seeds.from_mask(wm, 1000)

        # Generate the tractogram.
        config = tg.configuration.load(tg.Algorithm.DETERMINISTIC)
        tracto, _ = tg.algorithms.deterministic.tractogram(fod, seeds, config)
        streamlines = tracto.streamlines

        # Save the streamlines for QA.
        tractogram = nib.streamlines.Tractogram(streamlines, affine_to_rasmm=np.eye(4))
        tck = nib.streamlines.TckFile(tractogram)
        tck.save(_TEST_RESULTS_DIR / "cross-streamlines.tck")
