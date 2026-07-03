import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

import tractography as tg
import test


_OPENCL_DIR = Path(__file__).parents[2] / "src"
_TEST_RESULTS_DIR = (
    Path(__file__).parents[2] / "test-results" / "algorithms" / "deterministic"
)


class TestDeterministicHistogram(unittest.TestCase):

    def setUp(self):
        _TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def test_cross(self):
        """Test deterministic tractography on the cross dataset"""

        fod, wm, _ = test.data.cross()
        config = tg.configuration.load(tg.Algorithm.DETERMINISTIC)

        nib.save(wm, _TEST_RESULTS_DIR / "histogram-cross-wm.nii.gz")
        nib.save(fod, _TEST_RESULTS_DIR / "histogram-cross-fod.nii.gz")

        histogram = tg.algorithms.deterministic.histogram(
            fod.get_fdata(), fod.affine, fod.get_fdata(), fod.affine, 10000, config
        )
        nib.save(
            nib.Nifti1Image(histogram, fod.affine),
            _TEST_RESULTS_DIR / "histogram-cross-histogram.nii.gz",
        )

        # The histogram and the FOD should be very similar. The value of 0.5 is
        # arbitrary.
        self.assertTrue(
            np.linalg.norm(histogram - fod.get_fdata())
            / wm.get_fdata().size < 0.5
        )


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
