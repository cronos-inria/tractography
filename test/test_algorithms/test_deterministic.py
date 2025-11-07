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
        fod = test.data.cross()
        fod = tg.utils.normalize_odf(fod)
        wm = fod[..., 0] > 0
        affine = np.eye(4)
        config = tg.configuration.load(tg.Algorithm.DETERMINISTIC)
        nib.save(
            nib.Nifti1Image(wm.astype(np.uint8), affine),
            _TEST_RESULTS_DIR / "histogram-cross-wm.nii.gz",
        )
        nib.save(
            nib.Nifti1Image(fod, affine),
            _TEST_RESULTS_DIR / "histogram-cross-fod.nii.gz",
        )

        histogram = tg.algorithms.deterministic.histogram(
            fod, affine, fod, affine, 10000, config
        )
        nib.save(
            nib.Nifti1Image(histogram, affine),
            _TEST_RESULTS_DIR / "histogram-cross-histogram.nii.gz",
        )

        # The histogram and the FOD should be very similar. The value of 0.5 is
        # arbitrary.
        self.assertTrue(np.linalg.norm(histogram - fod) / wm.size < 0.5)


class TestDeterministic(unittest.TestCase):
    """Test the OpenCL implementation of Deterministic tractography"""

    def setUp(self):
        _TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def test_uniform_isotropic(self):
        """Test deterministic tractography on a uniform isotropic fOD field"""

        # Prepare the data.
        fod = test.data.uniform_isotropic()
        affine = np.eye(4)
        seeds = tg.seeds.from_fod(fod, affine, 1000)
        nib.save(nib.Nifti1Image(fod, affine), _TEST_RESULTS_DIR / "uniform-fod.nii.gz")

        # Generate the tractogram.
        config = tg.configuration.load(tg.Algorithm.DETERMINISTIC)
        algorithm = tg.algorithms.Deterministic(fod, affine, len(seeds), config)
        streamlines = algorithm.run(seeds)

        # Save the streamlines for QA.
        tractogram = nib.streamlines.Tractogram(streamlines, affine_to_rasmm=np.eye(4))
        tck = nib.streamlines.TckFile(tractogram)
        tck.save(_TEST_RESULTS_DIR / "uniform-streamlines.tck")

    def test_cross(self):
        """Test tractography on the cross dataset"""

        # Prepare the data.
        fod = test.data.cross()
        affine = np.eye(4)
        nib.save(nib.Nifti1Image(fod, affine), _TEST_RESULTS_DIR / "cross-fod.nii.gz")
        wm = fod[..., 0] > 0
        nib.save(
            nib.Nifti1Image(wm.astype(np.uint8), affine),
            _TEST_RESULTS_DIR / "cross-wm.nii.gz",
        )
        seeds = tg.seeds.from_mask(wm, affine, 1000)

        # Generate the tractogram.
        config = tg.configuration.load(tg.Algorithm.DETERMINISTIC)
        algorithm = tg.algorithms.Deterministic(fod, affine, len(seeds), config)
        streamlines = algorithm.run(seeds)

        # Save the streamlines for QA.
        tractogram = nib.streamlines.Tractogram(streamlines, affine_to_rasmm=np.eye(4))
        tck = nib.streamlines.TckFile(tractogram)
        tck.save(_TEST_RESULTS_DIR / "cross-streamlines.tck")
