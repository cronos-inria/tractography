import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

import tractography as tg
import test.data


_TEST_RESULTS_DIR = (
    Path(__file__).parents[2] / "test-results" / "algorithms" / "probabilistic"
)


class TestProbabilistic(unittest.TestCase):
    """Test the OpenCL implementation of Probabilistic tractography"""

    def setUp(self):
        _TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def test_uniform_isotropic(self):
        """Test probabilistic tractography on a uniform isotropic fOD field"""

        # Prepare the data.
        fod = test.data.uniform_isotropic()
        affine = np.eye(4)
        nii = nib.nifti1.Nifti1Image(fod, affine)
        seeds = tg.seeds.from_fod(nii, 1000)    
        nib.save(nii, _TEST_RESULTS_DIR / "uniform-fod.nii.gz")

        # Generate the tractogram.
        config = tg.configuration.load(tg.Algorithm.PROBABILISTIC)
        tracto, _ = tg.algorithms.probabilistic.tractogram(
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
        config = tg.configuration.load(tg.Algorithm.PROBABILISTIC)
        tracto, _ = tg.algorithms.probabilistic.tractogram(fod, seeds, config)
        streamlines = tracto.streamlines

        # Save the streamlines for QA.
        tractogram = nib.streamlines.Tractogram(streamlines, affine_to_rasmm=np.eye(4))
        tck = nib.streamlines.TckFile(tractogram)
        tck.save(_TEST_RESULTS_DIR / "cross-streamlines.tck")
