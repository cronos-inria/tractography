import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

import tractography as tg
import tractography.cli.tractogram
import test


_TEST_RESULTS_DIR = Path(__file__).parents[2] / "test-results" / "cli" / "tractogram"


class TestTractogram(unittest.TestCase):
    """Test the generation of tractograms from the CLI"""

    def setUp(self):
        _TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def test_cross_tractography(self):
        """Test tractography on the cross dataset"""

        # Prepare the data.
        fod = test.data.cross()
        affine = np.eye(4)
        fod_path = _TEST_RESULTS_DIR / "cross-fod.nii.gz"
        nib.save(nib.Nifti1Image(fod, affine), fod_path)

        wm = fod[..., 0] > 0
        wm_path = _TEST_RESULTS_DIR / "cross-wm.nii.gz"
        nib.save(nib.Nifti1Image(wm.astype(np.uint8), affine), wm_path)

        seeds = tg.seeds.from_mask(wm, affine, 1000)
        seeds_path = _TEST_RESULTS_DIR / "cross-seeds.txt"
        tg.seeds.save(seeds_path, seeds)

        tractogram_path = _TEST_RESULTS_DIR / "cross-streamlines.tck"
        tg.cli.tractogram.main(
            fod_path, seeds_path, tractogram_path,
            batch_size=1000,
        )
