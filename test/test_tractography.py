import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

import tractography as tg
import test


class TestHistogram(unittest.TestCase):
    """Test the tg.histogram function"""

    def test_simple(self):
        """Test the simplest use case"""

        fod = nib.Nifti1Image(test.data.cross(), np.eye(4))
        seed_mask_data = fod.get_fdata()[..., 0] > 0
        seed_mask = nib.Nifti1Image(seed_mask_data.astype(np.uint8), np.eye(4))
        n_seeds = 10000
        histogram = tg.histogram(fod, seed_mask, n_seeds)
        self.assertLess(
            np.linalg.norm(fod.get_fdata() - histogram.get_fdata())
            / seed_mask_data.size,
            0.005,
        )
