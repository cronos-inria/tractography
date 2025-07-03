import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

import tractography as tg
import test.test_algorithms


_OPENCL_DIR = Path(__file__).parents[2] / "src"
_DATA_DIR = Path(__file__).parents[1] / "data"


class TestDiffusion(unittest.TestCase):
    """Test the OpenCL implementation of Diffusion tractography"""

    def setUp(self):
        test.test_algorithms.generate_cross_test_data()

    def test_cross_tractography(self):
        """Test tractography on the cross dataset"""

        fod = nib.load(_DATA_DIR / "cross-fod.nii.gz")
        wm_mask = nib.load(_DATA_DIR / "cross-wm.nii.gz")
        seed_mask = nib.load(_DATA_DIR / "cross-seed.nii.gz")
        seeds = tg.seeds.from_mask(seed_mask.get_fdata(), seed_mask.affine, 1000)

        fod_data = tg.core.apply_mask(
            fod.get_fdata(), fod.affine, wm_mask.get_fdata(), wm_mask.affine
        )
        fod_data = tg.utils.normalize_odf(fod_data)

        config = tg.configuration.load()
        config.algorithms.diffusion.save_at = 0.1
        config.streamline.length.minimum = 1
        config.streamline.length.maximum = 20
        algorithm = tg.algorithms.Diffusion(fod_data, fod.affine, len(seeds), config)
        streamlines = algorithm.run(seeds)

        tractogram = nib.streamlines.Tractogram(streamlines, affine_to_rasmm=np.eye(4))
        tck = nib.streamlines.TckFile(tractogram)
        tck.save(_DATA_DIR / "cross-streamlines-diffusion.tck")
