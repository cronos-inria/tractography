import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

import tractography as tg
import test

_TEST_RESULTS_DIR = (
    Path(__file__).parents[1] / "test-results" / "tractography" / "connectome"
)
_TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

class TestHistogram(unittest.TestCase):
    """Test the tg.histogram function"""

    def test_simple(self):
        """Test the simplest use case"""

        fod, seed_mask, _ = test.data.cross()
        fod = tg.nifti.multiply(fod, seed_mask)
        n_seeds = 10000
        histogram = tg.histogram(fod, seed_mask, n_seeds)
        self.assertLess(
            np.linalg.norm(fod.get_fdata() - histogram.get_fdata())
            / seed_mask.get_fdata().size,
            0.005,
        )


class TestTractography(unittest.TestCase):
    """Test the tg.tractogram function"""

    def test_simple(self):
        """Test the simplest use case"""

        fod, _, _ = test.data.cross()
        n_seeds = 100  # The number of seeds can be smaller than the batch size.
        seeds = tg.seeds.from_fod(fod.get_fdata(), fod.affine, n_seeds)
        streamlines = tg.tractogram(fod, seeds)
        self.assertLessEqual(len(streamlines), n_seeds)


class TestConnectome(unittest.TestCase):
    """Test the tg.connectome function"""

    def test_cross_deterministic(self):
        """Test that the connectivity matrix from the deterministic algorithm has the expected shape and labels"""

        fod, wm, segmentation = test.data.cross()
        nib.save(fod, _TEST_RESULTS_DIR / "cross-fod.nii.gz")
        nib.save(wm, _TEST_RESULTS_DIR / "cross-wm.nii.gz")
        nib.save(segmentation, _TEST_RESULTS_DIR / "cross-segmentation.nii.gz")
        fod = tg.nifti.multiply(fod, wm)
        n_seeds = 10000
        config = tg.configuration.load(tg.Algorithm.DETERMINISTIC)
        config.streamline.length.minimum = 5
        matrix, labels = tg.connectome(fod, segmentation, n_seeds=n_seeds, config=config)

        # The crossing dataset has labels 1, 2, 3, 4.
        np.testing.assert_array_equal(labels, [1, 2, 3, 4])

        # The matrix should be square with one row/column per label.
        self.assertEqual(matrix.shape, (4, 4))

        # The matrix should be symmetric (it is symmetrized in the public API).
        np.testing.assert_array_equal(matrix, matrix.T)

        # The diagonal should be zero or very small (streamlines connecting a
        # region to itself are uncommon in a crossing geometry).
        self.assertTrue(np.all(matrix.diagonal() < 0.1 * matrix.sum()))

        # The total number of connections should be positive, meaning some
        # streamlines did reach labeled regions.
        self.assertGreater(matrix.sum(), 0)

        # Region 1 should be connected mostly to 2. Region 3 should be connected mostly to 4.
        self.assertTrue(np.argmax(matrix[0]) == 1)
        self.assertTrue(np.argmax(matrix[2]) == 3)

    def test_cross_probabilistic(self):
        """Test that the connectivity matrix from the probabilistic algorithm has the expected shape and labels"""

        fod, wm, segmentation = test.data.cross()
        nib.save(fod, _TEST_RESULTS_DIR / "cross-fod.nii.gz")
        nib.save(wm, _TEST_RESULTS_DIR / "cross-wm.nii.gz")
        nib.save(segmentation, _TEST_RESULTS_DIR / "cross-segmentation.nii.gz")
        fod = tg.nifti.multiply(fod, wm)
        n_seeds = 10000
        config = tg.configuration.load(tg.Algorithm.PROBABILISTIC)
        config.streamline.length.minimum = 5
        matrix, labels = tg.connectome(fod, segmentation, n_seeds=n_seeds, config=config)

        # The crossing dataset has labels 1, 2, 3, 4.
        np.testing.assert_array_equal(labels, [1, 2, 3, 4])

        # The matrix should be square with one row/column per label.
        self.assertEqual(matrix.shape, (4, 4))

        # The matrix should be symmetric (it is symmetrized in the public API).
        np.testing.assert_array_equal(matrix, matrix.T)

        # The diagonal should be zero or very small (streamlines connecting a
        # region to itself are uncommon in a crossing geometry).
        self.assertTrue(np.all(matrix.diagonal() < 0.1 * matrix.sum()))

        # The total number of connections should be positive, meaning some
        # streamlines did reach labeled regions.
        self.assertGreater(matrix.sum(), 0)

        # Region 1 should be connected mostly to 2. Region 3 should be connected mostly to 4.
        self.assertTrue(np.argmax(matrix[0]) == 1)
        self.assertTrue(np.argmax(matrix[2]) == 3)
