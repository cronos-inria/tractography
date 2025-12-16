import unittest
from unittest.mock import MagicMock, patch

import nibabel as nib
import numpy as np

import tractography as tg
import test


class TestConnectome(unittest.TestCase):

    def setUp(self):
        """Create dummy input data for testing."""
        self.shape = (10, 10, 10)
        self.affine = np.eye(4)

        # Create dummy FOD and Segmentation images
        self.fod = nib.Nifti1Image(np.zeros(self.shape + (45,)), self.affine)
        self.segmentation = nib.Nifti1Image(np.zeros(self.shape), self.affine)

    @patch("tractography.connectivity")
    @patch("tractography.tractogram")
    @patch("tractography.seeds")
    @patch("tractography.nifti")
    def test_connectome_workflow(
        self, mock_nifti, mock_seeds, mock_tractogram, mock_connectivity
    ):
        """Test the full flow of the connectome function using mocks."""

        # --- Setup Mocks ---

        # 1. Mock nifti.threshold
        mock_mask = MagicMock(spec=nib.Nifti1Image)
        mock_nifti.threshold.return_value = mock_mask

        # 2. Mock nifti.multiply
        # It must return an object with .get_fdata() and .affine for the next step.
        mock_seed_fod = MagicMock(spec=nib.Nifti1Image)
        mock_seed_fod.get_fdata.return_value = np.zeros((10, 10, 10, 45))
        mock_seed_fod.affine = self.affine
        mock_nifti.multiply.return_value = mock_seed_fod

        # 3. Mock seeds.from_fod
        mock_seeds_list = ["seed1", "seed2"]
        mock_seeds.from_fod.return_value = mock_seeds_list

        # 4. Mock tractogram
        # CRITICAL: Must return an object with a .streamlines attribute.
        mock_tg_object = MagicMock()
        mock_tg_object.streamlines = ["streamline1", "streamline2"]
        mock_tractogram.return_value = mock_tg_object

        # 5. Mock connectivity functions
        mock_connectivity.convert_segmentation.return_value = ("vertices", ["RegionA", "RegionB"])
        mock_connectivity.map_vertices.return_value = "mapping_data"
        mock_connectivity.symmetrize_mapping.return_value = "symmetric_mapping"

        expected_labels = ["RegionA", "RegionB"]
        expected_matrix = np.array([[0, 1], [1, 0]])
        mock_connectivity.compile_connectivity_matrix.return_value = (expected_matrix, expected_labels)

        # --- Run Function ---

        matrix, labels = tg.connectome(
            self.fod, self.segmentation, n_seeds=500, config=None
        )

        # --- Assertions ---

        # Verify mask creation logic
        mock_nifti.threshold.assert_called_once_with(self.segmentation, 0.1)

        # Verify multiply called with order=0 (Critical fix check)
        mock_nifti.multiply.assert_called_once_with(
            self.fod, mock_mask, order=0
        )

        # Verify seeds generation
        mock_seeds.from_fod.assert_called_once()
        args, _ = mock_seeds.from_fod.call_args
        # Check that the data passed to seeds.from_fod came from our mocked seed_fod
        np.testing.assert_array_equal(args[0], mock_seed_fod.get_fdata.return_value)

        # Verify tractogram execution
        mock_tractogram.assert_called_once_with(
            self.fod, mock_seeds_list, None, endpoints_only=True
        )

        # Verify connectivity mapping flow
        # Ensure it passed the .streamlines list, not the whole tractogram object
        mock_connectivity.map_vertices.assert_called_once_with(
            "vertices", mock_tg_object.streamlines, ["RegionA", "RegionB"]
        )

        # Verify return values
        np.testing.assert_array_equal(matrix, expected_matrix)
        self.assertEqual(labels, ["RegionA", "RegionB"])


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
