import unittest
from unittest.mock import MagicMock, patch, ANY
from pathlib import Path

import numpy as np
import nibabel as nib

from tractography.cli import connectome as cli_conn
import tractography as tg


class TestConnectome(unittest.TestCase):

    def setUp(self):
        """Setup common test paths and mocks."""
        self.image_path = Path("fake_fod.nii.gz")
        self.segmentation_path = Path("fake_seg.nii.gz")
        self.connectivity_path = Path("output_matrix.npy")
        self.mask_path = Path("fake_mask.nii.gz")

        # Default mock config
        self.mock_config = MagicMock()

    @patch("tractography.cli.connectome.np.save")
    @patch("tractography.cli.connectome.tg.connectome")
    @patch("tractography.cli.connectome.tg.nifti.multiply")
    @patch("tractography.cli.connectome.nib.load")
    @patch("tractography.cli.connectome.tg.configuration.load")
    @patch("tractography.cli.connectome.tg.cli.utils.set_tractography_config")
    def test_main_basic_flow(
        self,
        mock_set_config,
        mock_config_load,
        mock_nib_load,
        mock_multiply,
        mock_connectome,
        mock_np_save,
    ):
        """Test the main function with standard arguments (no mask)."""

        # --- Setup Returns ---
        mock_config_load.return_value = self.mock_config

        # Create distinct mock images for FOD and segmentation.
        mock_fod = MagicMock(spec=nib.Nifti1Image)
        mock_seg = MagicMock(spec=nib.Nifti1Image)

        # nib.load is called twice: once for FOD, once for the segmentation.
        mock_nib_load.side_effect = [mock_fod, mock_seg]

        # connectome returns (matrix, labels)
        expected_matrix = np.zeros((5, 5))
        mock_connectome.return_value = (expected_matrix, ["L1", "L2", "L3", "L4", "L5"])

        # --- Execute ---
        cli_conn.main(
            image_path=self.image_path,
            segmentation_path=self.segmentation_path,
            connectivity_path=self.connectivity_path,
            n_seeds=50000,
            algorithm=tg.Algorithm.DIFFUSION,
        )

        # --- Assertions ---

        # 1. Configuration loading.
        mock_config_load.assert_called_once_with(tg.Algorithm.DIFFUSION)

        # 2. File loading.
        # Check that nib.load was called for image and segmentation.
        mock_nib_load.assert_any_call(self.image_path)
        mock_nib_load.assert_any_call(self.segmentation_path)

        # 3. No masking should happen here.
        mock_multiply.assert_not_called()

        # 4. Connectome logic.
        mock_connectome.assert_called_once_with(
            mock_fod, mock_seg, 50000, self.mock_config
        )

        # 5. Saving.
        mock_np_save.assert_called_once_with(self.connectivity_path, expected_matrix)

    @patch("tractography.cli.connectome.np.save")
    @patch("tractography.cli.connectome.tg.connectome")
    @patch("tractography.cli.connectome.tg.nifti.multiply")
    @patch("tractography.cli.connectome.nib.load")
    @patch("tractography.cli.connectome.tg.configuration.load")
    @patch("tractography.cli.connectome.tg.cli.utils.set_tractography_config")
    def test_main_with_mask(
        self,
        mock_set_config,
        mock_config_load,
        mock_nib_load,
        mock_multiply,
        mock_connectome,
        mock_np_save,
    ):
        """Test the main function when an optional mask is provided."""

        # --- Setup Returns ---
        mock_fod = MagicMock(spec=nib.Nifti1Image)
        mock_mask = MagicMock(spec=nib.Nifti1Image)
        mock_masked_fod = MagicMock(spec=nib.Nifti1Image)  # Result of multiply
        mock_seg = MagicMock(spec=nib.Nifti1Image)

        # Sequence of loads: FOD -> Mask -> Segmentation
        mock_nib_load.side_effect = [mock_fod, mock_mask, mock_seg]

        mock_multiply.return_value = mock_masked_fod
        mock_connectome.return_value = (np.zeros((2, 2)), ["A", "B"])

        # --- Execute ---
        cli_conn.main(
            algorithm=tg.Algorithm.PROBABILISTIC,
            image_path=self.image_path,
            segmentation_path=self.segmentation_path,
            connectivity_path=self.connectivity_path,
            n_seeds=100,
            mask=self.mask_path  # <--- Providing mask here
        )

        # --- Assertions ---

        # 1. Check mask loading.
        mock_nib_load.assert_any_call(self.mask_path)

        # 2. Check mask application.
        mock_multiply.assert_called_once_with(mock_fod, mock_mask)

        # 3. Connectome should use the masked FOD, not the original.
        mock_connectome.assert_called_once_with(
            mock_masked_fod, mock_seg, 100, ANY
        )

    def test_parser_setup(self):
        """Sanity check that arguments are registered correctly."""
        mock_subparsers = MagicMock()
        mock_parser = MagicMock()
        mock_subparsers.add_parser.return_value = mock_parser

        cli_conn.add_parser(mock_subparsers)

        # Check required arguments exist
        calls = [args[0] for args, _ in mock_parser.add_argument.call_args_list]
        self.assertIn("image_path", calls)
        self.assertIn("segmentation_path", calls)
        self.assertIn("connectivity_path", calls)
        self.assertIn("--n_seeds", calls)
