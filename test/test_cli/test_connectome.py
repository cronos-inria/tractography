from unittest.mock import ANY, MagicMock, call
from pathlib import Path

import numpy as np
import pytest

from tractography.cli import connectome as cli_conn
import tractography as tg
import tractography.cli
import tractography.cli.utils as cli_utils


@pytest.fixture
def test_paths():
    return {
        "image_path": Path("fake_fod.nii.gz"),
        "segmentation_path": Path("fake_seg.nii.gz"),
        "connectome_path": Path("output_matrix.npz"),
        "mask_path": Path("fake_mask.nii.gz"),
    }


def test_main_basic_flow(monkeypatch, test_paths):
    """Test the main function with standard arguments (no mask)."""

    mock_set_config = MagicMock()
    mock_config_load = MagicMock()
    mock_nib_load = MagicMock()
    mock_multiply = MagicMock()
    mock_connectome = MagicMock()
    mock_np_savez = MagicMock()

    monkeypatch.setattr(cli_utils, "set_tractography_config", mock_set_config)
    monkeypatch.setattr(cli_conn.tg.configuration, "load", mock_config_load)
    monkeypatch.setattr(cli_conn.nib, "load", mock_nib_load)
    monkeypatch.setattr(cli_conn.tg.nifti, "multiply", mock_multiply)
    monkeypatch.setattr(cli_conn.tg, "connectome", mock_connectome)
    monkeypatch.setattr(cli_conn.np, "savez", mock_np_savez)

    mock_config = MagicMock()
    mock_config_load.return_value = mock_config

    mock_fod = MagicMock()
    mock_seg = MagicMock()
    mock_nib_load.side_effect = [mock_fod, mock_seg]

    expected_matrix = np.zeros((5, 5))
    expected_labels = ["L1", "L2", "L3", "L4", "L5"]
    mock_connectome.return_value = (expected_matrix, expected_labels)

    cli_conn.main(
        image_path=test_paths["image_path"],
        segmentation_path=test_paths["segmentation_path"],
        connectome_path=test_paths["connectome_path"],
        n_seeds=50000,
        algorithm=tg.Algorithm.DIFFUSION,
        batch_size=4096,
    )

    mock_config_load.assert_called_once_with(tg.Algorithm.DIFFUSION)
    mock_set_config.assert_called_once_with(mock_config, {"batch_size": 4096})
    mock_nib_load.assert_has_calls([
        call(test_paths["image_path"]),
        call(test_paths["segmentation_path"]),
    ])
    mock_multiply.assert_not_called()
    mock_connectome.assert_called_once_with(mock_fod, mock_seg, 50000, mock_config)
    mock_np_savez.assert_called_once_with(
        test_paths["connectome_path"],
        matrix=expected_matrix,
        labels=expected_labels,
    )


def test_main_with_mask(monkeypatch, test_paths):
    """Test the main function when an optional mask is provided."""

    mock_set_config = MagicMock()
    mock_config_load = MagicMock()
    mock_nib_load = MagicMock()
    mock_multiply = MagicMock()
    mock_connectome = MagicMock()
    mock_np_savez = MagicMock()

    monkeypatch.setattr(cli_utils, "set_tractography_config", mock_set_config)
    monkeypatch.setattr(cli_conn.tg.configuration, "load", mock_config_load)
    monkeypatch.setattr(cli_conn.nib, "load", mock_nib_load)
    monkeypatch.setattr(cli_conn.tg.nifti, "multiply", mock_multiply)
    monkeypatch.setattr(cli_conn.tg, "connectome", mock_connectome)
    monkeypatch.setattr(cli_conn.np, "savez", mock_np_savez)

    mock_config_load.return_value = MagicMock()

    mock_fod = MagicMock()
    mock_mask = MagicMock()
    mock_masked_fod = MagicMock()
    mock_seg = MagicMock()
    mock_nib_load.side_effect = [mock_fod, mock_mask, mock_seg]

    mock_multiply.return_value = mock_masked_fod
    mock_connectome.return_value = (np.zeros((2, 2)), ["A", "B"])

    cli_conn.main(
        algorithm=tg.Algorithm.PROBABILISTIC,
        image_path=test_paths["image_path"],
        segmentation_path=test_paths["segmentation_path"],
        connectome_path=test_paths["connectome_path"],
        n_seeds=100,
        mask=test_paths["mask_path"],
    )

    mock_nib_load.assert_has_calls([
        call(test_paths["image_path"]),
        call(test_paths["mask_path"]),
        call(test_paths["segmentation_path"]),
    ])
    mock_multiply.assert_called_once_with(mock_fod, mock_mask)
    mock_connectome.assert_called_once_with(mock_masked_fod, mock_seg, 100, ANY)
    mock_np_savez.assert_called_once_with(
        test_paths["connectome_path"],
        matrix=ANY,
        labels=ANY,
    )


def test_cli_parse_and_dispatch(monkeypatch, test_paths):
    """Ensure the connectome subcommand parses and dispatches correctly."""

    mock_connectome_main = MagicMock()
    monkeypatch.setattr(cli_conn, "main", mock_connectome_main)

    args = tractography.cli.parse_arguments([
        "connectome",
        str(test_paths["image_path"]),
        str(test_paths["segmentation_path"]),
        str(test_paths["connectome_path"]),
        "--mask",
        str(test_paths["mask_path"]),
        "--number-of-seeds",
        "1234",
        "--algorithm",
        "transport",
    ])

    tractography.cli.main(args)

    mock_connectome_main.assert_called_once_with(
        image_path=test_paths["image_path"],
        segmentation_path=test_paths["segmentation_path"],
        connectome_path=test_paths["connectome_path"],
        mask=test_paths["mask_path"],
        n_seeds=1234,
        algorithm=tg.Algorithm.TRANSPORT,
        batch_size=None,
        step_size=None,
        min_length=None,
        max_length=None,
    )
