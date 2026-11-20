from pathlib import Path
from unittest.mock import MagicMock

from nibabel.nifti1 import Nifti1Image
import numpy as np
from scipy.sparse import csr_matrix

from tractography.cli import connectome as cli_connectome
import tractography as tg
import tractography.cli


def test_main_loads_images_delegates_and_saves_labels(monkeypatch, tmp_path):
    field_path = tmp_path / "field.nii.gz"
    domain_path = tmp_path / "domain.nii.gz"
    segmentation_path = tmp_path / "regions.nii.gz"
    output_path = tmp_path / "connectome.npz"
    field = Nifti1Image(np.ones((2, 2, 2, 45), dtype=np.float32), np.eye(4))
    domain = Nifti1Image(np.ones((2, 2, 2), dtype=np.uint8), np.eye(4))
    segmentation = Nifti1Image(np.ones((2, 2, 2), dtype=np.uint8), np.eye(4))
    matrix = np.array([[0.0, 3.0], [1.0, 0.0]], dtype=np.float32)
    labels = np.array([3, 8])
    algorithm = object()
    monkeypatch.setattr(
        cli_connectome.nib.loadsave, "load",
        MagicMock(side_effect=[field, domain, segmentation]),
    )
    make_algorithm = MagicMock(return_value=algorithm)
    monkeypatch.setattr(
        cli_connectome.cli_utils, "make_tracking_algorithm", make_algorithm
    )
    run_connectome = MagicMock(return_value=(matrix, labels))
    monkeypatch.setattr(cli_connectome.tg, "connectome", run_connectome)
    save_sparse = MagicMock()
    monkeypatch.setattr(cli_connectome, "save_npz", save_sparse)
    save_labels = MagicMock()
    monkeypatch.setattr(cli_connectome.np, "save", save_labels)

    cli_connectome.main(
        field_path,
        output_path,
        n_seeds=123,
        domain_path=domain_path,
        segmentation_path=segmentation_path,
        distance_upper_bound=5.5,
        maximum_streamline_length=20.0,
        minimum_streamline_length=2.0,
    )

    make_algorithm.assert_called_once_with(field, tg.algorithm.Algorithm.TRANSPORT)
    run_connectome.assert_called_once_with(
        field,
        123,
        domain,
        algorithm,
        segmentation,
        distance_upper_bound=5.5,
        maximum_streamline_length=20.0,
        minimum_streamline_length=2.0,
    )
    save_sparse.assert_called_once()
    assert save_sparse.call_args.args[0] == output_path
    assert isinstance(save_sparse.call_args.args[1], csr_matrix)
    np.testing.assert_array_equal(save_sparse.call_args.args[1].toarray(), matrix)
    save_labels.assert_called_once_with(output_path.with_suffix(".labels.npy"), labels)


def test_main_saves_unspecified_segmentation_as_sparse_matrix(monkeypatch, tmp_path):
    field_path = tmp_path / "field.nii.gz"
    output_path = tmp_path / "connectome.npz"
    field = Nifti1Image(np.ones((2, 2, 2, 45), dtype=np.float32), np.eye(4))
    matrix = csr_matrix([[0.0, 2.0], [4.0, 0.0]])
    monkeypatch.setattr(cli_connectome.nib.loadsave, "load", MagicMock(return_value=field))
    monkeypatch.setattr(
        cli_connectome.cli_utils, "make_tracking_algorithm", MagicMock(return_value=object())
    )
    monkeypatch.setattr(
        cli_connectome.tg, "connectome", MagicMock(return_value=(matrix, np.arange(2)))
    )
    save_sparse = MagicMock()
    monkeypatch.setattr(cli_connectome, "save_npz", save_sparse)
    save_labels = MagicMock()
    monkeypatch.setattr(cli_connectome.np, "save", save_labels)

    cli_connectome.main(field_path, output_path)

    assert save_sparse.call_args.args[0] == output_path
    assert isinstance(save_sparse.call_args.args[1], csr_matrix)
    np.testing.assert_array_equal(save_sparse.call_args.args[1].toarray(), matrix.toarray())
    save_labels.assert_not_called()


def test_cli_parse_dispatch_and_defaults(monkeypatch, tmp_path):
    field_path = tmp_path / "field.nii.gz"
    output_path = tmp_path / "connectome.npz"
    domain_path = tmp_path / "domain.nii.gz"
    segmentation_path = tmp_path / "regions.nii.gz"
    mock_main = MagicMock()
    monkeypatch.setattr(cli_connectome, "main", mock_main)

    args = tractography.cli.parse_arguments(
        [
            "connectome", str(field_path), str(output_path),
            "--number-of-seeds", "12", "--domain", str(domain_path),
            "--segmentation", str(segmentation_path),
            "--distance-upper-bound", "6.5",
            "--maximum-streamline-length", "25",
            "--minimum-streamline-length", "1.5",
        ]
    )
    tractography.cli.main(args)
    mock_main.assert_called_once_with(
        field_path=field_path,
        connectome_path=output_path,
        n_seeds=12,
        domain_path=domain_path,
        algorithm=tg.algorithm.Algorithm.TRANSPORT,
        segmentation_path=segmentation_path,
        distance_upper_bound=6.5,
        maximum_streamline_length=25.0,
        minimum_streamline_length=1.5,
    )

    defaults = tractography.cli.parse_arguments(
        ["connectome", str(field_path), str(output_path)]
    )
    assert defaults.n_seeds == 1_000_000
    assert defaults.domain_path is None
    assert defaults.segmentation_path is None
    assert defaults.distance_upper_bound == 4.0
