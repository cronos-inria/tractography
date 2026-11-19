from pathlib import Path
from unittest.mock import MagicMock

from nibabel.nifti1 import Nifti1Image
import numpy as np

from tractography.cli import tod as cli_tod
import tractography as tg
import tractography.cli


def test_main_loads_files_and_delegates_to_public_tod(monkeypatch, tmp_path):
    field_path = tmp_path / "field.nii.gz"
    domain_path = tmp_path / "domain.nii.gz"
    output_path = tmp_path / "tod.nii.gz"
    field = Nifti1Image(np.ones((2, 2, 2, 45), dtype=np.float32), np.eye(4))
    domain = Nifti1Image(np.ones((2, 2, 2), dtype=np.uint8), np.eye(4))
    result = Nifti1Image(np.zeros((2, 2, 2, 45), dtype=np.float32), np.eye(4))
    algorithm = object()
    load = MagicMock(side_effect=[field, domain])
    make_algorithm = MagicMock(return_value=algorithm)
    run_tod = MagicMock(return_value=result)
    save = MagicMock()
    monkeypatch.setattr(cli_tod.nib.loadsave, "load", load)
    monkeypatch.setattr(cli_tod.cli_utils, "make_tracking_algorithm", make_algorithm)
    monkeypatch.setattr(cli_tod.tg, "tod", run_tod)
    monkeypatch.setattr(cli_tod.nib.loadsave, "save", save)

    cli_tod.main(
        field_path,
        output_path,
        n_seeds=123,
        domain_path=domain_path,
        maximum_streamline_length=12.0,
    )

    assert load.call_args_list[0].args == (field_path,)
    assert load.call_args_list[1].args == (domain_path,)
    make_algorithm.assert_called_once_with(field, tg.Algorithm.TRANSPORT)
    run_tod.assert_called_once_with(
        field, 123, domain, algorithm, maximum_streamline_length=12.0
    )
    save.assert_called_once_with(result, output_path)


def test_cli_parse_dispatch_and_defaults(tmp_path, monkeypatch):
    field_path = tmp_path / "field.nii.gz"
    output_path = tmp_path / "tod.nii.gz"
    domain_path = tmp_path / "domain.nii.gz"
    mock_main = MagicMock()
    monkeypatch.setattr(cli_tod, "main", mock_main)
    args = tractography.cli.parse_arguments(
        [
            "tod", str(field_path), str(output_path),
            "--number-of-seeds", "12", "--domain", str(domain_path),
            "--maximum-streamline-length", "25",
        ]
    )
    tractography.cli.main(args)
    mock_main.assert_called_once_with(
        field_path=field_path,
        tod_path=output_path,
        n_seeds=12,
        domain_path=domain_path,
        algorithm=tg.Algorithm.TRANSPORT,
        maximum_streamline_length=25.0,
    )

    defaults = tractography.cli.parse_arguments(
        ["tod", str(field_path), str(output_path)]
    )
    assert defaults.n_seeds == 1_000_000
    assert defaults.domain_path is None
    assert defaults.maximum_streamline_length == 300.0