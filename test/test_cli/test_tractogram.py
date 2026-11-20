from unittest.mock import MagicMock

import nibabel as nib
from nibabel.streamlines.tractogram import Tractogram
import numpy as np
from trx.trx_file_memmap import load as load_trx

import tractography as tg
import tractography.cli.tractogram
import tractography.cli
import test.data


class TestTractogram:
    """Test the generation of tractograms from the CLI"""

    def test_cross_tractography(self, tmp_path):
        """Test tractography on the cross dataset"""

        # Prepare the data.
        fod, wm, _ = test.data.cross()
        fod_path = tmp_path / "cross-fod.nii.gz"
        nib.save(fod, fod_path)

        wm_path = tmp_path / "cross-wm.nii.gz"
        nib.save(wm, wm_path)

        seeds = tg.seeds.from_mask(wm, 1000)
        seeds_path = tmp_path / "cross-seeds.txt"
        tg.seeds.save(seeds_path, seeds)

        tractogram_path = tmp_path / "cross-streamlines.tck"
        tg.cli.tractogram.main(
            fod_path, seeds_path, tractogram_path,
            batch_size=1000,
        )


def test_main_delegates_to_public_tractogram(monkeypatch, tmp_path):
    field_path = tmp_path / "field.nii.gz"
    seeds_path = tmp_path / "seeds.txt"
    domain_path = tmp_path / "domain.nii.gz"
    output_path = tmp_path / "tracks.tck"
    field = nib.Nifti1Image(np.ones((2, 2, 2, 45), dtype=np.float32), np.eye(4))
    domain = nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.uint8), np.eye(4))
    seeds = np.array([[0, 0, 0, 1, 1, 1, 1, 0]], dtype=np.float32)
    algorithm = object()
    result = Tractogram([], affine_to_rasmm=np.eye(4))
    monkeypatch.setattr(
        tg.cli.tractogram.nib.loadsave, "load",
        MagicMock(side_effect=[field, domain]),
    )
    monkeypatch.setattr(tg.seeds, "load", MagicMock(return_value=seeds))
    monkeypatch.setattr(
        tg.cli.tractogram.cli_utils,
        "make_tracking_algorithm",
        MagicMock(return_value=algorithm),
    )
    run_tractogram = MagicMock(return_value=result)
    monkeypatch.setattr(tg, "tractogram", run_tractogram)
    save = MagicMock()
    monkeypatch.setattr(tg.cli.tractogram.nib.streamlines.tck.TckFile, "save", save)

    tg.cli.tractogram.main(
        field_path,
        seeds_path,
        output_path,
        domain_path=domain_path,
        save_at=0.5,
        maximum_streamline_length=12.0,
        minimum_streamline_length=2.0,
        batch_size=7,
    )

    run_tractogram.assert_called_once_with(
        field,
        seeds,
        domain,
        algorithm,
        save_at=0.5,
        maximum_streamline_length=12.0,
        minimum_streamline_length=2.0,
        batch_size=7,
    )
    save.assert_called_once_with(output_path)


def test_cli_parse_dispatch_and_defaults(monkeypatch, tmp_path):
    field_path = tmp_path / "field.nii.gz"
    seeds_path = tmp_path / "seeds.txt"
    output_path = tmp_path / "tracks.tck"
    domain_path = tmp_path / "domain.nii.gz"
    mock_main = MagicMock()
    monkeypatch.setattr(tg.cli.tractogram, "main", mock_main)

    args = tractography.cli.parse_arguments(
        [
            "tractogram", str(field_path), str(seeds_path), str(output_path),
            "--domain", str(domain_path), "--save-at", "0.5",
            "--maximum-streamline-length", "25",
            "--minimum-streamline-length", "1.5", "--batch-size", "17",
        ]
    )
    tractography.cli.main(args)
    mock_main.assert_called_once_with(
        field_path=field_path,
        seeds_path=seeds_path,
        tractogram_path=output_path,
        domain_path=domain_path,
        algorithm=tg.algorithm.Algorithm.TRANSPORT,
        save_at=0.5,
        maximum_streamline_length=25.0,
        minimum_streamline_length=1.5,
        batch_size=17,
    )

    defaults = tractography.cli.parse_arguments(
        ["tractogram", str(field_path), str(seeds_path), str(output_path)]
    )
    assert defaults.domain_path is None
    assert defaults.batch_size == 50_000
    assert defaults.save_at == 1.0


def test_trx_output_is_written_incrementally(monkeypatch, tmp_path):
    fod = nib.Nifti1Image(np.zeros((2, 2, 2, 1), dtype=np.float32), np.eye(4))
    mask = nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.uint8), np.eye(4))
    seeds = np.array(
        [[i, 0, 0, 1, 1, 0, 0, 0] for i in range(3)], dtype=np.float32
    )
    monkeypatch.setattr(
        tg, "_tracking_components", lambda *args, **kwargs: (object(), object(), object())
    )
    batch_sizes = []

    def fake_track(_algorithm, _domain, _field, batch, **kwargs):
        batch_sizes.append(len(batch))
        return Tractogram(
            [
                np.asarray([[i, 0, 0], [i + 1, 0, 0]], dtype=np.float32)
                for i in range(len(batch))
            ],
            affine_to_rasmm=np.eye(4),
        )

    monkeypatch.setattr(tg.tracking, "tractogram", fake_track)
    output_path = tmp_path / "streamlines.trx"

    tg.cli.tractogram._write_trx_batches(
        fod,
        mask,
        seeds,
        object(),
        output_path,
        save_at=1.0,
        maximum_streamline_length=5.0,
        minimum_streamline_length=0.0,
        batch_size=2,
    )

    tractogram = load_trx(str(output_path))
    try:
        assert batch_sizes == [2, 1]
        assert tractogram.header["NB_STREAMLINES"] == 3
        assert tractogram.header["NB_VERTICES"] == 6
    finally:
        tractogram.close()
