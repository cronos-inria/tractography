"""File-based entry point for tractogram generation."""

from pathlib import Path
from typing import Any, cast

import nibabel as nib
from nibabel.nifti1 import Nifti1Image
from nibabel.streamlines.tractogram import Tractogram
import numpy as np

import tractography as tg
from . import utils as cli_utils


_ALGORITHM = tg.Algorithm.TRANSPORT
_SAVE_AT = 1.0
_MAXIMUM_STREAMLINE_LENGTH = 300.0
_MINIMUM_STREAMLINE_LENGTH = 0.0
_BATCH_SIZE = 50_000

_DESCRIPTION = """
Generate streamlines from a local-model image and a seed file. TRX output is
written incrementally in batches.
"""
_HELP = "perform diffusion MRI tractography"


def main(
    field_path: Path,
    seeds_path: Path,
    tractogram_path: Path,
    domain_path: Path | None = None,
    algorithm: tg.Algorithm = _ALGORITHM,
    save_at: float = _SAVE_AT,
    maximum_streamline_length: float = _MAXIMUM_STREAMLINE_LENGTH,
    minimum_streamline_length: float = _MINIMUM_STREAMLINE_LENGTH,
    batch_size: int = _BATCH_SIZE,
) -> None:
    """Load file inputs and delegate tracking to :func:`tractography.tractogram`.

    TRX files use a streaming implementation to avoid holding all streamlines
    in memory. Other supported outputs are written from the public in-memory
    tractogram result.
    """
    field = cast(Nifti1Image, nib.loadsave.load(field_path))
    seeds = tg.seeds.load(seeds_path)
    domain = (
        cast(Nifti1Image, nib.loadsave.load(domain_path))
        if domain_path is not None
        else None
    )
    tracking_algorithm = cli_utils.make_tracking_algorithm(field, algorithm)

    if tractogram_path.suffix.lower() == ".trx":
        _write_trx_batches(
            field,
            domain,
            seeds,
            tracking_algorithm,
            tractogram_path,
            save_at=save_at,
            maximum_streamline_length=maximum_streamline_length,
            minimum_streamline_length=minimum_streamline_length,
            batch_size=batch_size,
        )
        return

    result = tg.tractogram(
        field,
        seeds,
        domain,
        tracking_algorithm,
        save_at=save_at,
        maximum_streamline_length=maximum_streamline_length,
        minimum_streamline_length=minimum_streamline_length,
        batch_size=batch_size,
    )
    nib.streamlines.tck.TckFile(result).save(tractogram_path)


def _write_trx_batches(
    field: Nifti1Image,
    domain_image: Nifti1Image | None,
    seeds: list[tg.seeds.Seed],
    tracking_algorithm: tg.BaseAlgorithm,
    output_path: Path,
    *,
    save_at: float,
    maximum_streamline_length: float,
    minimum_streamline_length: float,
    batch_size: int,
) -> None:
    """Track and append seed batches directly to a memory-mapped TRX file."""
    try:
        from trx.trx_file_memmap import TrxFile, save as save_trx
    except ImportError as exc:
        raise ImportError(
            "Writing .trx output requires the trx-python package."
        ) from exc

    if not np.isfinite(save_at) or save_at <= 0:
        raise ValueError("save_at must be finite and positive.")
    if not np.isfinite(maximum_streamline_length) or maximum_streamline_length < 0:
        raise ValueError("maximum_streamline_length must be finite and non-negative.")
    if not np.isfinite(minimum_streamline_length) or minimum_streamline_length < 0:
        raise ValueError("minimum_streamline_length must be finite and non-negative.")
    if minimum_streamline_length > maximum_streamline_length:
        raise ValueError(
            "minimum_streamline_length must not exceed maximum_streamline_length."
        )
    if not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    tracking_algorithm, tracking_domain, tracking_field = tg._tracking_components(
        field, tracking_algorithm, domain_image=domain_image
    )
    writer = TrxFile(reference=cast(Any, field))
    minimum_points = int(np.ceil(minimum_streamline_length / save_at))
    try:
        for start in range(0, len(seeds), batch_size):
            batch = tg._seeds_to_tracking_array(seeds[start : start + batch_size])
            result = tg.tracking.tractogram(
                tracking_algorithm,
                tracking_domain,
                tracking_field,
                batch,
                save_at=save_at,
                maximum_streamline_length=maximum_streamline_length,
            )
            streamlines = [
                streamline
                for streamline in result.streamlines
                if len(streamline) > minimum_points
            ]
            if streamlines:
                writer.append(Tractogram(streamlines, affine_to_rasmm=np.eye(4)))
        save_trx(writer, str(output_path))
    finally:
        writer.close()


def add_parser(subparsers):
    """Register the tractogram command and its file-based arguments."""
    parser = subparsers.add_parser(
        "tractogram", description=_DESCRIPTION, help=_HELP
    )
    parser.add_argument("field_path", type=Path, help="4D local-model NIfTI image")
    parser.add_argument("seeds_path", type=Path, help="seed file")
    parser.add_argument(
        "tractogram_path", type=Path, help="output tractogram (.tck or .trx)"
    )
    cli_utils.add_domain_argument(parser)
    parser.add_argument(
        "--algorithm",
        type=tg.Algorithm,
        default=_ALGORITHM,
        choices=list(tg.Algorithm),
        help=f"tracking algorithm (default: {_ALGORITHM})",
    )
    parser.add_argument(
        "--save-at", type=float, default=_SAVE_AT,
        help="distance interval between saved streamline points",
    )
    parser.add_argument(
        "--maximum-streamline-length",
        dest="maximum_streamline_length",
        type=float,
        default=_MAXIMUM_STREAMLINE_LENGTH,
        help="maximum propagation distance per track",
    )
    parser.add_argument(
        "--minimum-streamline-length",
        dest="minimum_streamline_length",
        type=float,
        default=_MINIMUM_STREAMLINE_LENGTH,
        help="minimum saved-point distance for retaining a streamline",
    )
    parser.add_argument(
        "--batch-size", type=int, default=_BATCH_SIZE,
        help="maximum number of seeds tracked per batch",
    )
    parser.set_defaults(func=main)
    return parser