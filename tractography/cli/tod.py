"""File-based entry point for the track-orientation distribution command."""

from pathlib import Path
from typing import cast

import nibabel as nib
from nibabel.nifti1 import Nifti1Image

import tractography as tg
from . import utils as cli_utils


_ALGORITHM = tg.Algorithm.TRANSPORT
_N_SEEDS = 1_000_000
_MAXIMUM_STREAMLINE_LENGTH = 300.0

_DESCRIPTION = """
Generate a track orientation distribution (TOD) image from a local-model image.
By default, nonzero field voxels define the tracking domain and TOD output grid.
"""
_HELP = "generate a track orientation distribution image"


def main(
    field_path: Path,
    tod_path: Path,
    n_seeds: int = _N_SEEDS,
    domain_path: Path | None = None,
    algorithm: tg.Algorithm = _ALGORITHM,
    maximum_streamline_length: float = _MAXIMUM_STREAMLINE_LENGTH,
) -> None:
    """Load file inputs, call :func:`tractography.tod`, and save its result."""
    field = cast(Nifti1Image, nib.loadsave.load(field_path))
    domain = (
        cast(Nifti1Image, nib.loadsave.load(domain_path))
        if domain_path is not None
        else None
    )
    tracking_algorithm = cli_utils.make_tracking_algorithm(field, algorithm)
    result = tg.tod(
        field,
        n_seeds,
        domain,
        tracking_algorithm,
        maximum_streamline_length=maximum_streamline_length,
    )
    nib.loadsave.save(result, tod_path)


def add_parser(subparsers):
    """Register the TOD command and its file-based arguments."""
    parser = subparsers.add_parser("tod", description=_DESCRIPTION, help=_HELP)
    parser.add_argument("field_path", type=Path, help="4D local-model NIfTI image")
    parser.add_argument("tod_path", type=Path, help="output TOD NIfTI image")
    cli_utils.add_n_seeds_argument(parser)
    cli_utils.add_domain_argument(parser)
    parser.add_argument(
        "--algorithm",
        type=tg.Algorithm,
        default=_ALGORITHM,
        choices=list(tg.Algorithm),
        help=f"tracking algorithm (default: {_ALGORITHM})",
    )
    parser.add_argument(
        "--maximum-streamline-length",
        dest="maximum_streamline_length",
        type=float,
        default=_MAXIMUM_STREAMLINE_LENGTH,
        help="maximum propagation distance per track",
    )
    parser.set_defaults(func=main)
    return parser