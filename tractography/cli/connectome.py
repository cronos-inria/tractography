"""File-based entry point for the connectome command."""

from pathlib import Path
from typing import cast

import nibabel as nib
from nibabel.nifti1 import Nifti1Image
import numpy as np
from scipy.sparse import csr_matrix, save_npz

import tractography as tg
from . import utils as cli_utils


_ALGORITHM = tg.Algorithm.TRANSPORT
_N_SEEDS = 1_000_000
_DISTANCE_UPPER_BOUND = 4.0
_MAXIMUM_STREAMLINE_LENGTH = 300.0
_MINIMUM_STREAMLINE_LENGTH = 0.0

_DESCRIPTION = """
Generate a connectome from a local-model image. An optional segmentation
aggregates the directed boundary-vertex connectome by region label.
"""
_HELP = "generate a structural connectome"


def main(
    field_path: Path,
    connectome_path: Path,
    n_seeds: int = _N_SEEDS,
    domain_path: Path | None = None,
    algorithm: tg.Algorithm = _ALGORITHM,
    segmentation_path: Path | None = None,
    distance_upper_bound: float = _DISTANCE_UPPER_BOUND,
    maximum_streamline_length: float = _MAXIMUM_STREAMLINE_LENGTH,
    minimum_streamline_length: float = _MINIMUM_STREAMLINE_LENGTH,
) -> None:
    """Load file inputs, call :func:`tractography.connectome`, and save output."""
    field = cast(Nifti1Image, nib.loadsave.load(field_path))
    domain = (
        cast(Nifti1Image, nib.loadsave.load(domain_path))
        if domain_path is not None
        else None
    )
    segmentation = (
        cast(Nifti1Image, nib.loadsave.load(segmentation_path))
        if segmentation_path is not None
        else None
    )
    tracking_algorithm = cli_utils.make_tracking_algorithm(field, algorithm)
    matrix, labels = tg.connectome(
        field,
        n_seeds,
        domain,
        tracking_algorithm,
        segmentation,
        distance_upper_bound=distance_upper_bound,
        maximum_streamline_length=maximum_streamline_length,
        minimum_streamline_length=minimum_streamline_length,
    )
    save_npz(connectome_path, csr_matrix(matrix))
    if segmentation is not None:
        np.save(connectome_path.with_suffix(".labels.npy"), labels)


def add_parser(subparsers):
    """Register the connectome command and its file-based arguments."""
    parser = subparsers.add_parser(
        "connectome", description=_DESCRIPTION, help=_HELP
    )
    parser.add_argument("field_path", type=Path, help="4D local-model NIfTI image")
    parser.add_argument("connectome_path", type=Path, help="output connectome .npz")
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
        "--segmentation", dest="segmentation_path", type=Path,
        help="optional 3D region-label NIfTI image",
    )
    parser.add_argument(
        "--distance-upper-bound",
        dest="distance_upper_bound",
        type=float,
        default=_DISTANCE_UPPER_BOUND,
        help="maximum distance for assigning boundary vertices to labels",
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
        help="minimum propagated distance for a track to contribute",
    )
    parser.set_defaults(func=main)
    return parser
