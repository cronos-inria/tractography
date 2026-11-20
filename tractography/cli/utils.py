import argparse
from pathlib import Path

from nibabel.nifti1 import Nifti1Image

import tractography as tg

_N_SEEDS = 1000000


_N_SEEDS_HELP = f"""
the number of seeds to generate (default is {_N_SEEDS})
"""


def add_n_seeds_argument(parser: argparse.ArgumentParser):
    """Add the API-aligned seed-count argument to a parser."""
    parser.add_argument(
        "--number-of-seeds",
        dest="n_seeds",
        type=_non_negative_int,
        default=_N_SEEDS,
        help=_N_SEEDS_HELP,
    )


def add_domain_argument(parser: argparse.ArgumentParser):
    """Add an optional domain-image argument."""
    parser.add_argument(
        "--domain",
        dest="domain_path",
        type=Path,
        help="optional 3D domain image; defaults to nonzero field voxels",
    )


def make_tracking_algorithm(
    field: Nifti1Image, algorithm: tg.algorithm.Algorithm
) -> tg.BaseAlgorithm:
    """Construct the selected algorithm using the public constructor defaults."""
    local_model = tg.local.from_shape(field.shape)

    if algorithm == tg.algorithm.Algorithm.TRANSPORT:
        return tg.algorithm.Transport(local_model)
    if algorithm == tg.algorithm.Algorithm.DETERMINISTIC:
        return tg.algorithm.Deterministic(local_model)
    if algorithm == tg.algorithm.Algorithm.DIFFUSION:
        return tg.algorithm.Diffusion(local_model)
    if algorithm == tg.algorithm.Algorithm.PROBABILISTIC:
        return tg.algorithm.Probabilistic(local_model)
    raise ValueError(f"Unsupported tracking algorithm: {algorithm}")


def _non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"must be a valid integer, not {value!r}") from exc

    if parsed < 0:
        raise argparse.ArgumentTypeError(f"must be a non-negative integer, not {value!r}")

    return parsed
