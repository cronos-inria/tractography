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
    field: Nifti1Image, algorithm: tg.Algorithm
) -> tg.BaseAlgorithm:
    """Construct the selected algorithm using the public constructor defaults."""
    model_type = tg.algorithms.core.LocalModel.from_shape(field.shape)
    if model_type == tg.algorithms.core.LocalModel.DTI:
        local_model = tg.local.DiffusionTensorImaging()
    else:
        local_model = tg.local.RealSymmetricSphericalHarmonics(field.shape[-1])

    if algorithm == tg.Algorithm.TRANSPORT:
        return tg.algorithm.Transport(local_model)
    if algorithm == tg.Algorithm.DETERMINISTIC:
        return tg.algorithm.Deterministic(local_model)
    if algorithm == tg.Algorithm.DIFFUSION:
        return tg.algorithm.Diffusion(local_model)
    if algorithm == tg.Algorithm.PROBABILISTIC:
        return tg.algorithm.Probabilistic(local_model)
    raise ValueError(f"Unsupported tracking algorithm: {algorithm}")


def add_tractography_config(parser: argparse.ArgumentParser):
    """Add the common tractography configuration to a parser"""

    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=float,
    )

    # Global properties.
    parser.add_argument(
        "--step-size",
        dest="step_size",
        type=float,
    )

    # Streamline properties.
    parser.add_argument(
        "--streamline-minimum-length",
        dest="min_length",
        type=float,
    )
    parser.add_argument(
        "--streamline-maximum-length",
        dest="max_length",
        type=float,
    )


def set_tractography_config(
    config: tg.algorithms.core.BaseConfiguration, kwargs
):
    """Map CLI arguments back to the configuration"""

    if "batch_size" in kwargs and kwargs["batch_size"] is not None:
        config.batch_size = kwargs["batch_size"]

    if "step_size" in kwargs and kwargs["step_size"] is not None:
        config.step_size = kwargs["step_size"]

    if "min_length" in kwargs and kwargs["min_length"] is not None:
        config.streamline.length.minimum = kwargs["min_length"]

    if "max_length" in kwargs and kwargs["max_length"] is not None:
        config.streamline.length.maximum = kwargs["max_length"]


def _non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"must be a valid integer, not {value!r}") from exc

    if parsed < 0:
        raise argparse.ArgumentTypeError(f"must be a non-negative integer, not {value!r}")

    return parsed
