"""Entry-point for the histogram command of the CLI

This module implements the histogram subcommand of the tractography CLI. It
allows the user to generate tractography histograms from files.

"""

from pathlib import Path

import nibabel as nib
import numpy as np

import tractography as tg


_ALGORITHM = tg.Algorithm.DIFFUSION


_DESCRIPTION = """
Perform diffusion magnetic resonance imaging tractography but save only the
histogram. The histogram is the probability of streamline orientations for
each voxel, and therefore has the same representations as fibre orientation
distributions.
"""

_HELP = """
generate diffusion MRI tractography histogram
"""


_ALGORITHM_HELP = f"""
the algorithm used for tractography (default is {_ALGORITHM})
"""

_FOD_HELP = """
the filename of the FOD image on which to perform tractography
"""

_SEED_MASK_HELP = """
the seed mask where the seed will be randomly generated
"""

_HISTOGRAM_HELP = """
the filename of the generated histogram
"""


def main(
    fod_path: Path,
    seed_mask_path: Path,
    histogram_path: Path,
    n_seeds: int,
    algorithm: tg.Algorithm = tg.Algorithm.DIFFUSION,
    **kwargs,
):
    """Entry-point of the tractography CLI"""

    # Load the default config and set user parameters.
    config = tg.configuration.load(algorithm)
    tg.cli.utils.set_tractography_config(config, kwargs)

    # Load the seed mask and the FOD.
    seed_mask = nib.load(seed_mask_path)
    fod = nib.load(fod_path)

    histogram = tg.histogram(fod, seed_mask, n_seeds, config)
    nib.save(histogram, histogram_path)


def add_parser(subparsers):
    """Add the surparser for the histogram subcommand"""
    subparser = subparsers.add_parser("histogram", description=_DESCRIPTION, help=_HELP)
    subparser.add_argument("fod_path", type=Path, help=_FOD_HELP)
    subparser.add_argument("seed_mask_path", type=Path, help=_SEED_MASK_HELP)
    subparser.add_argument("histogram_path", type=Path, help=_HISTOGRAM_HELP)
    tg.cli.utils.add_n_seeds_argument(subparser)
    subparser.add_argument(
        "--algorithm",
        type=tg.Algorithm,
        default=_ALGORITHM,
        choices=list(tg.Algorithm),
        help=_ALGORITHM_HELP,
    )

    # Add the common configuration options.
    tg.cli.utils.add_tractography_config(subparser)

    subparser.set_defaults(func=main)
    return subparser