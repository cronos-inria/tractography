"""Entry-point for the connectome command of the CLI

This module implements the connectome subcommand of the tractography CLI. It
allows the user to generate connectivity matrices from files.

"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

import tractography as tg


_ALGORITHM = tg.Algorithm.TRANSPORT


_DESCRIPTION = """
Perform diffusion magnetic resonance imaging tractography but save only the
connectome instead of the full streamlines. The connectome is saved as a .npz file 
containing 'matrix' and 'labels' arrays.
"""

_HELP = """
generate a structural connectome
"""

_ALGORITHM_HELP = f"""
the algorithm used for tractography (default: {_ALGORITHM})
"""

_IMAGE_HELP = """
the filename of the image on which to perform tractography
"""

_SEGMENTATION_HELP = """
the filename of the segmentation image containing brain region labels
"""

_CONNECTOME_HELP = """
the filename of the generated connectome (saved as a .npz file containing 'matrix' and 'labels' arrays)
"""

_MASK_HELP = """
the filename of the mask used for tractography
"""


def main(
    image_path: Path,
    segmentation_path: Path,
    connectome_path: Path,
    n_seeds: int,
    algorithm: tg.Algorithm = tg.Algorithm.TRANSPORT,
    **kwargs,
):
    """Entry-point of the tractography CLI"""

    # Load the default config and set user parameters.
    config = tg.configuration.load(algorithm)
    tg.cli.utils.set_tractography_config(config, kwargs)

    # Load the FOD image.
    fod = nib.load(image_path)

    # Apply an optional global mask (e.g., brain mask) to the FOD.
    if "mask" in kwargs and kwargs["mask"] is not None:
        mask_nii = nib.load(kwargs["mask"])
        fod = tg.nifti.multiply(fod, mask_nii)

    # Load the volumetric segmentation.
    segmentation = nib.load(segmentation_path)

    # Generate the connectome.
    # connectome() handles seeding, tracking, and mapping internally.
    matrix, labels = tg.connectome(fod, segmentation, n_seeds, config)

    # Save the connectome.
    np.savez(connectome_path, matrix=matrix, labels=labels)


def add_parser(subparsers):
    """Add the subparser for the connectome subcommand"""
    subparser = subparsers.add_parser(
        "connectome", description=_DESCRIPTION, help=_HELP
    )
    subparser.add_argument("image_path", type=Path, help=_IMAGE_HELP)
    subparser.add_argument("segmentation_path", type=Path, help=_SEGMENTATION_HELP)
    subparser.add_argument("connectome_path", type=Path, help=_CONNECTOME_HELP)
    subparser.add_argument("--mask", type=Path, help=_MASK_HELP)
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
