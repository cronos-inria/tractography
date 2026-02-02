"""Entry-point for the connectivity command of the CLI

This module implements the connectivity subcommand of the tractography CLI. It
allows the user to generate connectivity matrices from files.

"""

from pathlib import Path

import nibabel as nib
import numpy as np

import tractography as tg


_DESCRIPTION = """
Perform diffusion magnetic resonance imaging tractography but save only the
connectivity matrix.
"""

_HELP = """
generate a structural connectivity matrix
"""

_ALGORITHM_HELP = """
the algorithm used for tractography
"""

_IMAGE_HELP = """
the filename of the image on which to perform tractography
"""

_SEGMENTATION_HELP = """
the filename of the segmentation image containing brain region labels
"""

_CONNECTIVITY_HELP = """
the filename of the generated connectivity matrix
"""

_MASK_HELP = """
the filename of the mask used for tractography
"""

_N_SEEDS_HELP = """
the total number of seeds to generate (default: 100000)
"""


def main(
    image_path: Path,
    segmentation_path: Path,
    connectivity_path: Path,
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

    # Generate the connectivity matrix.
    # connectome() handles seeding, tracking, and mapping internally.
    matrix, _ = tg.connectome(fod, segmentation, n_seeds, config)

    # Save the connectivity matrix.
    np.save(connectivity_path, matrix)


def add_parser(subparsers):
    """Add the subparser for the connectivity subcommand"""
    subparser = subparsers.add_parser(
        "connectivity", description=_DESCRIPTION, help=_HELP
    )
    subparser.add_argument("image_path", type=Path, help=_IMAGE_HELP)
    subparser.add_argument("segmentation_path", type=Path, help=_SEGMENTATION_HELP)
    subparser.add_argument("connectivity_path", type=Path, help=_CONNECTIVITY_HELP)
    subparser.add_argument("--n_seeds", type=int, default=100000, help=_N_SEEDS_HELP)
    subparser.add_argument("--mask", type=Path, help=_MASK_HELP)

    subparser.add_argument(
        "--algorithm",
        type=tg.Algorithm,
        choices=list(tg.Algorithm),
        help=_ALGORITHM_HELP,
    )

    # Add the common configuration options.
    tg.cli.utils.add_tractography_config(subparser)

    subparser.set_defaults(func=main)
    return subparser


if __name__ == "__main__":
    main()
