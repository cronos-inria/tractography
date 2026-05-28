"""Entry-point for the Command Line Interface

This module implements the main CLI for the tractography package. It
allows the user to perform tractography from files.

"""

from pathlib import Path

import nibabel as nib
import numpy as np

import tractography as tg


_ALGORITHM = tg.Algorithm.TRANSPORT


_DESCRIPTION = """
Perform diffusion magnetic resonance imaging tractography.
"""

_HELP = """
perform diffusion MRI tractography
"""


_ALGORITHM_HELP = f"""
the algorithm used for tractography (default: {_ALGORITHM})
"""

_IMAGE_HELP = """
the filename of the image on which to perform tractography
"""

_SEEDS_HELP = """
the seeding strategy and data
"""

_TRACTOGRAM_HELP = """
the filename of the generated tractogram
"""

_MASK_HELP = """
the filename of the mask used for tractography
"""


def main(
    image_path: Path,
    seeds_path: Path,
    tractogram_path: Path,
    algorithm: tg.Algorithm = _ALGORITHM,
    **kwargs,
):
    """Entry-point of the tractography CLI"""

    # Load the default config and set user parameters.
    config = tg.configuration.load(algorithm)
    tg.cli.utils.set_tractography_config(config, kwargs)

    # Load the seeds from the provided file.
    seeds = tg.seeds.load(seeds_path)

    # Load the FOD image and apply the mask if one was provided.
    fod = nib.loadsave.load(image_path)
    if "mask" in kwargs and kwargs["mask"] is not None:
        mask = nib.loadsave.load(kwargs["mask"])
        fod = tg.nifti.multiply(fod, mask)

    tractogram = tg.tractogram(fod, seeds, config)

    # Save the resulting tractogram.
    tck = nib.streamlines.tck.TckFile(tractogram)
    tck.save(tractogram_path)


def add_parser(subparsers):
    """Add the subparser for the tractogram subcommand"""
    subparser = subparsers.add_parser(
        "tractogram", description=_DESCRIPTION, help=_HELP
    )
    subparser.add_argument("image_path", type=Path, help=_IMAGE_HELP)
    subparser.add_argument("seeds_path", type=Path, help=_SEEDS_HELP)
    subparser.add_argument("tractogram_path", type=Path, help=_TRACTOGRAM_HELP)
    subparser.add_argument("--mask", type=Path, help=_MASK_HELP)
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