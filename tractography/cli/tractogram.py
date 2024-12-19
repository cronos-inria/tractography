"""Entry-point for the Command Line Interface

This module implements the main CLI for the tractography package. It
allows the user to perform tractography from files.

"""

from enum import Enum
from pathlib import Path

import nibabel as nib
import nimesh
import numpy as np

import tractography as tg


_DESCRIPTION = """
Perform diffusion magnetic resonance imaging tractography.
"""

_HELP = """
perform diffusion MRI tractography
"""


_ALGORITHM_HELP = """
the algorithm used for tractography
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

_N_STEPS = """
the number of steps for each streamline
"""

_MAX_ANGLE = """
the maximum angle between consecutive steps
"""

_STEP_SIZE = """
the size of each step
"""


class Algorithm(Enum):
    """The various tractography algorithms"""

    DETERMINISTIC = "det"
    PROBABILISTIC = "prob"

    def __str__(self):
        return self.value


def main(
    algorithm: Algorithm,
    image_path: Path,
    seeds_path: Path,
    tractogram_path: Path,
    step_size: int,
    n_steps: int,
    max_angle: float,
    **kwargs
):
    """Entry-point of the tractography CLI"""

    # Load the seeds from the provided surface.
    seeds = tg.seeds.load(seeds_path)

    # Load the FOD image.
    nii = nib.load(image_path)
    data = nii.get_fdata()

    # Create the mask from the segmentation and apply it to the data.
    if "mask" in kwargs and kwargs["mask"] is not None:
        mask_nii = nib.load(kwargs["mask"])
        mask = mask_nii.get_fdata()
        data = tg.core.apply_mask(data, nii.affine, mask, mask_nii.affine)

    if algorithm == Algorithm.DETERMINISTIC:
        streamlines = tg.algorithms.deterministic(
            data, nii.affine, seeds, step_size, n_steps, max_angle
        )
    else:
        streamlines = tg.algorithms.probabilistic(
            data, nii.affine, seeds, step_size, n_steps, max_angle
        )

    # Clean a bit.
    streamlines = tg.core.remove_duplicate_endpoints(streamlines, step_size)
    streamlines = [s for s in streamlines if len(s) != 0]

    # Save the resulting tractogram.
    tractogram = nib.streamlines.Tractogram(streamlines, affine_to_rasmm=np.eye(4))
    tck = nib.streamlines.TckFile(tractogram)
    tck.save(tractogram_path)


def add_parser(subparsers):
    """Add the surparser for the mask subcommand"""
    subparser = subparsers.add_parser(
        "tractogram", description=_DESCRIPTION, help=_HELP
    )
    subparser.add_argument(
        "algorithm", type=Algorithm, choices=list(Algorithm), help=_ALGORITHM_HELP
    )
    subparser.add_argument("image_path", type=Path, help=_IMAGE_HELP)
    subparser.add_argument("seeds_path", type=Path, help=_SEEDS_HELP)
    subparser.add_argument("tractogram_path", type=Path, help=_TRACTOGRAM_HELP)
    subparser.add_argument("--mask", type=Path, help=_MASK_HELP)
    subparser.add_argument(
        "--number-of-steps",
        "-ns",
        dest="n_steps",
        type=int,
        default=2000,
        help=_N_STEPS,
    )
    subparser.add_argument(
        "--maximum-angle",
        "-ma",
        dest="max_angle",
        type=float,
        default=45,
        help=_MAX_ANGLE,
    )
    subparser.add_argument(
        "--step-size",
        "-ss",
        dest="step_size",
        type=float,
        default=0.25,
        help=_STEP_SIZE,
    )

    subparser.set_defaults(func=main)

    return subparser


if __name__ == "__main__":
    main()
