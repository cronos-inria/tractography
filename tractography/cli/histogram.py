"""Entry-point for the histogram command of the CLI

This module implements the histogram subcommand of the tractography CLI. It
allows the user to generate tractography histograms from files.

"""

from enum import Enum
from pathlib import Path

import nibabel as nib
import numpy as np

import tractography as tg


_DESCRIPTION = """
Perform diffusion magnetic resonance imaging tractography but save only the
histogram (also known has tract-density imaging).
"""

_HELP = """
generate diffusion MRI tractography histogram (tract-density imaging)
"""


_ALGORITHM_HELP = """
the algorithm used for tractography
"""

_IMAGE_HELP = """
the filename of the image on which to perform tractography
"""

_SEEDS_HELP = """
the seeds used to generate streamlines (1 streamline per seed)
"""

_HISTOGRAM_HELP = """
the filename of the generated histogram
"""

_MASK_HELP = """
the filename of the mask used for tractography
"""

_SCALE_HELP = """
the scaling factor to apply to the histogram (2 means 2**3 more voxels)
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
    histogram_path: Path,
    step_size: int,
    n_steps: int,
    max_angle: float,
    scale: int,
    **kwargs,
):
    """Entry-point of the tractography CLI"""

    # Load the seeds from the provided file.
    seeds = tg.seeds.load(seeds_path)

    # Load the FOD image.
    nii = nib.load(image_path)
    data = nii.get_fdata()

    # Create the mask from the segmentation and apply it to the data.
    if "mask" in kwargs and kwargs["mask"] is not None:
        mask_nii = nib.load(kwargs["mask"])
        mask = mask_nii.get_fdata()
        data = tg.core.apply_mask(data, nii.affine, mask, mask_nii.affine)

    # Compute the affine to go back to voxel space in the histogram.
    affine_scale = np.eye(4)
    affine_scale[(0, 1, 2), (0, 1, 2)] = 1 / scale
    new_affine = np.dot(nii.affine, affine_scale)
    inverse_affine = np.linalg.inv(new_affine)

    # Reserve memory for the resulting image.
    histogram = np.zeros([n * scale for n in data.shape[:3]], dtype=np.uint32)

    # Because the number of seeds can be enormeous, we split them into batches.
    n_splits = len(seeds) // tg.BATCH_SIZE
    for subseeds in np.array_split(seeds, n_splits):

        if algorithm == Algorithm.DETERMINISTIC:
            streamlines = tg.algorithms.deterministic(
                data, nii.affine, subseeds, step_size, n_steps, max_angle
            )
        else:
            streamlines = tg.algorithms.probabilistic(
                data, nii.affine, subseeds, step_size, n_steps, max_angle
            )

        # Clean a bit.
        streamlines = tg.core.remove_duplicate_endpoints(streamlines, step_size)
        streamlines = [s for s in streamlines if len(s) != 0]

        # Add the streamlines to the histogram.
        points = np.vstack(streamlines)
        voxels = nib.affines.apply_affine(inverse_affine, points).astype(int)
        for voxel in voxels:
            histogram[*voxel] += 1

    # Save the histogram.
    nib.save(nib.Nifti1Image(histogram, new_affine), histogram_path)


def add_parser(subparsers):
    """Add the surparser for the mask subcommand"""
    subparser = subparsers.add_parser("histogram", description=_DESCRIPTION, help=_HELP)
    subparser.add_argument(
        "algorithm", type=Algorithm, choices=list(Algorithm), help=_ALGORITHM_HELP
    )
    subparser.add_argument("image_path", type=Path, help=_IMAGE_HELP)
    subparser.add_argument("seeds_path", type=Path, help=_SEEDS_HELP)
    subparser.add_argument("histogram_path", type=Path, help=_HISTOGRAM_HELP)
    subparser.add_argument("--mask", type=Path, help=_MASK_HELP)
    subparser.add_argument("--scale", type=int, default=2, help=_SCALE_HELP)
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
