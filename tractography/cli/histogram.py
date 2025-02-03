"""Entry-point for the histogram command of the CLI

This module implements the histogram subcommand of the tractography CLI. It
allows the user to generate tractography histograms from files.

"""

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


def main(
    algorithm: tg.Algorithm,
    image_path: Path,
    seeds_path: Path,
    histogram_path: Path,
    scale: int,
    **kwargs,
):
    """Entry-point of the tractography CLI"""

    # Load the default config and set user parameters.
    config = tg.configuration.load()
    tg.cli.utils.set_tractography_config(config, kwargs)

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

        streamlines = tg.tractogram(data, nii.affine, subseeds, algorithm, config)

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
        "algorithm", type=tg.Algorithm, choices=list(tg.Algorithm), help=_ALGORITHM_HELP
    )
    subparser.add_argument("image_path", type=Path, help=_IMAGE_HELP)
    subparser.add_argument("seeds_path", type=Path, help=_SEEDS_HELP)
    subparser.add_argument("histogram_path", type=Path, help=_HISTOGRAM_HELP)
    subparser.add_argument("--mask", type=Path, help=_MASK_HELP)
    subparser.add_argument("--scale", type=int, default=2, help=_SCALE_HELP)

    # Add the common configuration options.
    tg.cli.utils.add_tractography_config(subparser)

    subparser.set_defaults(func=main)
    return subparser


if __name__ == "__main__":
    main()
