"""Implements the 'tractography mask' CLI"""

from pathlib import Path

import nibabel as nib
import numpy as np
import scipy


_DESCRIPTION = """
Generate a tractography mask from a brain parcellation and segmentation
"""

_HELP = """
generate a tractography mask
"""

_IMAGE_PATH_HELP = """
the path to the image used to generate the mask
"""

_MASK_PATH_HELP = """
the path to the generated mask
"""

_WM_LABELS = (2, 16, 28, 41, 60, 251, 252, 253, 254, 255)


def main(image_path: Path, mask_path: Path, **kwargs: dict):
    """Generate a mask from image data"""
    mask_nii = nib.load(image_path)
    mask_data = mask_nii.get_fdata()
    mask = scipy.ndimage.binary_dilation(np.isin(mask_data, _WM_LABELS)).astype(np.uint8)
    nib.save(nib.Nifti1Image(mask, mask_nii.affine), mask_path)


def add_parser(subparsers):
    """Add the surparser for the mask subcommand"""
    subparser = subparsers.add_parser("mask", description=_DESCRIPTION, help=_HELP)
    subparser.add_argument("image_path", type=Path, help=_IMAGE_PATH_HELP)
    subparser.add_argument("mask_path", type=str, help=_MASK_PATH_HELP)
    subparser.set_defaults(func=main)


if __name__ == "__main__":
    main()
