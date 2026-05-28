import numpy as np
from nibabel.nifti1 import Nifti1Image
from scipy import ndimage


def multiply(
    left: Nifti1Image,
    right: Nifti1Image,
    order: int = 0,
) -> Nifti1Image:
    """Multiplies two NIfTI images in world space

    The `right` image is resampled to the geometry (affine and shape) of the
    `left` image prior to multiplication.

    Args:
        left: The reference image. Defines the output spatial grid and affine.
        right: The image to resample and multiply against the left image.
        order: The order of the spline interpolation, default is 0.
            Use 0 (nearest-neighbor) for masks/labels to preserve integer values.
            Use 1 (linear) or higher for continuous anatomical data.

    Returns:
        A new NIfTI image containing the product of the inputs in the `left`
        image's space.

    Note:
        This operation is **not commutative** (i.e., `multiply(A, B)` is not
        the same as `multiply(B, A)`). The output image always inherits the
        resolution and field of view of the `left` image.
    """

    # Calculate the mapping from 'left' voxels to 'right' voxels.
    affine = np.dot(np.linalg.inv(right.affine), left.affine)

    # Resample the right image data to match the left image grid.
    right_resampled = ndimage.affine_transform(
        input=right.get_fdata(),
        matrix=affine,
        output_shape=left.shape[:right.ndim],
        order=order,
        mode='constant',
        cval=0.0,
    ).reshape(left.shape[:right.ndim] + (1,) * (left.ndim - right.ndim))

    # Multiply the data.
    product = left.get_fdata() * right_resampled

    # Create a copy of the header to preserve voxel dimensions and units.
    # Update the header to match the new data type.
    new_header = left.header.copy()
    new_header.set_data_dtype(product.dtype)

    return Nifti1Image(product, left.affine, header=new_header)


def threshold(
    nii: Nifti1Image,
    value: float = 0.0,
) -> Nifti1Image:
    """Create bitwise image by thresholding image intensity

    Args:
        nii: The image to threshold.
        value: The threshold value. Voxel with values greater or equal to
            this threshold are set to 1, all other values are set to 0.

    Returns:
        A new NIfTI image containing the thresholded values (uint8), with
        metadata preserved from the input.
    """
    # Create the binary mask
    mask_data = (nii.get_fdata() >= value).astype(np.uint8)

    # Preserve the original header to keep voxel sizes and orientation details.
    # Update the header to reflect the new data type (uint8).
    new_header = nii.header.copy()
    new_header.set_data_dtype(np.uint8)

    return Nifti1Image(mask_data, nii.affine, header=new_header)
