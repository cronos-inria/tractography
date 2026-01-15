import nibabel as nib
import numpy as np
from scipy import ndimage


def multiply(
    left: nib.Nifti1Image,
    right: nib.Nifti1Image,
    order: int = 0,
) -> nib.Nifti1Image:
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

    return nib.Nifti1Image(product, left.affine)
