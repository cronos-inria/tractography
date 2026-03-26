import nibabel as nib
import numpy as np

import tractography as tg


# Some commons tensor shapes.
_BALL = [
    [5.0, 0, 0],
    [0, 5.0, 0],
    [0, 0, 5.0],
]
_X_TENSOR = [
    [5.0, 0, 0],
    [0, 1.0, 0],
    [0, 0, 5.0],
]
_Y_TENSOR = [
    [1.0, 0, 0],
    [0, 5.0, 0],
    [0, 0, 5.0],
]


def circle(shape=(10, 10, 1), n_coefficients=45, radius=2, width=1):
    """Generate a tensor field following a circular pattern"""

    fod_values = np.zeros(shape + (6,))
    fod_values[..., :3] = 0.003  # in mm^2/s.

    wm_data = np.zeros(shape, dtype=np.uint8)

    center = np.array(shape, dtype=float) / 2 - 0.5
    affine = np.eye(4)
    affine[:3, 3] = center

    
    for i, j, k in np.ndindex(*shape):
        x, y, z = np.r_[i, j, k] - center
        r = np.sqrt(x * x + y * y)
        if radius - width < r < radius + width:
            eigvec = np.array(
                [
                    [y / r, x / r, 0],
                    [-x / r, y / r, 0],
                    [0, 0, 1.0],
                ]
            )
            tensor = eigvec.dot(np.diag([1.5, 0.4, 0.4]) / 1000).dot(eigvec.T)
            fod_values[i, j, k] = tensor[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]
            wm_data[i, j, k] = 1

    tensors = nib.nifti1.Nifti1Image(fod_values, affine)
    wm = nib.nifti1.Nifti1Image(wm_data, affine)

    return tensors, wm, None


def cross(shape=(20, 20, 1)):
    """Generate a tensor field with crossing fibers (X and Y oriented)"""

    # Initialize the data arrays.
    fod_values = np.zeros(shape + (6,))
    wm_data = np.zeros(shape, dtype=np.uint8)

    # Initialize with isotropic (ball) tensor when not on the crossing regions.
    fod_values[..., :3] = 0.003  # in mm^2/s.
    x_offset = shape[0] // 3
    x_size = shape[0] - 2 * x_offset
    y_offset = shape[1] // 3
    y_size = shape[1] - 2 * y_offset
    fod_values[x_offset : x_offset + x_size, :, :, :3] = 0
    fod_values[:, y_offset : y_offset + y_size, :, :3] = 0

    # Place the 0 at the center of the image, for better visualization in viewers.
    center = np.array(shape, dtype=float) / 2 - 0.5
    affine = np.eye(4)
    affine[:3, 3] = center

    # X-oriented region: higher diffusion along X axis
    fod_values[x_offset : x_offset + x_size, :, :, :3] += [0.4e-3, 1.5e-3, 0.4e-3]
    wm_data[x_offset : x_offset + x_size, :, :] = 1

    # Y-oriented region: higher diffusion along Y axis
    fod_values[:, y_offset : y_offset + y_size, :, :3] += [1.5e-3, 0.4e-3, 0.4e-3]
    wm_data[:, y_offset : y_offset + y_size, :] = 1

    tensors = nib.nifti1.Nifti1Image(fod_values, affine)
    wm = nib.nifti1.Nifti1Image(wm_data, affine)

    return tensors, wm, None