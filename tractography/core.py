from enum import Enum

import nibabel as nib
import numpy as np
import numpy.typing as npt
import scipy


class Algorithm(Enum):
    """The available tractography algorithms"""

    DETERMINISTIC = "det"
    PROBABILISTIC = "prob"

    def __str__(self):
        return self.value


def cart2sph(x, y, z):
    radius = np.sqrt(x**2 + y**2 + z**2)
    azimuth = np.arctan2(y, x)
    azimuth[azimuth < 0] = 2 * np.pi + azimuth[azimuth < 0]
    colatitude = np.arccos(z / radius)
    return azimuth, colatitude, radius


def sph2cart(azimuth, colatitude, radius):
    x = radius * np.sin(colatitude) * np.cos(azimuth)
    y = radius * np.sin(colatitude) * np.sin(azimuth)
    z = radius * np.cos(colatitude)
    return x, y, z


def ishtmtx(azimuths, colatitudes, n_coefficients):
    """Generates the mrtrix3 compatible inverse SH transform matrix"""

    matrix = np.zeros((n_coefficients, len(colatitudes)))
    degree = 0
    order = 0
    for i in range(n_coefficients):
        if order < 0:
            matrix[i] = np.sqrt(2) * np.imag(
                scipy.special.sph_harm(-order, degree, azimuths, colatitudes)
            )
        elif order == 0:
            matrix[i] = np.real(
                scipy.special.sph_harm(order, degree, azimuths, colatitudes)
            )
        else:
            matrix[i] = np.sqrt(2) * np.real(
                scipy.special.sph_harm(order, degree, azimuths, colatitudes)
            )

        if degree == order:
            degree += 2
            order = -degree
        else:
            order += 1

    return matrix.T


def remove_duplicate_endpoints(streamlines: npt.ArrayLike, step_size: float) -> list(npt.NDArray):
    """Removes duplicate points at the end of streamlines

    During the tractography process, streamlines are sometimes generated with
    duplicated point at the end. For example when the streamline exists the
    tracking mask. This function removes these points.

    Args:
        streamlines: The streamlines whose duplicates endpoints should be
            removed.
        step_size: The step size used to generate the streamlines.

    Returns:
        The new streamlines, with the duplicate endpoints removed.

    """
    d = np.power(np.diff(streamlines, axis=1), 2)
    sum_squared = d[..., 0] + d[..., 1] + d[..., 2]  # much faster than np.sum

    new_streamlines = []
    for streamline, sq in zip(streamlines, sum_squared):
        i = np.searchsorted(sq[::-1], step_size**2 / 2)
        new_streamlines.append(streamline[: len(streamline) - i])

    return new_streamlines


def apply_mask(image, image_affine, mask, mask_affine):
    """Apply the mask to the image in world space"""

    # Get all voxel coordinates of the image.
    x, y, z = np.meshgrid(
        range(image.shape[0]),
        range(image.shape[1]),
        range(image.shape[2]),
        indexing="ij",
    )
    image_voxels = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
    affine = np.dot(np.linalg.inv(mask_affine), image_affine)
    mask_voxels = nib.affines.apply_affine(affine, image_voxels).astype(int)
    mask_values = mask[*mask_voxels.T].reshape(image.shape[:3])

    return image * mask_values[..., None]
