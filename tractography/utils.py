import nibabel as nib
import numpy as np
from numpy.typing import NDArray


def to_voxel(inv_affine: NDArray, location: NDArray) -> NDArray:
    """Change from world space to coordinate space

    The voxel coordinate (0, 0, 0) is assumed to be at the center
    of the voxel.

    Args:
        inv_affine: The affine to voxel space, i.e. the inverse of the
            transfrom in the file.
        location: The 3D coordinate to transform to voxel space.

    Returns:
        The voxel coordinates corresponding to the location.

    """
    return nib.affines.apply_affine(inv_affine, location)


def wrap(azimuth: float, colatitude: float) -> (float, float):
    """Wrap the azimuth and colatitude to keep them in valid intervals

    The azimuth is always in [0, 2pi[ and the colatitude is always in
    [0, pi[.

    Args:
        azimuth: The azimuthal angle (typically phi).
        theta: The colatitude angle (typically theta).

    Returns
        The two angles in the correct intervals.
    """

    # Fix wrapping of the colatitude.
    colatitude = np.mod(colatitude, 2 * np.pi)
    if colatitude >= np.pi:
        colatitude = np.pi - np.mod(colatitude, np.pi)
        azimuth += np.pi

    # Fix the wrapping of the azimuth.
    azimuth = np.mod(azimuth, 2 * np.pi)

    return azimuth, colatitude
