import nibabel as nib
import numpy as np
import numpy.typing as npt
import scipy


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
    matrix_derivative = np.zeros((n_coefficients, len(colatitudes), 2))
    degree = 0
    order = 0
    for i in range(n_coefficients):
        if order < 0:
            m, d = scipy.special.sph_harm_y(
                degree, -order, colatitudes, azimuths, diff_n=1
            )
            matrix[i] = np.sqrt(2) * np.imag(m)
            matrix_derivative[i] = np.sqrt(2) * np.imag(d)
        elif order == 0:
            m, d = scipy.special.sph_harm_y(
                degree, order, colatitudes, azimuths, diff_n=1
            )
            matrix[i] = np.real(m)
            matrix_derivative[i] = np.real(d)
        else:
            m, d = scipy.special.sph_harm_y(
                degree, order, colatitudes, azimuths, diff_n=1
            )
            matrix[i] = np.sqrt(2) * np.real(m)
            matrix_derivative[i] = np.sqrt(2) * np.real(d)

        if degree == order:
            degree += 2
            order = -degree
        else:
            order += 1

    return matrix.T, matrix_derivative.T


def remove_duplicate_endpoints(
    streamlines: npt.ArrayLike, step_size: float
) -> list[npt.NDArray]:
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
        i = np.searchsorted(sq[::-1], step_size**2 / 4)
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


def fibonacci_sphere(samples=1000):
    """
    Generates points approximately uniformly distributed on a sphere using the
    Fibonacci lattice method.

    Args:
        samples (int): Number of points to generate.

    Returns:
       An array of shape (samples, 3) containing the (x, y, z) coordinates of
       points on the sphere.
    """
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    golden_angle = 2 * np.pi / phi

    indices = np.arange(samples)
    theta = golden_angle * indices  # Azimuthal angle
    z = 1 - (2 * indices / (samples - 1))  # z-coordinates (uniformly spaced in [-1, 1])
    radius = np.sqrt(1 - z**2)  # Compute radius for x-y plane

    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    return np.vstack((x, y, z)).T  # Return as (N, 3) array
