import numpy as np
import numpy.typing as npt
import scipy


def cart2sph(x, y, z):
    """Convert Cartesian coordinates to spherical coordinates.

    Azimuth is in ``[0, 2π)`` and colatitude is in ``[0, π]``. At the
    origin, where both angles are undefined, this function returns zero for
    both angles and zero for the radius.
    """
    x, y, z = np.broadcast_arrays(x, y, z)
    radius = np.sqrt(x**2 + y**2 + z**2)
    nonzero = radius > 0
    cos_colatitude = np.divide(
        z,
        radius,
        out=np.zeros_like(radius, dtype=np.result_type(radius, np.float32)),
        where=nonzero,
    )
    azimuth = np.mod(np.arctan2(y, x), 2 * np.pi)
    colatitude = np.arccos(np.clip(cos_colatitude, -1.0, 1.0))
    azimuth = np.where(nonzero, azimuth, 0.0)
    colatitude = np.where(nonzero, colatitude, 0.0)
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
