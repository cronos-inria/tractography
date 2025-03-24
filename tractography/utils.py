import numpy as np


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
