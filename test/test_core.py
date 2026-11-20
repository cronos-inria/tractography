import numpy as np

import tractography as tg


def test_cart2sph_normalizes_and_handles_origin():
    points = np.array(
        [
            [2.0, 0.0, 0.0],
            [0.0, -3.0, 0.0],
            [0.0, 0.0, -4.0],
            [0.0, 0.0, 0.0],
        ]
    )

    azimuth, colatitude, radius = tg.core.cart2sph(*points.T)

    np.testing.assert_allclose(azimuth, [0.0, 1.5 * np.pi, 0.0, 0.0])
    np.testing.assert_allclose(colatitude, [0.5 * np.pi, 0.5 * np.pi, np.pi, 0.0])
    np.testing.assert_allclose(radius, [2.0, 3.0, 4.0, 0.0])


def test_ishtmtx_simple():
    """Test the simplest use-case"""

    azimuths, colatitudes = np.meshgrid(
        np.linspace(0, 2 * np.pi, 100), np.linspace(0, np.pi, 50)
    )
    azimuths = azimuths.ravel()
    colatitudes = colatitudes.ravel()
    n_coefficients = 15
    mtx, der = tg.core.ishtmtx(azimuths, colatitudes, n_coefficients)
