import unittest

import numpy as np

import tractography as tg


class TestRemoveDuplicateEndpoints(unittest.TestCase):
    def test_simple(self):
        """Test the simplest use-case"""

        step_size = 0.1
        directions = np.random.rand(10, 100, 3)
        directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
        streamlines = np.cumsum(directions * step_size, axis=1)

        # When there are no duplicates, nothing should happen.
        new_streamlines = tg.core.remove_duplicate_endpoints(streamlines, step_size)
        for a, b in zip(new_streamlines, streamlines):
            np.testing.assert_array_equal(a, b)

        # Add a single duplicate and see if it is removed.
        duplicate_streamlines = np.hstack((streamlines, streamlines[:, -1:, :]))
        new_streamlines = tg.core.remove_duplicate_endpoints(
            duplicate_streamlines, step_size
        )
        for a, b in zip(new_streamlines, streamlines):
            np.testing.assert_array_equal(a, b)


class TestISHTMTX(unittest.TestCase):
    def test_simple(self):
        """Test the simplest use-case"""

        azimuths, colatitudes = np.meshgrid(
            np.linspace(0, 2 * np.pi, 100), np.linspace(0, np.pi, 50)
        )
        azimuths = azimuths.ravel()
        colatitudes = colatitudes.ravel()
        n_coefficients = 15
        mtx, der = tg.core.ishtmtx(azimuths, colatitudes, n_coefficients)

        # import matplotlib.pyplot as plt
        # coeffs, _ = tg.core.ishtmtx([np.pi/3], [np.pi/3], n_coefficients)
        # coeffs = np.squeeze(coeffs)
        # values = np.dot(mtx, coeffs)
        # derv = np.dot(der, coeffs)
        # plt.scatter(azimuths, colatitudes, c=values)
        # plt.quiver(azimuths, colatitudes, derv[1], derv[0])
        # plt.colorbar()
        # plt.show()
