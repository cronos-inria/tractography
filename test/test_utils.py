import unittest

import numpy as np

import tractography as tg


class TestNormalizeODF(unittest.TestCase):
    """Test the tg.utils.normalize_odf function"""

    def test_simple(self):

        coeffs = np.random.rand(3, 3, 3, 45)
        normalized = tg.utils.normalize_odf(coeffs)
        np.testing.assert_array_almost_equal(
            normalized[..., 0] * np.sqrt(4 * np.pi), np.ones((3, 3, 3))
        )


class TestWrap(unittest.TestCase):
    """Test angle wrapping"""

    def test_simple(self):

        # Valid angles should not change.
        azimuths = np.linspace(0, 2 * np.pi - 2 * np.pi / 20, 20)
        colatitudes = np.linspace(0, np.pi - np.pi / 20, 20)
        na, nc = zip(*[tg.utils.wrap(a, c) for a, c in zip(azimuths, colatitudes)])
        np.testing.assert_array_almost_equal(np.array(na), azimuths)
        np.testing.assert_array_almost_equal(np.array(nc), colatitudes)

        # Test wrapping in the colatitude.
        tc = colatitudes + 2 * np.pi
        na, nc = zip(*[tg.utils.wrap(a, c) for a, c in zip(azimuths, tc)])
        np.testing.assert_array_almost_equal(np.array(na), azimuths)
        np.testing.assert_array_almost_equal(np.array(nc), colatitudes)
        tc = colatitudes - 2 * np.pi
        na, nc = zip(*[tg.utils.wrap(a, c) for a, c in zip(azimuths, tc)])
        np.testing.assert_array_almost_equal(np.array(na), azimuths)
        np.testing.assert_array_almost_equal(np.array(nc), colatitudes)

        # Test wrapping in the azimuth.
        ta = azimuths + 2 * np.pi
        na, nc = zip(*[tg.utils.wrap(a, c) for a, c in zip(ta, colatitudes)])
        np.testing.assert_array_almost_equal(np.array(na), azimuths)
        np.testing.assert_array_almost_equal(np.array(nc), colatitudes)
        ta = azimuths - 2 * np.pi
        na, nc = zip(*[tg.utils.wrap(a, c) for a, c in zip(ta, colatitudes)])
        np.testing.assert_array_almost_equal(np.array(na), azimuths)
        np.testing.assert_array_almost_equal(np.array(nc), colatitudes)

        # Test wrapping both angles.
        ta = azimuths - 2 * np.pi
        tc = colatitudes + 2 * np.pi
        na, nc = zip(*[tg.utils.wrap(a, c) for a, c in zip(ta, tc)])
        np.testing.assert_array_almost_equal(np.array(na), azimuths)
        np.testing.assert_array_almost_equal(np.array(nc), colatitudes)
        ta = azimuths + 2 * np.pi
        tc = colatitudes - 2 * np.pi
        na, nc = zip(*[tg.utils.wrap(a, c) for a, c in zip(ta, tc)])
        np.testing.assert_array_almost_equal(np.array(na), azimuths)
        np.testing.assert_array_almost_equal(np.array(nc), colatitudes)
