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
