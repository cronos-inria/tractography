import unittest
from pathlib import Path

import numpy as np

import tractography as tg


_DATA_DIR = Path(__file__).parents[1] / "data"


class TestFACT(unittest.TestCase):
    """Test FACT tractography algorithm"""

    def test_opencl_simple(self):
        """Simple test for the OpenCL implementation"""

        # All peaks pointing up. The streamlines should only move up.
        peaks = np.zeros((32, 32, 32, 6))
        peaks[..., 2] = 1
        peaks[..., 5] = 1
        seed_mask = np.zeros((32, 32, 32))
        seed_mask[1:31, 1:31, 1:31] = 1
        seeds = tg.seeds.from_mask(seed_mask, np.eye(4), 100)

        config = tg.configuration.load()
        algorithm = tg.algorithms.FACT(peaks, np.eye(4), len(seeds), config)
        streamlines = algorithm.run(seeds)
        for streamline, seed in zip(streamlines, seeds):
            move_direction = streamline[-1] - streamline[0]

            # If the seeds orientation is not aligned with the peaks, we go nowhere.
            if abs(np.dot(seed.orientation, [0, 0, 1])) < np.cos(
                np.deg2rad(config.algorithms.fact.maximum_angle)
            ):
                self.assertAlmostEqual(np.sum(move_direction), 0)
            else:
                move_direction /= np.linalg.norm(move_direction)
                self.assertAlmostEqual(abs(np.dot(move_direction, [0, 0, 1])), 1)
