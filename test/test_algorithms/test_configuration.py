import unittest


import tractography as tg


class TestConfiguration(unittest.TestCase):
    """Test the tg.algorithms.configuration module"""

    def test_load(self):
        """Test the tg.configuration.load function"""

        options = tg.algorithms.transport.Configuration.load()
        self.assertEqual(options.save_at, 0.25)
        self.assertEqual(options.step_size, 1e-3)
        self.assertEqual(options.streamline.length.minimum, 10.0)
        self.assertEqual(options.streamline.length.maximum, 300.0)

        options = tg.algorithms.diffusion.Configuration.load()
        self.assertEqual(options.save_at, 0.25)
        self.assertEqual(options.step_size, 1e-3)
        self.assertEqual(options.inverse_curvature, 0.5)
        self.assertEqual(options.noise_variance, 2.0)
        self.assertEqual(options.streamline.length.minimum, 10.0)
        self.assertEqual(options.streamline.length.maximum, 300.0)

        options = tg.algorithms.deterministic.Configuration.load()
        self.assertEqual(options.save_at, 0.25)
        self.assertEqual(options.step_size, 0.25)
        self.assertEqual(options.maximum_angle, 30)
        self.assertEqual(options.streamline.length.minimum, 10.0)
        self.assertEqual(options.streamline.length.maximum, 300.0)

        options = tg.algorithms.probabilistic.Configuration.load()
        self.assertEqual(options.save_at, 0.25)
        self.assertEqual(options.step_size, 0.25)
        self.assertEqual(options.maximum_angle, 20)
        self.assertEqual(options.streamline.length.minimum, 10.0)
        self.assertEqual(options.streamline.length.maximum, 300.0)
