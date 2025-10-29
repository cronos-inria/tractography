import unittest


import tractography as tg


class TestConfiguration(unittest.TestCase):
    """Test the configuration module"""

    def test_load(self):
        """Test the tg.configuration.load function"""

        options = tg.configuration.load(tg.Algorithm.TRANSPORT)
        self.assertIsInstance(options, tg.algorithms.transport.Configuration)

        options = tg.configuration.load(tg.Algorithm.DIFFUSION)
        self.assertIsInstance(options, tg.algorithms.diffusion.Configuration)

        options = tg.configuration.load(tg.Algorithm.PROBABILISTIC)
        self.assertIsInstance(options, tg.algorithms.probabilistic.Configuration)

        options = tg.configuration.load(tg.Algorithm.DETERMINISTIC)
        self.assertIsInstance(options, tg.algorithms.deterministic.Configuration)
