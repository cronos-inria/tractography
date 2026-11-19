import unittest

from tractography.algorithms import register, resolve, RegistryEntry, _REGISTRY
from tractography.algorithms.core import Algorithm
from tractography.algorithms.deterministic import Configuration as DeterministicConfiguration
from tractography.algorithms.diffusion import Configuration as DiffusionConfiguration
from tractography.algorithms.probabilistic import Configuration as ProbabilisticConfiguration
from tractography.algorithms.transport import Configuration as TransportConfiguration


class TestRegister(unittest.TestCase):
    """Test the algorithm registration mechanism"""

    def test_all_algorithms_registered(self):
        """Every Algorithm enum member should be registered"""
        for algorithm in Algorithm:
            with self.subTest(algorithm=algorithm):
                self.assertIn(algorithm, _REGISTRY)

    def test_registered_entries_are_registry_entries(self):
        """Each registered value should be a RegistryEntry"""
        for algorithm in Algorithm:
            with self.subTest(algorithm=algorithm):
                self.assertIsInstance(_REGISTRY[algorithm], RegistryEntry)

    def test_duplicate_registration_raises(self):
        """Registering the same algorithm twice should raise ValueError"""
        with self.assertRaises(ValueError):
            register(
                Algorithm.DETERMINISTIC,
                DeterministicConfiguration,
            )

    def test_registered_configurations(self):
        """Each algorithm should map to its configuration class"""
        expected = {
            Algorithm.DETERMINISTIC: DeterministicConfiguration,
            Algorithm.PROBABILISTIC: ProbabilisticConfiguration,
            Algorithm.DIFFUSION: DiffusionConfiguration,
            Algorithm.TRANSPORT: TransportConfiguration,
        }
        for algorithm, configuration in expected.items():
            with self.subTest(algorithm=algorithm):
                self.assertIs(_REGISTRY[algorithm].configuration, configuration)

class TestResolve(unittest.TestCase):
    """Test the algorithm resolution mechanism"""

    def test_resolve_returns_registry_entry(self):
        """resolve() should return a RegistryEntry for known algorithms"""
        for algorithm in Algorithm:
            with self.subTest(algorithm=algorithm):
                entry = resolve(algorithm)
                self.assertIsInstance(entry, RegistryEntry)

    def test_resolve_configuration(self):
        """resolve().configuration should return the correct class"""
        expected = {
            Algorithm.DETERMINISTIC: DeterministicConfiguration,
            Algorithm.PROBABILISTIC: ProbabilisticConfiguration,
            Algorithm.DIFFUSION: DiffusionConfiguration,
            Algorithm.TRANSPORT: TransportConfiguration,
        }
        for algorithm, configuration in expected.items():
            with self.subTest(algorithm=algorithm):
                self.assertIs(resolve(algorithm).configuration, configuration)

    def test_resolve_unknown_raises(self):
        """resolve() should raise ValueError for an unregistered algorithm"""
        # Create a fake algorithm value not in the registry.
        fake_algorithm = "nonexistent"
        with self.assertRaises((ValueError, KeyError)):
            resolve(fake_algorithm)  # noqa