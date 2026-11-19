import unittest

from tractography.algorithms import register, resolve, RegistryEntry, _REGISTRY
from tractography.algorithms.core import Algorithm
from tractography.algorithms.deterministic import tractogram as deterministic_tractogram
from tractography.algorithms.deterministic import Configuration as DeterministicConfiguration
from tractography.algorithms.diffusion import tractogram as diffusion_tractogram
from tractography.algorithms.probabilistic import tractogram as probabilistic_tractogram
from tractography.algorithms.transport import tractogram as transport_tractogram


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
                deterministic_tractogram,
            )

    def test_registered_tractograms(self):
        """Each algorithm should map to the correct tractogram function"""
        expected = {
            Algorithm.DETERMINISTIC: deterministic_tractogram,
            Algorithm.PROBABILISTIC: probabilistic_tractogram,
            Algorithm.DIFFUSION: diffusion_tractogram,
            Algorithm.TRANSPORT: transport_tractogram,
        }
        for algorithm, tractogram_fn in expected.items():
            with self.subTest(algorithm=algorithm):
                self.assertIs(_REGISTRY[algorithm].tractogram, tractogram_fn)

class TestResolve(unittest.TestCase):
    """Test the algorithm resolution mechanism"""

    def test_resolve_returns_registry_entry(self):
        """resolve() should return a RegistryEntry for known algorithms"""
        for algorithm in Algorithm:
            with self.subTest(algorithm=algorithm):
                entry = resolve(algorithm)
                self.assertIsInstance(entry, RegistryEntry)

    def test_resolve_tractogram(self):
        """resolve().tractogram should return the correct function"""
        expected = {
            Algorithm.DETERMINISTIC: deterministic_tractogram,
            Algorithm.PROBABILISTIC: probabilistic_tractogram,
            Algorithm.DIFFUSION: diffusion_tractogram,
            Algorithm.TRANSPORT: transport_tractogram,
        }
        for algorithm, tractogram_fn in expected.items():
            with self.subTest(algorithm=algorithm):
                self.assertIs(resolve(algorithm).tractogram, tractogram_fn)

    def test_resolve_unknown_raises(self):
        """resolve() should raise ValueError for an unregistered algorithm"""
        # Create a fake algorithm value not in the registry.
        fake_algorithm = "nonexistent"
        with self.assertRaises((ValueError, KeyError)):
            resolve(fake_algorithm)  # noqa