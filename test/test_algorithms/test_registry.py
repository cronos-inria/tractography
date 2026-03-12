import unittest

from tractography.algorithms import register, resolve, RegistryEntry, _REGISTRY
from tractography.algorithms.configuration import Algorithm
from tractography.algorithms.deterministic import connectome as deterministic_connectome
from tractography.algorithms.deterministic import Deterministic, histogram as deterministic_histogram
from tractography.algorithms.diffusion import connectome as diffusion_connectome
from tractography.algorithms.diffusion import Diffusion, histogram as diffusion_histogram
from tractography.algorithms.probabilistic import connectome as probabilistic_connectome
from tractography.algorithms.probabilistic import Probabilistic, histogram as probabilistic_histogram
from tractography.algorithms.transport import connectome as transport_connectome
from tractography.algorithms.transport import Transport, histogram as transport_histogram


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
            register(Algorithm.DETERMINISTIC, Deterministic, deterministic_histogram)

    def test_registered_trackers(self):
        """Each algorithm should map to the correct tracker class"""
        expected = {
            Algorithm.DETERMINISTIC: Deterministic,
            Algorithm.PROBABILISTIC: Probabilistic,
            Algorithm.DIFFUSION: Diffusion,
            Algorithm.TRANSPORT: Transport,
        }
        for algorithm, tracker_cls in expected.items():
            with self.subTest(algorithm=algorithm):
                self.assertIs(_REGISTRY[algorithm].tracker, tracker_cls)

    def test_registered_histograms(self):
        """Each algorithm should map to the correct histogram function"""
        expected = {
            Algorithm.DETERMINISTIC: deterministic_histogram,
            Algorithm.PROBABILISTIC: probabilistic_histogram,
            Algorithm.DIFFUSION: diffusion_histogram,
            Algorithm.TRANSPORT: transport_histogram,
        }
        for algorithm, histogram_fn in expected.items():
            with self.subTest(algorithm=algorithm):
                self.assertIs(_REGISTRY[algorithm].histogram, histogram_fn)

    def test_registered_connectomes(self):
        """Algorithms with connectome support should expose the correct function"""
        expected = {
            Algorithm.DETERMINISTIC: deterministic_connectome,
            Algorithm.PROBABILISTIC: probabilistic_connectome,
            Algorithm.DIFFUSION: diffusion_connectome,
            Algorithm.TRANSPORT: transport_connectome,
        }
        for algorithm, connectome_fn in expected.items():
            with self.subTest(algorithm=algorithm):
                self.assertIs(_REGISTRY[algorithm].connectome, connectome_fn)


class TestResolve(unittest.TestCase):
    """Test the algorithm resolution mechanism"""

    def test_resolve_returns_registry_entry(self):
        """resolve() should return a RegistryEntry for known algorithms"""
        for algorithm in Algorithm:
            with self.subTest(algorithm=algorithm):
                entry = resolve(algorithm)
                self.assertIsInstance(entry, RegistryEntry)

    def test_resolve_tracker(self):
        """resolve().tracker should return the correct class"""
        expected = {
            Algorithm.DETERMINISTIC: Deterministic,
            Algorithm.PROBABILISTIC: Probabilistic,
            Algorithm.DIFFUSION: Diffusion,
            Algorithm.TRANSPORT: Transport,
        }
        for algorithm, tracker_cls in expected.items():
            with self.subTest(algorithm=algorithm):
                self.assertIs(resolve(algorithm).tracker, tracker_cls)

    def test_resolve_histogram(self):
        """resolve().histogram should return the correct function"""
        expected = {
            Algorithm.DETERMINISTIC: deterministic_histogram,
            Algorithm.PROBABILISTIC: probabilistic_histogram,
            Algorithm.DIFFUSION: diffusion_histogram,
            Algorithm.TRANSPORT: transport_histogram,
        }
        for algorithm, histogram_fn in expected.items():
            with self.subTest(algorithm=algorithm):
                self.assertIs(resolve(algorithm).histogram, histogram_fn)

    def test_resolve_connectome(self):
        """resolve().connectome should return the correct function"""
        expected = {
            Algorithm.DETERMINISTIC: deterministic_connectome,
            Algorithm.PROBABILISTIC: probabilistic_connectome,
            Algorithm.DIFFUSION: diffusion_connectome,
            Algorithm.TRANSPORT: transport_connectome,
        }
        for algorithm, connectome_fn in expected.items():
            with self.subTest(algorithm=algorithm):
                self.assertIs(resolve(algorithm).connectome, connectome_fn)

    def test_resolve_unknown_raises(self):
        """resolve() should raise ValueError for an unregistered algorithm"""
        # Create a fake algorithm value not in the registry.
        fake_algorithm = "nonexistent"
        with self.assertRaises((ValueError, KeyError)):
            resolve(fake_algorithm)  # noqa