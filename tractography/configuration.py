from . import algorithms
from .algorithms.configuration import Algorithm, BaseConfiguration


def load(algorithm: Algorithm) -> BaseConfiguration:
    """Loads the default configuration for the specified algorithm

    The default configuration is loaded from the registry entry associated with
    the requested algorithm.

    Args:
        algorithm: The algorithm for which to load the configuration.

    Returns:
        A subclass of BaseConfiguration that contains the default parameter
        values for the specified algorithm.

    """
    return algorithms.resolve(algorithm).configuration.load()
