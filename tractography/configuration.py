from .algorithms.deterministic import Configuration as DeterministicConfiguration
from .algorithms.probabilistic import Configuration as ProbabilisticConfiguration
from .algorithms.transport import Configuration as TransportConfiguration
from .algorithms.diffusion import Configuration as DiffusionConfiguration
from .algorithms.configuration import Algorithm, BaseConfiguration


def load(algorithm: Algorithm) -> BaseConfiguration:
    """Loads the default configuration for the specified algorithm

    The default configuration is loaded from the file in config/<algoname>.toml
    where <algoname> is deterministic, probabilistic, transport, or diffusion.

    Args:
        algorithm: The algorithm for which to load the configuration.

    Returns:
        A subclass of BaseConfiguration that contains the default parameter
        values for the specified algorithm.

    """
    if algorithm == Algorithm.DETERMINISTIC:
        config = DeterministicConfiguration
    elif algorithm == Algorithm.PROBABILISTIC:
        config = ProbabilisticConfiguration
    elif algorithm == Algorithm.DIFFUSION:
        config = DiffusionConfiguration
    elif algorithm == Algorithm.TRANSPORT:
        config = TransportConfiguration
    else:
        raise ValueError(f"No algorithm associated with {algorithm}.")

    return config.load()
