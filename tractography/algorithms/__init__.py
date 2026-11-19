from __future__ import annotations
from typing import Final
from dataclasses import dataclass

from .core import Algorithm, BaseConfiguration

@dataclass(frozen=True)
class RegistryEntry:
    """Registry entry for an algorithm's configuration type.
    
    Attributes:
        configuration: Configuration class for the algorithm.
    """
    configuration: type[BaseConfiguration]


_REGISTRY: Final[dict[Algorithm, RegistryEntry]] = {}


def register(
    algorithm: Algorithm,
    configuration: type[BaseConfiguration],
) -> None:
    """Register an algorithm's legacy CLI configuration type.
    
    Args:
        algorithm: The algorithm enum value.
        configuration: Configuration class for the algorithm.
    """
    if algorithm in _REGISTRY:
        raise ValueError(f"Tracker already registered for algorithm: {algorithm}")
    _REGISTRY[algorithm] = RegistryEntry(configuration=configuration)


def resolve(algorithm: Algorithm) -> RegistryEntry:
    """Resolve configuration metadata for the given algorithm."""
    try:
        return _REGISTRY[algorithm]
    except KeyError as e:
        raise ValueError(
            f"Nothing registered for algorithm: {algorithm}."
        ) from e


# Import modules for side effects (self-registration).
from . import deterministic as deterministic  # noqa: F401,E402
from . import diffusion as diffusion  # noqa: F401,E402
from . import probabilistic as probabilistic  # noqa: F401,E402
from . import transport as transport  # noqa: F401,E402