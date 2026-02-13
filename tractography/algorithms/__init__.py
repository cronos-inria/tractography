from __future__ import annotations
from typing import Optional

from dataclasses import dataclass
from typing import Callable, Final

from .configuration import Algorithm

@dataclass(frozen=True)
class RegistryEntry:
    tracker: type
    histogram: Callable
    connectome: Optional[Callable] = None


_REGISTRY: Final[dict[Algorithm, RegistryEntry]] = {}


def register(algorithm: Algorithm, tracker: type, histogram: Callable, connectome: Optional[Callable] = None) -> None:
    """Register a tracker implementation class for an algorithm."""
    if algorithm in _REGISTRY:
        raise ValueError(f"Tracker already registered for algorithm: {algorithm}")
    _REGISTRY[algorithm] = RegistryEntry(tracker=tracker, histogram=histogram, connectome=connectome)


def resolve(algorithm: Algorithm) -> RegistryEntry:
    """Resolve the tracker class for the given algorithm."""
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

# Imported directly for testing. Should be removed.
from .deterministic import Deterministic
from .diffusion import Diffusion
from .probabilistic import Probabilistic
from .transport import Transport