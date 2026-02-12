from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from .configuration import Algorithm

@dataclass(frozen=True)
class RegistryEntry:
    tracker: type


_REGISTRY: Final[dict[Algorithm, RegistryEntry]] = {}


def register(algorithm: Algorithm, tracker: type) -> None:
    """Register a tracker implementation class for an algorithm."""
    if algorithm in _REGISTRY:
        raise ValueError(f"Tracker already registered for algorithm: {algorithm}")
    _REGISTRY[algorithm] = RegistryEntry(tracker=tracker)


def resolve(algorithm: Algorithm) -> type:
    """Resolve the tracker class for the given algorithm."""
    try:
        return _REGISTRY[algorithm].tracker
    except KeyError as e:
        raise ValueError(
            f"No tracker registered for algorithm: {algorithm}."
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