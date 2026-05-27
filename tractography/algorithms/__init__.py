from __future__ import annotations
from typing import Any, Optional, Protocol, Tuple, TypeVar, runtime_checkable
from dataclasses import dataclass
from typing import Callable, Final

from nibabel.nifti1 import Nifti1Image
from nibabel.streamlines.tractogram import Tractogram

from .core import Algorithm, BaseConfiguration

# TypeVar for cache that allows flexible algorithm-specific cache types
TCache = TypeVar("TCache")


@runtime_checkable
class TractogramCallable(Protocol[TCache]):
    """Callable protocol for tractogram functions.
    
    A tractogram function takes FOD/ODF data, seed points, configuration,
    and an optional cache, returning streamlines and the cache for reuse.
    """

    def __call__(
        self,
        fod: Nifti1Image,
        seeds: list,  # list[Seed], but avoid circular import
        config: BaseConfiguration,
        cache: TCache | None = None,
    ) -> Tuple[Tractogram, TCache]: ...

@dataclass(frozen=True)
class RegistryEntry:
    """Registry entry for a tractography algorithm implementation.
    
    Attributes:
        configuration: Configuration class for the algorithm.
        tractogram: Tractogram function conforming to TractogramCallable protocol.
            Signature: (fod: Nifti1Image, seeds: list[Seed], config, cache=None) -> (Tractogram, cache)
        histogram: Histogram function for generating streamline distributions.
        connectome: Optional connectome function for generating connectivity matrices.
    """
    configuration: type[BaseConfiguration]
    tractogram: TractogramCallable[Any]
    histogram: Callable
    connectome: Optional[Callable] = None


_REGISTRY: Final[dict[Algorithm, RegistryEntry]] = {}


def register(
    algorithm: Algorithm,
    configuration: type[BaseConfiguration],
    tractogram: TractogramCallable[Any],
    histogram: Callable,
    connectome: Optional[Callable] = None,
) -> None:
    """Register algorithm implementations.
    
    Args:
        algorithm: The algorithm enum value.
        configuration: Configuration class for the algorithm.
        tractogram: Tractogram function conforming to TractogramCallable protocol.
        histogram: Callable for histogram computation.
        connectome: Optional callable for connectome computation.
    """
    if algorithm in _REGISTRY:
        raise ValueError(f"Tracker already registered for algorithm: {algorithm}")
    _REGISTRY[algorithm] = RegistryEntry(
        configuration=configuration,
        tractogram=tractogram,
        histogram=histogram,
        connectome=connectome,
    )


def resolve(algorithm: Algorithm) -> RegistryEntry:
    """Resolve implementations for the given algorithm."""
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