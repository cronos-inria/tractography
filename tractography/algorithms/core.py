from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from importlib.resources import files
from typing import Any
import tomllib

import numpy as np
import pydantic

from .. import core as tractography_core


_DEFAULT_CONFIG_DIR = files("tractography.resources") / "config"


class Algorithm(Enum):
    """The available tractography algorithms."""

    DETERMINISTIC = "deterministic"
    PROBABILISTIC = "probabilistic"
    DIFFUSION = "diffusion"
    TRANSPORT = "transport"

    def __str__(self):
        return self.value


class LocalModel(Enum):
    """The supported local models."""

    DTI = "MODEL_DTI"
    SYMMETRIC_REAL_SPHERICAL_HARMONICS = "MODEL_SYMMETRIC_REAL_SPHERICAL_HARMONICS"

    def __str__(self):
        return self.value

    @classmethod
    def from_shape(cls, shape):
        if shape[-1] == 6:
            return cls.DTI
        if shape[-1] == 45:
            return cls.SYMMETRIC_REAL_SPHERICAL_HARMONICS
        raise ValueError(
            "For now, only DTI and fODF with 45 coefficients are supported (lmax=8)."
        )


class Length(pydantic.BaseModel):
    minimum: pydantic.PositiveFloat
    maximum: pydantic.PositiveFloat


class Streamline(pydantic.BaseModel):
    length: Length


class BaseConfiguration(pydantic.BaseModel):
    algorithm: Algorithm
    batch_size: pydantic.PositiveInt
    step_size: pydantic.PositiveFloat
    save_at: pydantic.PositiveFloat
    streamline: Streamline

    @property
    def n_steps(self) -> int:
        return int(np.floor(self.streamline.length.maximum / self.save_at))

    @property
    def min_n_points(self) -> int:
        return int(np.ceil(self.streamline.length.minimum / self.save_at))

    @property
    def min_n_steps(self) -> int:
        return int(np.ceil(self.streamline.length.minimum / self.step_size))

    @classmethod
    def load(cls, algorithm):
        """Load the configuration from a file."""
        config_file = _DEFAULT_CONFIG_DIR / (str(algorithm) + ".toml")
        with config_file.open("rb") as f:
            config = tomllib.load(f)

        return cls.model_validate(config)


@dataclass
class BaseCache:
    """Common cache fields shared by all tractography algorithms."""

    fod_shape: tuple[int, ...] | None = None
    n_streamlines: int | None = None
    n_steps: int | None = None

    seeds: Any = None
    streamlines: Any = None
    lengths: Any = None
    program: Any = None


def cache_needs_rebuild(
    cache: BaseCache | None,
    cache_type: type[BaseCache],
    fod_shape: tuple[int, ...],
    n_streamlines: int,
    n_steps: int,
) -> bool:
    """Return True when cache is missing, invalid, uninitialized, or incompatible."""

    if cache is None or not isinstance(cache, cache_type):
        return True

    if (
        cache.fod_shape != fod_shape
        or cache.n_streamlines != n_streamlines
        or cache.n_steps != n_steps
    ):
        return True

    # All cache dataclass fields must be initialized for safe reuse.
    for field in fields(cache):
        if getattr(cache, field.name) is None:
            return True

    return False


def discretize_fod(odf: np.ndarray, n_directions: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Discretize FOD data using spherical harmonics at direction vertices.

    Generates a uniform distribution of direction vectors on the sphere using
    fibonacci sampling, then converts spherical harmonics coefficients to 
    1D probability mass functions evaluated at those directions using the 
    inverse SH transform.

    Args:
        odf: FOD data array of shape (nx, ny, nz, n_coefficients).
        n_directions: Number of direction vertices to generate (default 400).

    Returns:
        Tuple of (vertices, fod_values):
            vertices: Array of shape (n_directions, 3) with unit vectors on the sphere.
            fod_values: Discretized FOD array of shape (nx, ny, nz, n_directions), clipped to [0, inf).
    """
    vertices = tractography_core.fibonacci_sphere(n_directions)
    n_coefficients = odf.shape[-1]
    azimuths, colatitudes, _ = tractography_core.cart2sph(*vertices.T)
    matrix, _ = tractography_core.ishtmtx(azimuths, colatitudes, n_coefficients)
    fod_values = np.maximum(np.dot(odf.reshape((-1, n_coefficients)), matrix.T), 0)
    fod_values = fod_values.reshape((*odf.shape[:3], -1))
    return vertices, fod_values
