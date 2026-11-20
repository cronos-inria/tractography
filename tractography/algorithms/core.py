from __future__ import annotations

from enum import Enum


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


