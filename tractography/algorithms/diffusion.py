import pydantic

from . import register
from .core import Algorithm, BaseConfiguration


class Configuration(BaseConfiguration):
    """Legacy CLI options for diffusion tracking."""

    inverse_curvature: pydantic.PositiveFloat
    noise_variance: pydantic.PositiveFloat

    @classmethod
    def load(cls):
        return super().load(Algorithm.DIFFUSION)


register(Algorithm.DIFFUSION, Configuration)
