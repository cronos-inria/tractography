import pydantic

from . import register
from .core import Algorithm, BaseConfiguration


class Configuration(BaseConfiguration):
    """Legacy CLI options for transport tracking."""

    inverse_curvature: pydantic.PositiveFloat

    @classmethod
    def load(cls):
        return super().load(Algorithm.TRANSPORT)


register(Algorithm.TRANSPORT, Configuration)
