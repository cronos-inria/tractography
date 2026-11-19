import pydantic

from . import register
from .core import Algorithm, BaseConfiguration


class Configuration(BaseConfiguration):
    """Legacy CLI options for probabilistic tracking."""

    maximum_angle: pydantic.PositiveInt

    @classmethod
    def load(cls):
        return super().load(Algorithm.PROBABILISTIC)


register(Algorithm.PROBABILISTIC, Configuration)
