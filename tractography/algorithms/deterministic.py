import pydantic

from . import register
from .core import Algorithm, BaseConfiguration


class Configuration(BaseConfiguration):
    """Legacy CLI options for deterministic tracking."""

    maximum_angle: pydantic.PositiveInt

    @classmethod
    def load(cls):
        return super().load(Algorithm.DETERMINISTIC)


register(Algorithm.DETERMINISTIC, Configuration)
