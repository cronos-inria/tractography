from pathlib import Path
import tomllib

import numpy as np
import pydantic


_DEFAULT_CONFIG_FILER = Path(__file__).parents[1] / "config.toml"


class Length(pydantic.BaseModel):
    minimum: pydantic.PositiveFloat
    maximum: pydantic.PositiveFloat


class Streamline(pydantic.BaseModel):
    length: Length


class Configuration(pydantic.BaseModel):
    batch_size: pydantic.PositiveInt
    step_size: pydantic.PositiveFloat
    max_angle: pydantic.PositiveInt
    streamline: Streamline

    @property
    def n_steps(self) -> int:
        return int(np.floor(self.streamline.length.maximum / self.step_size))

    @property
    def min_steps(self) -> int:
        return int(np.ceil(self.streamline.length.minimum / self.step_size))


def load():
    """Load the configuration from a file"""
    with open(_DEFAULT_CONFIG_FILER, "rb") as f:
        config = tomllib.load(f)

    return Configuration.model_validate(config)
