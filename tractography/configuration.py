from pathlib import Path
import tomllib

import numpy as np
import pydantic


_DEFAULT_CONFIG_FILE = Path(__file__).parents[1] / "config.toml"


class Length(pydantic.BaseModel):
    minimum: pydantic.PositiveFloat
    maximum: pydantic.PositiveFloat


class Streamline(pydantic.BaseModel):
    length: Length


class Algorithm(pydantic.BaseModel): ...


class Deterministic(Algorithm):
    maximum_angle: pydantic.PositiveFloat


class Probabilistic(Algorithm):
    maximum_angle: pydantic.PositiveFloat


class Boltzmann(Algorithm):
    acceleration_factor: pydantic.PositiveFloat


class FACT(Algorithm):
    maximum_angle: pydantic.PositiveFloat


class Algorithms(pydantic.BaseModel):
    boltzmann: Boltzmann
    deterministic: Deterministic
    fact: FACT
    probabilistic: Probabilistic


class Configuration(pydantic.BaseModel):
    batch_size: pydantic.PositiveInt
    step_size: pydantic.PositiveFloat
    max_angle: pydantic.PositiveInt
    streamline: Streamline
    algorithms: Algorithms

    @property
    def n_steps(self) -> int:
        return int(np.floor(self.streamline.length.maximum / self.step_size))

    @property
    def min_steps(self) -> int:
        return int(np.ceil(self.streamline.length.minimum / self.step_size))


def load():
    """Load the configuration from a file"""
    with open(_DEFAULT_CONFIG_FILE, "rb") as f:
        config = tomllib.load(f)

    return Configuration.model_validate(config)
