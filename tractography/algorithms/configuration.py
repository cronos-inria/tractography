from enum import Enum
from pathlib import Path
import tomllib

import numpy as np
import pydantic


_DEFAULT_CONFIG_DIR = Path(__file__).parents[2] / "config"


class Algorithm(Enum):
    """The available tractography algorithms"""

    DETERMINISTIC = "deterministic"
    PROBABILISTIC = "probabilistic"
    DIFFUSION = "diffusion"
    TRANSPORT = "transport"

    def __str__(self):
        return self.value


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
        """Load the configuration from a file"""
        with open(_DEFAULT_CONFIG_DIR / (str(algorithm) + ".toml"), "rb") as f:
            config = tomllib.load(f)

        return cls.model_validate(config)
