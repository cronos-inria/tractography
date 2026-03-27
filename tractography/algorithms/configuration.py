from enum import Enum
from importlib.resources import files
import tomllib

import numpy as np
import pydantic


_DEFAULT_CONFIG_DIR = files("tractography.resources") / "config"


class Algorithm(Enum):
    """The available tractography algorithms"""

    DETERMINISTIC = "deterministic"
    PROBABILISTIC = "probabilistic"
    DIFFUSION = "diffusion"
    TRANSPORT = "transport"

    def __str__(self):
        return self.value
    

class LocalModel(Enum):
    """The supported local models"""

    DTI = "MODEL_DTI"
    SYMMETRIC_REAL_SPHERICAL_HARMONICS = "MODEL_SYMMETRIC_REAL_SPHERICAL_HARMONICS"

    def __str__(self):
        return self.value
    
    @classmethod
    def from_shape(cls, shape):
        if shape[-1] == 6:
            return cls.DTI
        elif shape[-1] == 45:
            return cls.SYMMETRIC_REAL_SPHERICAL_HARMONICS
        else:
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
        """Load the configuration from a file"""
        config_file = _DEFAULT_CONFIG_DIR / (str(algorithm) + ".toml")
        with config_file.open("rb") as f:
            config = tomllib.load(f)

        return cls.model_validate(config)
