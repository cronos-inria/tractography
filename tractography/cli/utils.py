import argparse

import tractography as tg

_MAX_ANGLE = """
the maximum angle between consecutive steps
"""


def add_tractography_config(parser: argparse.ArgumentParser):
    """Add the common tractography configuration to a parser"""

    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=float,
    )

    # Global properties.
    parser.add_argument(
        "--step-size",
        dest="step_size",
        type=float,
    )

    parser.add_argument(
        "--maximum-angle",
        dest="max_angle",
        type=float,
        help=_MAX_ANGLE,
    )

    # Streamline properties.
    parser.add_argument(
        "--streamline-minimum-length",
        dest="min_length",
        type=float,
    )
    parser.add_argument(
        "--streamline-maximum-length",
        dest="max_length",
        type=float,
    )


def set_tractography_config(
    config: tg.algorithms.configuration.BaseConfiguration, kwargs
):
    """Map CLI arguments back to the configuration"""

    if "batch_size" in kwargs and kwargs["batch_size"] is not None:
        config.batch_size = kwargs["batch_size"]

    if "step_size" in kwargs and kwargs["step_size"] is not None:
        config.step_size = kwargs["step_size"]

    if "max_angle" in kwargs and kwargs["max_angle"] is not None:
        config.maximum_angle = kwargs["max_angle"]

    if "min_length" in kwargs and kwargs["min_length"] is not None:
        config.streamline.length.minimum = kwargs["min_length"]

    if "max_length" in kwargs and kwargs["max_length"] is not None:
        config.streamline.length.maximum = kwargs["max_length"]
