"""Entry-point for the Command Line Interface

This module implements the main CLI for the tractography package. It
allows the user to perform tractography from files.

"""

import argparse
from enum import Enum
from pathlib import Path

from .. import deterministic


_DESCRIPTION = """
Perform diffusion magnetic resonance imaging tractography.
"""

_ALGORITHM_HELP = """
the algorithm used for tractography
"""

_IMAGE_HELP = """
the filename of the image on which to perform tractography
"""

_TRACTOGRAM_HELP = """
the filename of the generated tractogram
"""

_N_STREAMLINES_HELP = """
the number of streamlines to generate
"""

_N_STEPS = """
the number of steps for each streamline
"""

_MAX_ANGLE = """
the maximum angle between consecutive steps
"""

_STEP_SIZE = """
the size of each step
"""


class Algorithm(Enum):
    """The various tractography algorithms"""

    DETERMINISTIC = "det"
    PROBABILISTIC = "prob"

    def __str__(self):
        return self.value


def main():
    """Entry-point of the tractography CLI"""
    args = parse_args()
    print(args)


def parse_args():
    """Parse CLI arguments"""

    parser = argparse.ArgumentParser(description=_DESCRIPTION)
    parser.add_argument(
        "algorithm", type=Algorithm, choices=list(Algorithm), help=_ALGORITHM_HELP
    )
    parser.add_argument("image", type=Path, help=_IMAGE_HELP)
    parser.add_argument("output", type=Path, help=_TRACTOGRAM_HELP)
    parser.add_argument(
        "--n_streamlines", type=int, default=10000, help=_N_STREAMLINES_HELP
    )
    parser.add_argument("--n_steps", type=int, default=100, help=_N_STEPS)
    parser.add_argument("--max_angle", type=float, default=45, help=_MAX_ANGLE)
    parser.add_argument("--step_size", type=float, default=0.1, help=_STEP_SIZE)

    return parser.parse_args()


if __name__ == "__main__":
    main()
