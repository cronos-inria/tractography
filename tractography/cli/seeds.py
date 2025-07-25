"""Implements the 'tractography mask' CLI"""

from pathlib import Path

import nimesh
import numpy as np

import tractography as tg


_DESCRIPTION = """
Generate tractography seeds from a triangular surface

The seeds are generated randomly over the triangles of the
surface. The orientation of each seed is in a cone centered on
the normal of the surface.
"""

_HELP = """
generate tractography seeds
"""

_SURFACE_PATH_HELP = """
the path to the surface used to generate the seeds
"""

_N_SEEDS_HELP = """
the number of seeds to generate
"""

_SEEDS_PATH_HELP = """
the path to the generated seeds
"""

_CONE_ANGLE_HELP = """
the opening of the code in degrees
"""


def main(surface_path: Path, n_seeds: int, seeds_path: Path, **kwargs: dict):
    """Generate a seeds from the surface"""

    surface = nimesh.io.load(surface_path, hemisphere="lh", surface="white")
    seeds = tg.seeds.from_surface(surface, n_seeds, **kwargs)
    tg.seeds.save(seeds_path, seeds)


def add_parser(subparsers):
    """Add the surparser for the seeds subcommand"""
    subparser = subparsers.add_parser("seeds", description=_DESCRIPTION, help=_HELP)
    subparser.add_argument("surface_path", type=Path, help=_SURFACE_PATH_HELP)
    subparser.add_argument("n_seeds", type=int, help=_N_SEEDS_HELP)
    subparser.add_argument("seeds_path", type=Path, help=_SEEDS_PATH_HELP)
    subparser.add_argument("--cone-angle", type=float, default=0.0, help=_CONE_ANGLE_HELP)
    subparser.set_defaults(func=main)


if __name__ == "__main__":
    main()
