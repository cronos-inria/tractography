"""Implements the 'tractography mask' CLI"""

from pathlib import Path

import nimesh
import numpy as np

import tractography as tg


_DESCRIPTION = """
Generate tractography seeds from a triangular surface
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


def main(surface_path: Path, n_seeds: int, seeds_path: Path, **kwargs: dict):
    """Generate a seeds from the surface"""

    if n_seeds % tg.BATCH_SIZE != 0:
        raise SystemExit(
            f"Error: The number of seeds is not a multiple of the batch size ({tg.BATCH_SIZE})."
        )

    surface = nimesh.io.load(surface_path, hemisphere="lh", surface="white")
    seeds = tg.seeds.from_surface(surface, n_seeds)
    tg.seeds.save(seeds_path, seeds)


def add_parser(subparsers):
    """Add the surparser for the seeds subcommand"""
    subparser = subparsers.add_parser("seeds", description=_DESCRIPTION, help=_HELP)
    subparser.add_argument("surface_path", type=Path, help=_SURFACE_PATH_HELP)
    subparser.add_argument("n_seeds", type=int, help=_N_SEEDS_HELP)
    subparser.add_argument("seeds_path", type=str, help=_SEEDS_PATH_HELP)
    subparser.set_defaults(func=main)


if __name__ == "__main__":
    main()
