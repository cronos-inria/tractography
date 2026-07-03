"""Implements the 'tractography seeds' command-line interface"""

from pathlib import Path

import nibabel as nib
import nimesh

import tractography as tg

_DESCRIPTION = """
Generate tractography seeds from a triangular surface or from
fibre orientation distributions.
"""

_HELP = """
generate tractography seeds
"""

_HELP_FROM_SURFACE = """
generate tractography seeds from a triangular surface
"""

_HELP_FROM_FOD = """
generate tractography seeds from fibre orientation distributions
"""

_DESCRIPTION_FROM_SURFACE = """
Generate tractography seeds from a triangular surface. The seeds are generated
uniformly over the triangles of the surface. The orientation of each seed is
uniform in a cone centered on the normal of the surface.
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

_DESCRIPTION_FROM_FOD = r"""
Generate tractography seeds from fibre orientation distributions (FOD).
The seeds are generated uniformly in space where the FOD are non-zero
and the orientations follow the local fibre orientation distribution.
"""

_FOD_PATH_HELP = """
the path to the fibre orientation distributions
"""


def from_surface(surface_path: Path, n_seeds: int, seeds_path: Path, **kwargs: dict):
    """Generate seeds from a surface"""

    surface = nimesh.io.load(surface_path, hemisphere="lh", surface="white")
    seeds = tg.seeds.from_surface(surface, n_seeds, **kwargs)
    tg.seeds.save(seeds_path, seeds)


def from_fod(fod_path: Path, n_seeds: int, seeds_path: Path, **kwargs: dict):
    """Generate seeds from fibre orientation distributions"""
    fod = nib.load(fod_path)
    seeds = tg.seeds.from_fod(fod, n_seeds, **kwargs)
    tg.seeds.save(seeds_path, seeds)


def add_parser(subparsers):
    """Add the surparser for the seeds subcommand"""

    # The seed subcommand itself has subcommands.
    subparser = subparsers.add_parser("seeds", description=_DESCRIPTION, help=_HELP)
    subsubparsers = subparser.add_subparsers()
    subsubparsers.required = True
    subsubparsers.dest = "subsubcommand"

    # Add the subsubparser to generate seeds from surfaces.
    subsubparser = subsubparsers.add_parser(
        "from-surface", description=_DESCRIPTION, help=_HELP_FROM_SURFACE
    )
    subsubparser.add_argument("surface_path", type=Path, help=_SURFACE_PATH_HELP)
    subsubparser.add_argument("n_seeds", type=int, help=_N_SEEDS_HELP)
    subsubparser.add_argument("seeds_path", type=Path, help=_SEEDS_PATH_HELP)
    subsubparser.add_argument(
        "--cone-angle", type=float, default=0.0, help=_CONE_ANGLE_HELP
    )
    subsubparser.set_defaults(func=from_surface)

    # Add the subsubparser to generate seeds from FODs.
    subsubparser = subsubparsers.add_parser(
        "from-fod", description=_DESCRIPTION_FROM_FOD, help=_HELP_FROM_FOD
    )
    subsubparser.add_argument("fod_path", type=Path, help=_FOD_PATH_HELP)
    subsubparser.add_argument("n_seeds", type=int, help=_N_SEEDS_HELP)
    subsubparser.add_argument("seeds_path", type=Path, help=_SEEDS_PATH_HELP)
    subsubparser.set_defaults(func=from_fod)
