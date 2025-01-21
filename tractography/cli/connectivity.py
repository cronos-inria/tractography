"""Entry-point for the connectivity command of the CLI

This module implements the connectivity subcommand of the tractography CLI. It
allows the user to generate connectivity matrices from files.

"""

from enum import Enum
from pathlib import Path

import nibabel as nib
import nimesh
import numpy as np
import scipy

import tractography as tg


_DESCRIPTION = """
Perform diffusion magnetic resonance imaging tractography but save only the
connectivity matrix.
"""

_HELP = """
generate a structural connectivity matrix
"""

_ALGORITHM_HELP = """
the algorithm used for tractography
"""

_IMAGE_HELP = """
the filename of the image on which to perform tractography
"""

_SURFACE_PATH_HELP = """
the path to the surface used to generate the seeds
"""

_CONNECTIVITY_HELP = """
the filename of the generated connectivity matrix
"""

_MASK_HELP = """
the filename of the mask used for tractography
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


def main(
    algorithm: Algorithm,
    image_path: Path,
    surface_path: Path,
    connectivity_path: Path,
    step_size: int,
    n_steps: int,
    max_angle: float,
    **kwargs,
):
    """Entry-point of the tractography CLI"""

    # Merge left and right hemisphere.
    lh = nimesh.io.load(surface_path, hemisphere="lh", surface="white")
    rh = nimesh.io.load(surface_path, hemisphere="rh", surface="white")
    vertices = np.vstack((lh.vertices, rh.vertices))
    triangles = np.vstack((lh.triangles, rh.triangles + len(lh.vertices)))
    surface = nimesh.Mesh(vertices, triangles)

    # Merge the parcellations.
    lh_keys = lh.segmentations[0].keys
    lh_keys[lh_keys == -1] = 0
    rh_keys = rh.segmentations[0].keys
    rh_keys[rh_keys == -1] = 0
    keys = np.hstack((lh_keys, rh_keys + np.max(lh_keys) + 1))

    # Load the FOD image.
    nii = nib.load(image_path)
    data = nii.get_fdata()

    # Create the mask from the segmentation and apply it to the data.
    if "mask" in kwargs and kwargs["mask"] is not None:
        mask_nii = nib.load(kwargs["mask"])
        mask = mask_nii.get_fdata()
        data = tg.core.apply_mask(data, nii.affine, mask, mask_nii.affine)

    # The connectivity is represented as a dict.
    connectivity = {}

    # Track each label independently.
    labels = np.arange(np.max(keys) + 1)
    for label in labels:

        # Extract the triangles with this label.
        to_keep = keys == label
        if not np.any(to_keep):
            continue
        triangles = surface.triangles[np.all(to_keep[surface.triangles], axis=1)]

        # Generate the seeds for the subsurface associated with the label.
        subsurface = nimesh.Mesh(surface.vertices, triangles)
        seeds = tg.seeds.from_surface(subsurface, tg.BATCH_SIZE)

        if algorithm == Algorithm.DETERMINISTIC:
            streamlines = tg.algorithms.deterministic(
                data, nii.affine, seeds, step_size, n_steps, max_angle
            )
        else:
            streamlines = tg.algorithms.probabilistic(
                data, nii.affine, seeds, step_size, n_steps, max_angle
            )

        # Clean a bit.
        streamlines = tg.core.remove_duplicate_endpoints(streamlines, step_size)
        streamlines = [s for s in streamlines if len(s) != 0]

        # Add the streamlines to the connectivity.
        vertex_connectivity = map_vertices(surface.vertices, streamlines)
        connectivity |= {
            (label, k): np.sum(vertex_connectivity[keys == k]) for k in labels
        }

    # Save the connectivity as a matrix.
    matrix = np.zeros((len(labels),) * 2)
    for k, v in connectivity.items():
        matrix[*k] = v
        matrix[*k[::-1]] = v

    np.save(connectivity_path, matrix)


def add_parser(subparsers):
    """Add the surparser for the mask subcommand"""
    subparser = subparsers.add_parser(
        "connectivity", description=_DESCRIPTION, help=_HELP
    )
    subparser.add_argument(
        "algorithm", type=Algorithm, choices=list(Algorithm), help=_ALGORITHM_HELP
    )
    subparser.add_argument("image_path", type=Path, help=_IMAGE_HELP)
    subparser.add_argument("surface_path", type=Path, help=_SURFACE_PATH_HELP)
    subparser.add_argument("connectivity_path", type=Path, help=_CONNECTIVITY_HELP)
    subparser.add_argument("--mask", type=Path, help=_MASK_HELP)
    subparser.add_argument(
        "--number-of-steps",
        "-ns",
        dest="n_steps",
        type=int,
        default=2000,
        help=_N_STEPS,
    )
    subparser.add_argument(
        "--maximum-angle",
        "-ma",
        dest="max_angle",
        type=float,
        default=45,
        help=_MAX_ANGLE,
    )
    subparser.add_argument(
        "--step-size",
        "-ss",
        dest="step_size",
        type=float,
        default=0.25,
        help=_STEP_SIZE,
    )

    subparser.set_defaults(func=main)

    return subparser


def map_vertices(vertices, streamlines, distance_upper_bound: float = 2.0):

    # Find the neighbors of the streamline end points on the surface.
    vertices_tree = scipy.spatial.cKDTree(vertices)
    streamline_ends = np.array([s[-1, :] for s in streamlines])
    end_distances, end_vertices = vertices_tree.query(
        streamline_ends, distance_upper_bound=distance_upper_bound
    )

    # Prune streamlines that are too far from points.
    to_keep = end_distances < distance_upper_bound

    # Add them to the mapping.
    mapping = np.zeros((len(vertices),))
    for i, end in enumerate(end_vertices):
        if to_keep[i]:
            mapping[end] += 1

    return mapping


if __name__ == "__main__":
    main()
