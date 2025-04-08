"""Generate seeds for tractography"""

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import nimesh
import numpy as np
import numpy.typing as npt

import tractography as tg


@dataclass
class Seed:
    """A single streamline seed"""

    location: npt.ArrayLike
    orientation: npt.ArrayLike


def from_surface(surface: nimesh.Mesh, n_seeds: int) -> list[Seed]:
    """Generate seeds from a surface

    The seeds are generated randomly over the triangles of the
    surface. The orientation of each seed is opposite to the normal
    of the surface.

    It is assumed the ordering of the triangle indices follows the
    right-hand rule and the normals point outward.

    Args:
        surface: The surface from which to generate the seeds, for example the
            lh.white from FreeSurfer.
        n_seeds: The number of seeds to generate.

    Return:
        A list of `n_seeds` seeds suitable for tractography.

    """

    # First choose a triangle for every seed and generate a random point
    # inside that triangle.
    vertices, triangles = surface.vertices, surface.triangles
    indices = np.random.randint(len(triangles), size=n_seeds)
    barycentric = [_random_barycentric_coordinates() for _ in range(n_seeds)]
    triangle_vertices = [vertices[triangles[i]] for i in indices]
    locations = [np.dot(b, v) for v, b in zip(triangle_vertices, barycentric)]

    # The orientations of the seeds are just triangle normals.
    normals = _triangle_normals(vertices, triangles)
    orientations = -normals[indices]

    return [Seed(el, n) for el, n in zip(locations, orientations)]


def from_mask(mask: npt.NDArray, affine: npt.NDArray, n_seeds: int) -> list[Seed]:
    """Generate seeds from a 3D mask

    The seeds are generated randomly inside voxels with non-zero
    values in the mask. The orientations are random.

    Args:
        mask: The numpy array containing the mask.
        affine: The affine transform to native space.
        n_seeds: The number of seeds to generate.

    Return:
        A list of `n_seeds` seeds suitable for tractography

    """

    # Get the non-zero voxels.
    voxels = np.array(list(zip(*np.nonzero(mask))))
    indices = np.random.randint(len(voxels), size=n_seeds)
    locations_voxel = voxels[indices] + np.random.rand(n_seeds, 3) - [0.5, 0.5, 0.5]
    locations = nib.affines.apply_affine(affine, locations_voxel)
    orientations = np.random.randn(n_seeds, 3)
    orientations /= np.linalg.norm(orientations, axis=1, keepdims=True)

    return [Seed(el, n) for el, n in zip(locations, orientations)]


def from_odf(odf: npt.NDArray, affine: npt.NDArray, n_seeds: int) -> list[Seed]:
    """Generate seeds from orientation distribution functions

    The seeds are generate in voxels with non-zero average ODFs with the
    orientations importance sampled.

    Args:
        odf: The numpy array containing the ODFs.
        affine: The affine transform to native space.
        n_seeds: The number of seeds to generate.

    Return:
        A list of `n_seeds` seeds suitable for tractography

    """

    # Get the non-zero voxels.
    voxels = np.array(list(zip(*np.nonzero(odf[..., 0]))))
    indices = np.random.randint(len(voxels), size=n_seeds)
    locations_voxel = voxels[indices] + np.random.rand(n_seeds, 3) - [0.5, 0.5, 0.5]
    locations = nib.affines.apply_affine(affine, locations_voxel)

    # Preprare discretization of the ODFs.
    vertices = tg.core.fibonacci_sphere(1000)
    azimuths, colatitudes, _ = tg.core.cart2sph(*vertices.T)
    ishtmtx, _ = tg.core.ishtmtx(azimuths, colatitudes, odf.shape[-1])

    # Importance sample the ODFs based on the discretization.
    orientations = []
    for index in indices:
        voxel = voxels[index]
        local = odf[*voxel]
        values = np.dot(ishtmtx, local)
        cumsum = np.cumsum(np.maximum(values, np.max(values) * 0.00))
        i = np.searchsorted(cumsum, np.random.rand() * cumsum[-1])
        orientations.append(vertices[i])

    return [Seed(el, n) for el, n in zip(locations, orientations)]


def to_array(seeds: list[Seed]) -> npt.NDArray:
    """Split seeds into location and orientation"""
    return np.array([np.hstack((s.location, s.orientation)) for s in seeds])


def save(filename: Path, seeds: list[Seed]):
    """Save a list of seeds to file"""
    data = [np.hstack((s.location, s.orientation)) for s in seeds]
    np.savetxt(filename, data)


def load(filename: Path) -> list[Seed]:
    """Load a list of seed from a file"""
    data = np.loadtxt(filename)
    return [Seed(d[:3], d[3:]) for d in data]


def _triangle_normals(vs, ts):
    """Compute the normal of triangles"""
    ns = [np.cross(vs[t[1]] - vs[t[0]], vs[t[2]] - vs[t[0]]) for t in ts]
    return np.array([n / w if (w := np.linalg.norm(n)) != 0 else n for n in ns])


def _random_barycentric_coordinates():
    """Generate a random point in a triangle"""
    a, b = np.random.rand(2)
    if a + b > 1:
        a = 1 - a
        b = 1 - b
    return 1 - a - b, a, b
