from __future__ import annotations
from collections import Counter
from itertools import combinations
from typing import Literal

import numpy as np
import numpy.typing as npt
from nibabel.affines import apply_affine
from nibabel.nifti1 import Nifti1Image

from trimesh import Trimesh

from . import _mesh


type Vertices = np.ndarray[tuple[int, Literal[3]], np.dtype[np.float64]]
type Triangles = np.ndarray[tuple[int, Literal[3]], np.dtype[np.uint64]]

# Phase space point with the u part expressed in Cartesian.
type Points = np.array[tuple[int, Literal[6]], np.dtype[np.float64]]


class DomainBoundary(Trimesh):
    """The boundary of a phase space domain for tractography."""

    def __init__(self, vertices: Vertices, triangles: Triangles):
        """Initialize a new boundary.

        The boundary is represented by a triangular surface in R3. The angular
        part of the domain does not need to be specified, as all vectors in
        S2 are part of the domain.

        Args:
            vertices: The vertex array. Must have a shape of (N, 3).
            triangles: The triangle indices array. Must have a shape of (M, 3).

        """

        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("'vertices' must have a shape of (N, 3).")
        if triangles.ndim != 2 or triangles.shape[1] != 3:
            raise ValueError("'triangles' must have a shape of (M, 3).")

        super().__init__(vertices, triangles)

        # Normals must be pointing out.
        self.fix_normals()

        if not self.is_volume:
            raise ValueError("The boundary must be a volume.")

    def __contains__(self, points: Points) -> np.bool:
        """Syntactic sugar, all points must be on the boundary for True."""
        return np.all(self.contains(points))

    def contains(self, points: Points) -> np.bool:
        """Indicates if points are on the boundary."""

        # We ignore the angular part as all S2 points are included.
        distances = self.nearest.signed_distance(points[:, :3])
        return np.isclose(distances, 0.0)


class Domain:
    """A phase space domain for tractography.

    The domain is represented by a tetrahedral volume. The angular part of the
    domain does not need to be specified, as all vector in S2 are part of the
    domain.

    Args:
        vertices: The vertex array. Must have a shape of (N, 3).
        tetrahedra: The tetrahedra indices array. Must have a shape of (M, 4).

    """

    def __init__(
        self,
        vertices: Vertices,
        tetrahedra: npt.ArrayLike,
    ) -> None:
        vertex_array = np.asarray(vertices)
        tetrahedron_array = np.asarray(tetrahedra)

        if vertex_array.ndim != 2 or vertex_array.shape[1:] != (3,):
            raise ValueError("'vertices' must have shape (N, 3).")
        if not np.issubdtype(vertex_array.dtype, np.number) or np.issubdtype(
            vertex_array.dtype, np.complexfloating
        ):
            raise TypeError("'vertices' must contain real numeric values.")
        if not np.all(np.isfinite(vertex_array)):
            raise ValueError("'vertices' must contain only finite values.")

        if tetrahedron_array.ndim != 2 or tetrahedron_array.shape[1:] != (4,):
            raise ValueError("'tetrahedra' must have shape (M, 4).")
        if not np.issubdtype(tetrahedron_array.dtype, np.integer) or np.issubdtype(
            tetrahedron_array.dtype, np.bool_
        ):
            raise TypeError("'tetrahedra' must contain integer indices.")
        if tetrahedron_array.size:
            if np.any(tetrahedron_array < 0):
                raise ValueError("'tetrahedra' contain negative vertex indices.")
            if np.any(tetrahedron_array >= len(vertex_array)):
                raise ValueError("'tetrahedra' contain out-of-range vertex indices.")

        self.vertices = vertices
        self.tetrahedra = tetrahedra

        # Extract the boundary; all triangles that appear in a single
        # tetrahedra.
        triangles = [
            tuple(sorted(t)) for a in tetrahedra for t in combinations(a, 3)
        ]
        boundary_tri = [k for k, v in Counter(triangles).items() if v == 1]
        boundary = DomainBoundary(vertices, np.array(boundary_tri))
        self.boundary = boundary


def from_image(
    image: Nifti1Image,
    tetrahedra_size: float = 1.0,
    distance: float = 1.0,
) -> Domain:
    """Generate a tetrahedral mesh from a 3D NIfTI image.

    Every nonzero voxel belongs to one domain. Integer voxel indices are
    voxel centers, and each voxel extends half a voxel beyond its center.
    Output vertices use NIfTI world coordinates.

    Args:
        image: A three-dimensional NIfTI image.
        tetrahedra_size: Uniform upper bound on tetrahedron circumradii.
        distance: Uniform surface approximation distance in physical units.
    """

    mask = np.ascontiguousarray(image.get_fdata() != 0, dtype=np.uint8)
    if not np.any(mask):
        raise ValueError("image must contain at least one nonzero voxel.")

    zooms = np.asarray(image.header.get_zooms()[:3], dtype=np.float64)
    if zooms.shape != (3,) or not np.all(np.isfinite(zooms)) or np.any(zooms <= 0):
        raise ValueError("image voxel sizes must be positive and finite.")

    affine = np.asarray(image.affine, dtype=np.float64)
    if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
        raise ValueError("image affine must be a finite 4 by 4 matrix.")
    axis_physical_to_world = affine @ np.diag(np.r_[1.0 / zooms, 1.0])
    try:
        np.linalg.inv(axis_physical_to_world)
    except np.linalg.LinAlgError as error:
        raise ValueError("image affine must be invertible.") from error

    spacing = tuple(float(value) for value in zooms)
    native_vertices, native_tetrahedra = _mesh.create(
        mask,
        spacing[0],
        spacing[1],
        spacing[2],
        tetrahedra_size,
        distance,
    )

    world_vertices = apply_affine(axis_physical_to_world, native_vertices)
    return Domain(world_vertices, native_tetrahedra)
