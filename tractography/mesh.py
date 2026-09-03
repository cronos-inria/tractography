"""Generate and persist tetrahedral meshes."""

from __future__ import annotations

from pathlib import Path

import meshio
from nibabel.affines import apply_affine
from nibabel.nifti1 import Nifti1Image
import numpy as np
import numpy.typing as npt

from . import _mesh


class Mesh:
    """A tetrahedral mesh.

    Args:
        vertices: Finite vertex positions with shape ``(N, 3)``.
        tetrahedra: Integer vertex indices with shape ``(M, 4)``.
    """

    def __init__(
        self,
        vertices: npt.ArrayLike,
        tetrahedra: npt.ArrayLike,
    ) -> None:
        vertex_array = np.asarray(vertices)
        tetrahedron_array = np.asarray(tetrahedra)

        if vertex_array.ndim != 2 or vertex_array.shape[1:] != (3,):
            raise ValueError("vertices must have shape (N, 3).")
        if not np.issubdtype(vertex_array.dtype, np.number) or np.issubdtype(
            vertex_array.dtype, np.complexfloating
        ):
            raise TypeError("vertices must contain real numeric values.")
        if not np.all(np.isfinite(vertex_array)):
            raise ValueError("vertices must contain only finite values.")

        if tetrahedron_array.ndim != 2 or tetrahedron_array.shape[1:] != (4,):
            raise ValueError("tetrahedra must have shape (M, 4).")
        if not np.issubdtype(tetrahedron_array.dtype, np.integer) or np.issubdtype(
            tetrahedron_array.dtype, np.bool_
        ):
            raise TypeError("tetrahedra must contain integer indices.")
        if tetrahedron_array.size:
            if np.any(tetrahedron_array < 0):
                raise ValueError("tetrahedra contain negative vertex indices.")
            if np.any(tetrahedron_array >= len(vertex_array)):
                raise ValueError("tetrahedra contain out-of-range vertex indices.")
            if np.any(tetrahedron_array > np.iinfo(np.uint32).max):
                raise ValueError("tetrahedra indices exceed the uint32 range.")

        self.vertices = np.ascontiguousarray(vertex_array, dtype=np.float32)
        self.tetrahedra = np.ascontiguousarray(
            tetrahedron_array, dtype=np.uint32
        )

    def nearest_vertex(self, point: npt.ArrayLike) -> npt.NDArray[np.float32]:
        """Return the mesh vertex nearest to a three-dimensional point."""
        point_array = _validate_point(point)
        if not len(self.vertices):
            raise ValueError("nearest_vertex() requires a non-empty mesh.")
        squared_distances = np.sum(
            (self.vertices.astype(np.float64) - point_array) ** 2,
            axis=1,
        )
        return self.vertices[np.argmin(squared_distances)]


def _validate_point(point: npt.ArrayLike) -> npt.NDArray[np.float64]:
    point_array = np.asarray(point, dtype=np.float64)
    if point_array.shape != (3,):
        raise ValueError("point must have shape (3,).")
    if not np.all(np.isfinite(point_array)):
        raise ValueError("point must contain only finite values.")
    return point_array


def _validate_positive(value: float, name: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real scalar.") from error
    if not np.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return converted


def nearest_vertex(mesh: Mesh, point: npt.ArrayLike) -> npt.NDArray[np.float32]:
    """Return the vertex in ``mesh`` nearest to ``point``."""
    return mesh.nearest_vertex(point)


def from_image(
    image: Nifti1Image,
    tetrahedra_size: float = 1.0,
    distance: float = 1.0,
) -> Mesh:
    """Generate a tetrahedral mesh from a 3D NIfTI image.

    Every finite nonzero voxel belongs to one domain. Integer voxel indices are
    voxel centers, and each voxel extends half a voxel beyond its center. Output
    vertices use NIfTI world coordinates.

    Args:
        image: A three-dimensional NIfTI image.
        tetrahedra_size: Uniform upper bound on tetrahedron circumradii.
        distance: Uniform surface approximation distance in physical units.
    """
    if not isinstance(image, Nifti1Image):
        raise TypeError("image must be a nibabel Nifti1Image.")

    image_data = np.asanyarray(image.dataobj)
    if image_data.ndim != 3:
        raise ValueError("image must be three-dimensional.")
    if any(size == 0 for size in image_data.shape):
        raise ValueError("image dimensions must be non-empty.")
    try:
        finite = np.isfinite(image_data)
    except TypeError as error:
        raise TypeError("image data must be numeric.") from error
    if not np.all(finite):
        raise ValueError("image data must contain only finite values.")

    mask = np.ascontiguousarray(image_data != 0, dtype=np.uint8)
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
        _validate_positive(tetrahedra_size, "tetrahedra_size"),
        _validate_positive(distance, "distance"),
    )

    native_mesh = Mesh(native_vertices, native_tetrahedra)
    world_vertices = apply_affine(axis_physical_to_world, native_mesh.vertices)
    return Mesh(world_vertices, native_mesh.tetrahedra)


def load(filename: str | Path) -> Mesh:
    """Load a tetrahedral mesh with meshio."""
    mesh_data = meshio.read(filename)
    tetrahedron_blocks = [
        np.asarray(block.data)
        for block in mesh_data.cells
        if block.type == "tetra"
    ]
    if not tetrahedron_blocks:
        raise ValueError("The file does not contain tetrahedral cells.")
    tetrahedra = np.concatenate(tetrahedron_blocks, axis=0)
    return Mesh(mesh_data.points, tetrahedra)


def save(
    filename: str | Path,
    mesh: Mesh,
) -> None:
    """Save a tetrahedral mesh."""
    if not isinstance(mesh, Mesh):
        raise TypeError("mesh must be a Mesh instance.")

    meshio.write(
        filename,
        meshio.Mesh(
            points=mesh.vertices,
            cells=[("tetra", mesh.tetrahedra)],
        ),
    )
