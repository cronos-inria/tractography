from __future__ import annotations

import nibabel as nib
import numpy as np
import trimesh

from nibabel.nifti1 import Nifti1Image
from nibabel import affines
from trimesh import Trimesh


class Domain:
    """A lightweight spatial domain with a triangular boundary surface."""

    def __init__(self, boundary: Trimesh):
        self._boundary = boundary

    @property
    def boundary(self) -> Trimesh:
        return self._boundary

    @classmethod
    def from_image(cls, image: Nifti1Image):
        """Create a Domain from a Nifti1Image
        
        We assume the domain is defined by the non-zero voxels in the image.
        The boundary is extracted using marching cubes.
        
        Args:
            image: The input image defining the domain.
        """

        image_data = image.get_fdata()
        mask = image_data > 0
        affine = np.asarray(image.affine, dtype=float)
        boundary_voxel = trimesh.voxel.ops.matrix_to_marching_cubes(mask)

        # Apply the affine.
        boundary = Trimesh(
            vertices=nib.affines.apply_affine(affine, boundary_voxel.vertices),
            faces=boundary_voxel.faces
        )

        return cls(boundary)

    def contains(self, points):
        """Check if points are inside the domain."""
        points = np.asarray(points, dtype=float)
        if points.ndim == 1:
            points = points[None, :]
        return self._boundary.contains(points)

    def boundary_contains(self, points):
        """Check if points are on the boundary of the domain."""

        points = np.asarray(points, dtype=float)
        if points.ndim == 1:
            points = points[None, :]
        distances = self._boundary.nearest.signed_distance(points)
        return np.isclose(distances, 0.0)

    def to_mask(self, shape, affine) -> Nifti1Image:
        """Create a binary mask of the domain.

        Args:
            shape: The shape of the output mask.
            affine: The affine transformation from voxel to world coordinates.

        Returns:
            A binary mask of the domain.
        """

        # Create a grid of voxel coordinates and convert voxel coordinates to
        # world coordinates.
        grid = np.indices(shape).reshape(len(shape), -1).T
        world_coords = affines.apply_affine(affine, grid)

        # Check which points are inside the domain.
        inside = self.contains(world_coords)

        # Create a binary mask.
        mask = np.zeros(shape, dtype=np.uint8)
        mask[tuple(grid[inside].T)] = 1

        return Nifti1Image(mask, affine)

    def sample(self, n_samples):
        """Sample points uniformly from the boundary of the domain.

        Args:
            n_samples: The number of points to sample.

        Returns:
            An array of sampled points on the boundary of the domain.
        """

        points, _ = trimesh.sample.sample_surface(self._boundary, n_samples)
        return points
