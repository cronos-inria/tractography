import unittest

import nibabel as nib
import numpy as np
import forward

import tractography as tg
import tractography.algorithms.opencl as opencl


def cube(shape: tuple[int, int, int]) -> np.ndarray:
    """Generate a 3D label volume where each layer along the last axis has a 
    distinct label.

    Each slice ``image[..., i]`` is filled with the value ``i``, producing
    a volume with as many unique labels as there are elements along the last
    axis (excluding 0). The outermost voxels on every face are then set to 0,
    creating a one-voxel-wide background border.

    Args:
        shape: The (nx, ny, nz) dimensions of the output volume.

    Returns:
        A numpy array of the given *shape* with per-layer labels and a
        zero-valued border on all faces.
    """
    # Generate an image where each layer has a different label.
    image = np.zeros(shape)
    for i in range(shape[-1]):
        image[..., i] = i

    # Remove the border (set all boundary voxels to 0).
    for axis in range(image.ndim):
        np.moveaxis(image, axis, 0)[0] = 0
        np.moveaxis(image, axis, 0)[-1] = 0

    return image


class TestFindTetrahedronContainingPoints(unittest.TestCase):

    def test_cube(self):

        # Create a cube with 3 layers along the z-axis.
        shape = (5, 5, 3)
        image = nib.nifti1.Nifti1Image(cube(shape), affine=np.eye(4))

        # Generate random points within the cube.
        n_points = 100
        points = np.random.rand(n_points, 3) * (np.array(shape) - 1)

        # Generate the tetrahedral mesh from the cube.
        mesh = forward.mesh.from_image(image, 0.5)

        # Dummy values for the mesh vertices.
        n_values_per_vertex = 1
        values = np.ones(
            (mesh.vertices.shape[0], 
            n_values_per_vertex),
            dtype=np.float32
        )

        # Setup OpenCL context and build the program.
        program = opencl.build_program(dict(), ["fields/mesh.cl"])

        vertices = np.c_[
            mesh.vertices, 
            np.ones(mesh.vertices.shape[0])
        ].astype(np.float32)
        vertices_buffer = opencl.new_read_only_buffer(vertices)
        tetrahedra = mesh.tetrahedra.astype(np.int32)
        tetrahedra_buffer = opencl.new_read_only_buffer(tetrahedra)
        values_buffer = opencl.new_read_only_buffer(values)
        
        points_homogeneous = np.c_[points, np.ones(points.shape[0])].astype(
            np.float32
        )
        points_buffer = opencl.new_read_only_buffer(points_homogeneous)

        # 4 bytes per index
        output_buffer = opencl.new_write_only_buffer(n_points * 4)  

        args = (
            vertices_buffer,
            np.int32(vertices.shape[0]),
            tetrahedra_buffer,
            np.int32(tetrahedra.shape[0]),
            values_buffer,
            np.int32(n_values_per_vertex),
            points_buffer,
            np.int32(n_points),
            output_buffer,
        )
        program.find_tetrahedra_containing_points(
            opencl._queue, (n_points,), None, *args)
        indices = np.empty(n_points, dtype=np.int32)
        opencl.copy_from_buffer(output_buffer, indices).wait()

        # If the indices are correct, the barycentric coordinates of the points
        # with respect to the tetrahedra should be non-negative. If the index is
        # -1, the barycentric coordinates should contain a negative value for
        # all tetrahedra.
        for point, tetrahedron_index in zip(points, indices):
            rhs = np.r_[point, 1.0]

            if tetrahedron_index > -1:
                tetra = mesh.tetrahedra[tetrahedron_index]
                verts = mesh.vertices[tetra]
                A = np.c_[verts, np.ones(4)]
                bary_coords = np.linalg.solve(A.T, rhs)
                self.assertTrue(np.all(bary_coords >= -1e-6))
            else:
                for tetra in mesh.tetrahedra:
                    verts = mesh.vertices[tetra]
                    A = np.c_[verts, np.ones(4)]
                    bary_coords = np.linalg.solve(A.T, rhs)
                    self.assertTrue(np.any(bary_coords < -1e-6))


class TestInterpolateFieldAtPoints(unittest.TestCase):

    def test_cube(self):

        # Create a cube with 3 layers along the z-axis.
        shape = (5, 5, 3)
        image = nib.nifti1.Nifti1Image(cube(shape), affine=np.eye(4))

        # Generate random points within the cube.
        n_points = 100
        points = np.random.rand(n_points, 3) * (np.array(shape) - 1)

        # Generate the tetrahedral mesh from the cube.
        mesh = forward.mesh.from_image(image)

        # Setup OpenCL context and build the program.
        program = opencl.build_program(dict(), ["fields/mesh.cl"])

        vertices = np.c_[
            mesh.vertices, 
            np.ones(mesh.vertices.shape[0])
        ].astype(np.float32)
        vertices_buffer = opencl.new_read_only_buffer(vertices)
        tetrahedra = mesh.tetrahedra.astype(np.int32)
        tetrahedra_buffer = opencl.new_read_only_buffer(tetrahedra)

        # We use the vertex coordinates as values. The interpolated values
        # should then be equal to the barycentric coordinates of the points with
        # respect to the tetrahedra.
        n_values_per_vertex = 3
        values_buffer = opencl.new_read_only_buffer(
            mesh.vertices.astype(np.float32)
        )
        
        points_homogeneous = np.c_[points, np.ones(points.shape[0])].astype(
            np.float32
        )
        points_buffer = opencl.new_read_only_buffer(points_homogeneous)

        # 4 bytes per output value, times the number of values per vertex.
        output_buffer = opencl.new_write_only_buffer(
            n_points * n_values_per_vertex * 4
        )  

        args = (
            vertices_buffer,
            np.int32(vertices.shape[0]),
            tetrahedra_buffer,
            np.int32(tetrahedra.shape[0]),
            values_buffer,
            np.int32(n_values_per_vertex),
            points_buffer,
            np.int32(n_points),
            output_buffer,
        )
        program.interpolate_field_at_points(
            opencl._queue, (n_points,), None, *args)
        interpolated_values = np.empty(
            (n_points, n_values_per_vertex), 
            dtype=np.float32,
        )
        opencl.copy_from_buffer(output_buffer, interpolated_values).wait()

        # Precompute the tetrahedron indices for the points.
        tetrahedra_indices = np.empty(n_points, dtype=np.int32)
        program.find_tetrahedra_containing_points(
            opencl._queue, (n_points,), None, *args[:-1], output_buffer)
        opencl.copy_from_buffer(output_buffer, tetrahedra_indices).wait()

        # The interpolated values should be equal to the point coordinates if
        # the point is inside a tetrahedron, and NaN if it is outside.
        for point, values, index in zip(points, interpolated_values, tetrahedra_indices):
            if index == -1:
                self.assertTrue(np.all(np.isnan(values)))
            else:
                np.testing.assert_array_almost_equal(values, point, decimal=5)