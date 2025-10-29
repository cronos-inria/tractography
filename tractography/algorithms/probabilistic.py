import numpy as np
import pydantic
import trimesh

import tractography as tg
from . import opencl as cl
from .configuration import Algorithm, BaseConfiguration


class Configuration(BaseConfiguration):
    maximum_angle: pydantic.PositiveInt  # in degrees

    @property
    def implementation(self):
        return Probabilistic

    @classmethod
    def load(cls):
        return super().load(Algorithm.PROBABILISTIC)


class Probabilistic:
    """OpenCL implementation of probabilistic tractography

    This class implements a local probabilistic tractography algorithm. The
    streamlines are propagated step-by-step by choosing a random orientation
    from the discritized ODFs.

    """

    def __init__(self, odf, affine, n_streamlines, config):
        """Initializes the algorithm with tracking data and parameters

        Prepares OpenCL buffers for ODFs, seed points, output streamlines,
        and tracking parameters. Discitizes the ODFs, reshapes them
        for OpenCL compatibility, and allocates memory for results on the
        device.

        Args:
            odf: 4D array representing fiber orientation distributions at each
                    voxel, in spherical harmonics format.
            affine: 4x4 affine matrix mapping voxel indices to world
                coordinates.
            n_streamlines: Number of streamlines to generate when running the
                algorithm (equal to the number of seeds).
            config: Configuration object containing tracking parameters (e.g.,
                step size, number of steps).

        """

        self._odf_shape = odf.shape
        self._n_streamlines = n_streamlines
        self._config = config

        # Generate a set of orientation where the FODs are evaluated. On the
        # device the vertices are represented as float4.
        n_points = 300
        vertices = tg.core.fibonacci_sphere(n_points)
        device_vertices = np.c_[vertices, np.zeros((n_points,))]
        self._vertices = cl.new_read_only_buffer(device_vertices.astype(np.float32))

        # Precompute the inverse affine.
        iaffine = np.linalg.inv(affine).astype(np.float32)
        self._iaffine = cl.new_read_only_buffer(iaffine)

        # Convert the spherical harmonics to 1D probability mass functions.
        n_coefficients = odf.shape[-1]
        azimuths, colatitudes, _ = tg.core.cart2sph(*vertices.T)
        matrix, _ = tg.core.ishtmtx(azimuths, colatitudes, n_coefficients)
        odf_values = np.maximum(np.dot(odf.reshape((-1, n_coefficients)), matrix.T), 0)
        odf_values = odf_values.reshape((*odf.shape[:3], -1))
        self._values = cl.new_read_only_buffer(odf_values.astype(np.float32))

        # Create the seed buffer on the device. They are stored as two float4.
        seeds_array = np.empty((n_streamlines, 8), dtype=np.float32)
        self._seeds = cl.new_read_only_buffer(seeds_array)

        # Reserve space for the streamlines on the device. The are
        # stored as float4.
        streamlines_nbytes = n_streamlines * config.n_steps * 4 * 4

        # Fill columns for seeds.
        self._zeros = np.zeros(self._n_streamlines, dtype=np.float32)
        self._ones = np.ones(self._n_streamlines, dtype=np.float32)

        # Preallocate the output.
        buffer = cl.cl.Buffer(
            cl._context,
            cl.cl.mem_flags.READ_WRITE | cl.cl.mem_flags.ALLOC_HOST_PTR,
            size=streamlines_nbytes,
        )
        self._output, event = cl.cl.enqueue_map_buffer(
            cl._queue,
            buffer,
            cl.cl.map_flags.READ,
            0,
            shape=(n_streamlines, config.n_steps, 4),
            dtype=np.float32,
        )
        self._streamlines = buffer
        buffer = cl.cl.Buffer(
            cl._context,
            cl.cl.mem_flags.READ_WRITE | cl.cl.mem_flags.ALLOC_HOST_PTR,
            size=n_streamlines * 4,
        )
        self._output_lengths, event = cl.cl.enqueue_map_buffer(
            cl._queue,
            buffer,
            cl.cl.map_flags.READ,
            0,
            shape=(n_streamlines,),
            dtype=np.uint32,
        )
        self._lengths = buffer

        # Build the OpenCL program that implements tractography.
        values = {
            "nx": odf.shape[0],
            "ny": odf.shape[1],
            "nz": odf.shape[2],
            "n_directions": len(vertices),
            "n_steps": config.n_steps,
            "n_streamlines": n_streamlines,
        }
        self._program = cl.build_program(values, "probabilistic.cl")

    def histogram(self, seeds):

        # Transfer the seeds to the buffer.
        array = tg.seeds.to_array(seeds).astype(np.float32)
        array = np.c_[array[:, :3], self._ones, array[:, 3:], self._zeros]
        cl.copy_to_buffer(self._seeds, array)

        # Reserve space on the device for the histogram.
        hist_nbytes = np.prod(self._odf_shape[:3]) * 162 * 4
        hist_buffer = cl.new_write_only_buffer(hist_nbytes)

        # Reserve space for the bin centers.
        mesh = trimesh.creation.icosphere(2)
        bin_areas = np.zeros(len(mesh.vertices))
        for i, indices in enumerate(mesh.vertex_faces):
            for index in indices:
                if index != -1:
                    bin_areas[i] += mesh.area_faces[index] / 3
        bin_centers = mesh.vertices / np.linalg.norm(
            mesh.vertices, axis=1, keepdims=True
        )
        bin_centers = np.c_[bin_centers, np.zeros((162, 1))].astype(np.float32)
        bin_centers_buffer = cl.new_read_only_buffer(bin_centers)

        max_angle = self._config.maximum_angle
        args = (
            self._values,
            self._iaffine,
            bin_centers_buffer,  # We use the bin centers instead of vertices.
            self._seeds,
            np.float32(self._config.step_size),
            np.float32(np.cos(np.deg2rad(max_angle))),
            hist_buffer,
        )
        cl.run_histogram(self._program, args, self._n_streamlines)
        hist = np.zeros(self._odf_shape[:3] + (162,), dtype=np.uint32)
        cl.copy_from_buffer(hist_buffer, hist)

        # Normalize the histogram.
        hist = hist / np.expand_dims(bin_areas, axis=(0, 1, 2))
        hist_norm = np.linalg.norm(hist, axis=-1, keepdims=True)
        np.divide(hist, hist_norm, where=hist_norm != 0, out=hist)

        return hist, bin_centers[:, :3]

    def run(self, seeds):
        """Run the probabilistic algorithm on the given seed points

        Args:
            seeds: List of seed points in world coordinates with associated
                direction vectors.

        Returns:
            A list of streamlines, each represented as a 2D array of 3D
                coordinates (N x 3), where N is the number of steps
                successfully tracked for that streamline.

        """

        events = []

        # Transfer the seeds to the buffer.
        array = tg.seeds.to_array(seeds).astype(np.float32)
        array = np.c_[array[:, :3], self._ones, array[:, 3:], self._zeros]
        events.append(cl.copy_to_buffer(self._seeds, array))

        # Track streamlines.
        max_angle = self._config.maximum_angle
        args = (
            self._values,
            self._iaffine,
            self._vertices,
            self._seeds,
            np.float32(self._config.step_size),
            np.float32(np.cos(np.deg2rad(max_angle))),
            self._streamlines,
            self._lengths,
        )
        events.append(cl.run_program(self._program, args, self._n_streamlines))
        events.append(cl.copy_from_buffer(self._lengths, self._output_lengths))
        events.append(cl.copy_from_buffer(self._streamlines, self._output))
        cl.cl.wait_for_events(events)

        return [self._output[i, :n, :3] for i, n in enumerate(self._output_lengths)]
