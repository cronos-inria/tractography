import numpy as np
import pydantic

import tractography as tg
from . import opencl as cl
from .configuration import Algorithm, BaseConfiguration
from . import register


class Configuration(BaseConfiguration):
    maximum_angle: pydantic.PositiveInt  # in degrees

    @classmethod
    def load(cls):
        return super().load(Algorithm.DETERMINISTIC)


def histogram(fod, fod_affine, seed_fod, seed_fod_affine, n_seeds, config):
    """Generate a streamline histogram

    The histogram correponds to the FOD associated to a particular tracgogram. That is,
    the distribution of streamline orientations, for each voxel. This function
    generates the histogram directly, without saving the intermediate streamlines and
    therefore allows a much larger number of seeds to be used.

    """

    # Determine the number of seeds per thread. We want 1k threads.
    n_threads = 1000
    n_seeds_per_thread = np.ceil(n_seeds / n_threads)

    # Prepare the data necessary for the seeds.
    mask = seed_fod[..., 0] > 0
    voxels = np.array(np.nonzero(mask), dtype=np.float32).T
    voxels = np.hstack((voxels, np.ones((len(voxels), 1))))
    inline_fod = fod[mask]
    randoms = np.random.randint(4294967295, size=(n_threads, 2))

    # Generate a set of orientation where the FODs are evaluated. On the
    # device the vertices are represented as float4.
    n_directions = 300
    directions = tg.core.fibonacci_sphere(n_directions)
    directions_homogeneous = np.c_[directions, np.zeros((n_directions,))]

    # Convert the spherical harmonics to 1D probability mass functions.
    n_coefficients = fod.shape[-1]
    azimuths, colatitudes, _ = tg.core.cart2sph(*directions.T)
    matrix, _ = tg.core.ishtmtx(azimuths, colatitudes, n_coefficients)
    fod_values = np.maximum(np.dot(fod.reshape((-1, n_coefficients)), matrix.T), 0)
    fod_values = fod_values.reshape((*fod.shape[:3], -1))

    # Send the data to the device.
    fod_values_buffer = cl.new_read_only_buffer(fod_values.astype(np.float32))
    fod_inverse_affine_buffer = cl.new_read_only_buffer(
        np.linalg.inv(fod_affine).astype(np.float32)
    )
    directions_buffer = cl.new_read_only_buffer(
        directions_homogeneous.astype(np.float32)
    )
    seed_fod_buffer = cl.new_read_only_buffer(inline_fod.astype(np.float32))
    seed_fod_voxels_buffer = cl.new_read_only_buffer(voxels.astype(np.float32))
    seed_fod_affine_buffer = cl.new_read_only_buffer(seed_fod_affine.astype(np.float32))
    randoms_buffer = cl.new_buffer(randoms.astype(np.uint32))

    hist = np.zeros(fod.shape, dtype=np.float32)
    hist_buffer = cl.new_buffer(hist)

    # Set constants in the OpenCL code.
    values = {
        "nx": fod.shape[0],
        "ny": fod.shape[1],
        "nz": fod.shape[2],
        "nnz": len(voxels),
        "n_coefficients": fod.shape[-1],
        "n_steps": config.n_steps,
        "n_directions": n_directions,
        "n_seeds": n_threads,
    }
    program = cl.build_program(values, ["utils/seeds.cl", "deterministic/histogram.cl"])

    args = (
        fod_values_buffer,
        fod_inverse_affine_buffer,
        directions_buffer,
        seed_fod_buffer,
        seed_fod_voxels_buffer,
        seed_fod_affine_buffer,
        randoms_buffer,
        np.float32(config.step_size),
        np.float32(config.save_at),
        np.float32(np.cos(np.deg2rad(config.maximum_angle))),
        np.uint32(n_seeds_per_thread),
        hist_buffer,
    )
    cl.run_histogram(program, args, n_threads)
    cl.copy_from_buffer(hist_buffer, hist)

    hist = tg.utils.normalize_odf(hist)
    return hist


class Deterministic:
    """OpenCL implementation of deterministic tractography

    This class implements a local deterministic tractography algorithm. The
    streamlines are propagated step-by-step by choosing the orientation
    with maximal amplitude from the discritized ODFs.

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

        self._n_streamlines = n_streamlines
        self._config = config

        # Generate a set of orientation where the FODs are evaluated. On the
        # device the vertices are represented as float4.
        n_points = 400
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
        self._streamlines = cl.new_write_only_buffer(streamlines_nbytes)

        # Reserve space for the length of the streamlines on the device.
        self._lengths = cl.new_write_only_buffer(n_streamlines * 4)

        # Fill columns for seeds.
        self._zeros = np.zeros(self._n_streamlines, dtype=np.float32)
        self._ones = np.ones(self._n_streamlines, dtype=np.float32)

        # Build the OpenCL program that implements tractography.
        values = {
            "nx": odf.shape[0],
            "ny": odf.shape[1],
            "nz": odf.shape[2],
            "n_directions": len(vertices),
            "n_steps": config.n_steps,
            "n_streamlines": n_streamlines,
        }
        self._program = cl.build_program(values, "deterministic/tractogram.cl")

    def run(self, seeds):
        """Run the deterministic algorithm on the given seed points

        Args:
            seeds: List of seed points in world coordinates with associated
                direction vectors.

        Returns:
            A list of streamlines, each represented as a 2D array of 3D
                coordinates (N x 3), where N is the number of steps
                successfully tracked for that streamline.

        """

        # Transfer the seeds to the buffer.
        array = tg.seeds.to_array(seeds).astype(np.float32)
        cl.copy_to_buffer(self._seeds, array)

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
        cl.run_tractogram(self._program, args, self._n_streamlines)
        streamlines = np.zeros(
            (self._n_streamlines, self._config.n_steps, 4), dtype=np.float32
        )
        cl.copy_from_buffer(self._streamlines, streamlines)
        lengths = np.zeros((self._n_streamlines,), dtype=np.uint32)
        cl.copy_from_buffer(self._lengths, lengths)

        return [streamlines[i, :n, :3] for i, n in enumerate(lengths)]


register(Algorithm.DETERMINISTIC, Deterministic)
