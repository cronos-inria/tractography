import numpy as np
import pydantic
import trimesh

import tractography as tg
from . import opencl as cl
from .configuration import Algorithm, BaseConfiguration
from . import register


class Configuration(BaseConfiguration):
    inverse_curvature: pydantic.PositiveFloat
    noise_variance: pydantic.PositiveFloat

    @classmethod
    def load(cls):
        return super().load(Algorithm.DIFFUSION)


def histogram(fod, fod_affine, seed_fod, seed_fod_affine, n_seeds, config):
    """Generate a streamline histogram

    The histogram corresponds to the FOD associated to a particular tractogram. That is,
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

    # Send the data to the device.
    fod_buffer = cl.new_read_only_buffer(fod.astype(np.float32))
    fod_inverse_affine_buffer = cl.new_read_only_buffer(np.linalg.inv(fod_affine).astype(np.float32))
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
        "n_seeds": n_threads,
    }
    program = cl.build_program(values, ["utils/seeds.cl", "diffusion/histogram.cl"])

    args = (
        fod_buffer,
        fod_inverse_affine_buffer,
        seed_fod_buffer,
        seed_fod_voxels_buffer,
        seed_fod_affine_buffer,
        randoms_buffer,
        np.float32(config.step_size),
        np.float32(config.save_at),
        np.float32(config.inverse_curvature),
        np.float32(config.noise_variance),
        np.uint32(n_seeds_per_thread),
        hist_buffer,
    )
    cl.run_histogram(program, args, n_threads)
    cl.copy_from_buffer(hist_buffer, hist)

    hist = tg.utils.normalize_odf(hist)
    return hist


class Diffusion:

    def __init__(self, odf, affine, n_streamlines, config):

        self._odf_shape = odf.shape
        self._n_streamlines = n_streamlines
        self._config = config

        # Precompute the inverse affine.
        iaffine = np.linalg.inv(affine).astype(np.float32)
        self._iaffine = cl.new_read_only_buffer(iaffine)

        # Check the shape of the fODF data. For now, only 45 coefficients are
        # supported.
        if odf.shape[-1] != 45:
            raise ValueError(
                "For now, only fODF with 45 coefficients are supported (lmax=8)."
            )
        self._odf = cl.new_read_only_buffer(odf.astype(np.float32))

        # Add the random number states.
        randoms_array = np.random.randint(4294967295, size=(n_streamlines, 2)).astype(
            np.uint32
        )
        self._randoms = cl.new_read_only_buffer(randoms_array)

        # Create the seed buffer on the device. They are stored as two float4.
        seeds_array = np.empty((n_streamlines, 8), dtype=np.float32)
        self._seeds = cl.new_read_only_buffer(seeds_array)

        # Reserve space for the streamlines on the device. The are
        # stored as float4.
        streamlines_nbytes = n_streamlines * config.n_steps * 4 * 4
        self._streamlines = cl.new_write_only_buffer(streamlines_nbytes)

        # Reserve space for the length of the streamlines on the device.
        self._lengths = cl.new_write_only_buffer(n_streamlines * 4)

        # Set constants in the OpenCL code.
        values = {
            "nx": odf.shape[0],
            "ny": odf.shape[1],
            "nz": odf.shape[2],
            "n_coefficients": odf.shape[-1],
            "n_steps": config.n_steps,
            "n_streamlines": n_streamlines,
        }
        self._program = cl.build_program(values, "diffusion/tractogram.cl")

    def run(self, seeds):

        # Transfer the seeds to the buffer.
        array = tg.seeds.to_array(seeds).astype(np.float32)
        cl.copy_to_buffer(self._seeds, array)

        # Track streamlines.
        args = (
            self._odf,
            self._iaffine,
            self._seeds,
            self._randoms,
            np.float32(self._config.step_size),
            np.float32(self._config.save_at),
            np.float32(self._config.inverse_curvature),
            np.float32(self._config.noise_variance),
            self._streamlines,
            self._lengths,
        )
        cl.run_program(self._program, args, self._n_streamlines)
        streamlines = np.zeros(
            (self._n_streamlines, self._config.n_steps, 4), dtype=np.float32
        )
        cl.copy_from_buffer(self._streamlines, streamlines)
        lengths = np.zeros((self._n_streamlines,), dtype=np.uint32)
        cl.copy_from_buffer(self._lengths, lengths)

        return [streamlines[i, :n, :3] for i, n in enumerate(lengths)]


def connectome(
    fod,
    fod_affine,
    seed_fod,
    seed_fod_affine,
    vertices,
    vertex_labels,
    n_labels,
    n_seeds,
    config,
    distance_upper_bound=4.0,
):
    """Generate a connectome directly without storing streamlines.

    Each streamline is seeded from the FOD, propagated using diffusion tracking,
    and then its start and end points are matched to the nearest labelled vertex.
    The corresponding matrix entry is atomically incremented.

    Args:
        fod: The FOD data array (4D).
        fod_affine: The affine for the FOD image.
        seed_fod: The masked seed FOD data array (4D).
        seed_fod_affine: The affine for the seed FOD image.
        vertices: World-space coordinates of labelled points (N, 3), as returned
            by tractography.connectivity.convert_segmentation.
        vertex_labels: Integer label for each vertex (N,), as returned by
            tractography.connectivity.convert_segmentation.
        n_labels: The number of unique labels (matrix dimension).
        n_seeds: The number of seeds (streamlines) to generate.
        config: The configuration object.
        distance_upper_bound: Maximum distance between a streamline endpoint
            and a vertex for the endpoint to be assigned that vertex's label.

    Returns:
        matrix: A (n_labels, n_labels) uint32 connectivity matrix.

    """

    n_threads = 1000
    n_seeds_per_thread = np.ceil(n_seeds / n_threads)

    mask = seed_fod[..., 0] > 0
    voxels = np.array(np.nonzero(mask), dtype=np.float32).T
    voxels = np.hstack((voxels, np.ones((len(voxels), 1))))
    inline_fod = fod[mask]
    randoms = np.random.randint(4294967295, size=(n_threads, 2))

    fod_buffer = cl.new_read_only_buffer(fod.astype(np.float32))
    fod_inverse_affine_buffer = cl.new_read_only_buffer(
        np.linalg.inv(fod_affine).astype(np.float32)
    )
    seed_fod_buffer = cl.new_read_only_buffer(inline_fod.astype(np.float32))
    seed_fod_voxels_buffer = cl.new_read_only_buffer(voxels.astype(np.float32))
    seed_fod_affine_buffer = cl.new_read_only_buffer(
        seed_fod_affine.astype(np.float32)
    )
    randoms_buffer = cl.new_buffer(randoms.astype(np.uint32))

    vertices_homogeneous = np.c_[vertices, np.zeros((len(vertices), 1))]
    vertices_buffer = cl.new_read_only_buffer(vertices_homogeneous.astype(np.float32))
    vertex_labels_buffer = cl.new_read_only_buffer(vertex_labels.astype(np.int32))

    conn_matrix = np.zeros((n_labels, n_labels), dtype=np.uint32)
    conn_matrix_buffer = cl.new_buffer(conn_matrix)

    values = {
        "nx": fod.shape[0],
        "ny": fod.shape[1],
        "nz": fod.shape[2],
        "nnz": len(voxels),
        "n_coefficients": fod.shape[-1],
        "n_steps": config.n_steps,
        "n_seeds": n_threads,
        "n_labels": n_labels,
        "n_vertices": len(vertices),
    }
    program = cl.build_program(values, ["utils/seeds.cl", "diffusion/connectome.cl"])

    args = (
        fod_buffer,
        fod_inverse_affine_buffer,
        seed_fod_buffer,
        seed_fod_voxels_buffer,
        seed_fod_affine_buffer,
        randoms_buffer,
        vertices_buffer,
        vertex_labels_buffer,
        np.float32(config.step_size),
        np.float32(config.save_at),
        np.uint32(config.min_n_points),
        np.float32(config.inverse_curvature),
        np.float32(config.noise_variance),
        np.float32(distance_upper_bound),
        np.uint32(n_seeds_per_thread),
        conn_matrix_buffer,
    )
    program.connectome(cl._queue, (n_threads,), None, *args)
    cl.copy_from_buffer(conn_matrix_buffer, conn_matrix)

    return conn_matrix


register(Algorithm.DIFFUSION, Diffusion, histogram, connectome)
