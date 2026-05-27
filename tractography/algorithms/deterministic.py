from dataclasses import dataclass
from typing import Tuple

from nibabel.nifti1 import Nifti1Image
from nibabel.streamlines.tractogram import Tractogram
import numpy as np
import pydantic

from ..seeds import Seed
from .. import core, seeds as seeds_module, utils
from . import opencl as cl
from .core import (
    Algorithm,
    BaseCache,
    BaseConfiguration,
    LocalModel,
    cache_needs_rebuild,
    discretize_fod,
)
from . import register


class Configuration(BaseConfiguration):
    maximum_angle: pydantic.PositiveInt  # in degrees

    @classmethod
    def load(cls):
        return super().load(Algorithm.DETERMINISTIC)


@dataclass
class Cache(BaseCache):
    """Cache for the OpenCL setup of the deterministic algorithm.

    This cache is used to store the OpenCL buffers and program, so that they can
    be reused across multiple tractography runs without needing to reallocate
    memory or rebuild the program.

    """

    # The discretized FOD values at each vertex of the direction sphere.
    values: cl.Buffer | None = None

    # The affine transform from world to voxel space.
    fod_inverse_affine: cl.Buffer | None = None

    # The discretized directions on the sphere.
    directions: cl.Buffer | None = None

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
    mask = np.any(seed_fod != 0, axis=-1)
    voxels = np.array(np.nonzero(mask), dtype=np.float32).T
    voxels = np.hstack((voxels, np.ones((len(voxels), 1))))
    inline_fod = fod[mask]
    randoms = np.random.randint(4294967295, size=(n_threads, 2))

    # Generate a set of orientation where the FODs are evaluated. On the
    # device the vertices are represented as float4.
    n_directions = 300
    directions, fod_values = discretize_fod(fod, n_directions)
    directions_homogeneous = np.c_[directions, np.zeros((n_directions,))]

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

    hist = np.zeros(fod.shape[:3] + (45,), dtype=np.float32)
    hist_buffer = cl.new_buffer(hist)

    # Determine the local model based on the data shape.
    model = LocalModel.from_shape(fod.shape)

    # Set constants in the OpenCL code.
    values = {
        "model": str(model),
        "nx": fod.shape[0],
        "ny": fod.shape[1],
        "nz": fod.shape[2],
        "nnz": len(voxels),
        "n_coefficients": fod.shape[-1],
        "n_steps": config.n_steps,
        "n_directions": n_directions,
        "n_seeds": n_threads,
    }
    program = cl.build_program(values, ["utils/seeds.cl", "algorithms/deterministic/histogram.cl"])

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

    hist = utils.normalize_odf(hist)
    return hist


def tractogram(fod: Nifti1Image, seeds: list[Seed], config: Configuration, cache: Cache | None = None) -> Tuple[Tractogram, Cache]:

    n_streamlines = len(seeds)

    if cache_needs_rebuild(
        cache,
        cache_type=Cache,
        fod_shape=fod.shape,
        n_streamlines=n_streamlines,
        n_steps=config.n_steps,
    ):

        # Generate a new cache for the OpenCL setup.
        cache = Cache()

        cache.fod_shape = tuple(fod.shape)
        cache.n_streamlines = n_streamlines
        cache.n_steps = config.n_steps

        odf = fod.get_fdata()
        affine = fod.affine if fod.affine is not None else np.eye(4)

        # Generate a set of orientations where the FODs are evaluated. On the
        # device the vertices are represented as float4.
        n_points = 400
        vertices, odf_values = discretize_fod(odf, n_points)
        device_vertices = np.c_[vertices, np.zeros((n_points,))]
        cache.directions = cl.new_read_only_buffer(device_vertices.astype(np.float32))

        # Precompute the inverse affine.
        iaffine = np.linalg.inv(affine).astype(np.float32)
        cache.fod_inverse_affine = cl.new_read_only_buffer(iaffine)

        # Use the discretized ODF values from the helper.
        cache.values = cl.new_read_only_buffer(odf_values.astype(np.float32))

        # Create the seed buffer on the device. They are stored as two float4.
        seeds_array = np.empty((n_streamlines, 8), dtype=np.float32)
        cache.seeds = cl.new_read_only_buffer(seeds_array)

        # Reserve space for the streamlines on the device. They are stored as float4.
        streamlines = np.zeros((n_streamlines, config.n_steps, 4), dtype=np.float32)
        cache.streamlines = cl.new_write_only_buffer(streamlines.nbytes)

        # Reserve space for the length of the streamlines on the device.
        lengths = np.zeros((n_streamlines,), dtype=np.uint32)
        cache.lengths = cl.new_write_only_buffer(lengths.nbytes)

        # Build the OpenCL program that implements tractography.
        values = {
            "nx": odf.shape[0],
            "ny": odf.shape[1],
            "nz": odf.shape[2],
            "n_directions": len(vertices),
            "n_steps": config.n_steps,
            "n_streamlines": n_streamlines,
        }
        cache.program = cl.build_program(values, "algorithms/deterministic/tractogram.cl")

    # Type narrowing.
    assert cache is not None

    # Move seeds to the device.
    seeds_array = seeds_module.to_array(seeds).astype(np.float32)
    cl.copy_to_buffer(cache.seeds, seeds_array)

    # Track streamlines.
    streamlines = np.zeros((n_streamlines, config.n_steps, 4), dtype=np.float32)
    lengths = np.zeros((n_streamlines,), dtype=np.uint32)
    args = (
        cache.values,
        cache.fod_inverse_affine,
        cache.directions,
        cache.seeds,
        np.float32(config.step_size),
        np.float32(np.cos(np.deg2rad(config.maximum_angle))),
        cache.streamlines,
        cache.lengths,
    )
    cl.run_tractogram(cache.program, args, n_streamlines)

    # Fetch the data.
    cl.copy_from_buffer(cache.streamlines, streamlines)
    cl.copy_from_buffer(cache.lengths, lengths)

    return Tractogram([streamlines[i, :n, :3] for i, n in enumerate(lengths)], affine_to_rasmm=np.eye(4)), cache


def connectome(fod, fod_affine, seed_fod, seed_fod_affine, vertices, vertex_labels, n_labels, n_seeds, config, distance_upper_bound=4.0):
    """Generate a connectome directly without storing streamlines.

    Each streamline is seeded from the FOD, propagated using deterministic tracking,
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

    # Determine the number of seeds per thread.
    n_threads = 1000
    n_seeds_per_thread = np.ceil(n_seeds / n_threads)

    # Prepare the data necessary for the seeds.
    mask = seed_fod[..., 0] > 0
    voxels = np.array(np.nonzero(mask), dtype=np.float32).T
    voxels = np.hstack((voxels, np.ones((len(voxels), 1))))
    inline_fod = fod[mask]
    randoms = np.random.randint(4294967295, size=(n_threads, 2))

    # Generate a set of orientations where the FODs are evaluated.
    n_directions = 300
    directions, fod_values = discretize_fod(fod, n_directions)
    directions_homogeneous = np.c_[directions, np.zeros((n_directions,))]

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

    # Vertex buffers: positions as float4 and labels as int32.
    vertices_homogeneous = np.c_[vertices, np.zeros((len(vertices), 1))]
    vertices_buffer = cl.new_read_only_buffer(vertices_homogeneous.astype(np.float32))
    vertex_labels_buffer = cl.new_read_only_buffer(vertex_labels.astype(np.int32))

    # Output connectivity matrix.
    conn_matrix = np.zeros((n_labels, n_labels), dtype=np.uint32)
    conn_matrix_buffer = cl.new_buffer(conn_matrix)

    # Determine the local model based on the data shape.
    model = LocalModel.from_shape(fod.shape)

    # Set constants in the OpenCL code.
    values = {
        "model": str(model),
        "nx": fod.shape[0],
        "ny": fod.shape[1],
        "nz": fod.shape[2],
        "nnz": len(voxels),
        "n_coefficients": fod.shape[-1],
        "n_steps": config.n_steps,
        "n_directions": n_directions,
        "n_seeds": n_threads,
        "n_labels": n_labels,
        "n_vertices": len(vertices),
    }
    program = cl.build_program(values, ["utils/seeds.cl", "algorithms/deterministic/connectome.cl"])

    args = (
        fod_values_buffer,
        fod_inverse_affine_buffer,
        directions_buffer,
        seed_fod_buffer,
        seed_fod_voxels_buffer,
        seed_fod_affine_buffer,
        randoms_buffer,
        vertices_buffer,
        vertex_labels_buffer,
        np.float32(config.step_size),
        np.float32(config.save_at),
        np.uint32(config.min_n_steps),
        np.float32(np.cos(np.deg2rad(config.maximum_angle))),
        np.float32(distance_upper_bound),
        np.uint32(n_seeds_per_thread),
        conn_matrix_buffer,
    )
    program.connectome(cl._queue, (n_threads,), None, *args)
    cl.copy_from_buffer(conn_matrix_buffer, conn_matrix)

    return conn_matrix


register(Algorithm.DETERMINISTIC, Configuration, tractogram, histogram, connectome)
