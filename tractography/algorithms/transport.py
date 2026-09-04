from dataclasses import dataclass
from typing import Tuple

from nibabel.nifti1 import Nifti1Image
from nibabel.streamlines.tractogram import Tractogram
import numpy as np
import pydantic

from ..seeds import Seed
from .. import seeds as seeds_module, utils
from . import opencl as cl

from .core import (
    Algorithm,
    BaseCache,
    BaseConfiguration,
    LocalModel,
    cache_needs_rebuild,
)
from . import register


class Configuration(BaseConfiguration):
    inverse_curvature: pydantic.PositiveFloat

    @classmethod
    def load(cls):
        return super().load(Algorithm.TRANSPORT)
    

@dataclass
class Cache(BaseCache):
    """Cache for the OpenCL setup of the transport algorithm.
    
    This cache is used to store the OpenCL buffers and program, so that they can
    be reused across multiple tractography runs without needing to reallocate
    memory or rebuild the program.

    """

    # The model parameters modelling the local FOD (e.g. spherical harmonics coefficients).
    fod: cl.Buffer | None = None

    # The affine transform from world to voxel space.
    fod_inverse_affine: cl.Buffer | None = None

    # The mask of the FOD, where streamlines can propagate.
    mask: cl.Buffer | None = None
    mask_shape: cl.Buffer | None = None
    mask_affine: cl.Buffer | None = None

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
    mask = np.any(seed_fod != 0, axis=-1)
    voxels = np.array(np.nonzero(mask), dtype=np.float32).T
    voxels = np.hstack((voxels, np.ones((len(voxels), 1))))
    inline_fod = fod[mask]
    randoms = np.random.randint(4294967295, size=(n_threads, 2))

    # Determine the local model based on the data shape.
    model = LocalModel.from_shape(fod.shape)

    # Send the data to the device.
    fod_buffer = cl.new_read_only_buffer(fod.astype(np.float32))
    fod_inverse_affine_buffer = cl.new_read_only_buffer(np.linalg.inv(fod_affine).astype(np.float32))
    seed_fod_buffer = cl.new_read_only_buffer(inline_fod.astype(np.float32))
    seed_fod_voxels_buffer = cl.new_read_only_buffer(voxels.astype(np.float32))
    seed_fod_affine_buffer = cl.new_read_only_buffer(seed_fod_affine.astype(np.float32))
    randoms_buffer = cl.new_buffer(randoms.astype(np.uint32))

    hist = np.zeros(fod.shape[:3] + (45,), dtype=np.float32)
    hist_buffer = cl.new_buffer(hist)

    # Set constants in the OpenCL code.
    values = {
        "model": str(model),
        "nx": fod.shape[0],
        "ny": fod.shape[1],
        "nz": fod.shape[2],
        "nnz": len(voxels),
        "n_coefficients": fod.shape[-1],
        "n_steps": config.n_steps,
        "n_seeds": n_threads,
    }
    program = cl.build_program(values, ["utils/seeds.cl", "algorithms/transport/histogram.cl"])

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
        np.uint32(n_seeds_per_thread),
        hist_buffer,
    )
    cl.run_histogram(program, args, n_threads)
    cl.copy_from_buffer(hist_buffer, hist)

    hist = utils.normalize_odf(hist)
    return hist


def tractogram(
    fod: Nifti1Image,
    seeds: list[Seed],
    config: Configuration,
    cache: Cache | None = None
) -> Tuple[Tractogram, Cache]:
    """Generate streamlines using transport tracking.

    Args:
        fod: The FOD image used to estimate the local tracking directions.
        seeds: Seed locations used to initialize each streamline.
        config: Transport tracking configuration, including step size, inverse
            curvature, and number of steps.
        cache: Optional OpenCL cache reused across calls when the input shape and
            configuration are unchanged.

    The cache stores the OpenCL buffers and compiled program used for tracking.
    When the input FOD shape, number of streamlines, and number of steps match the
    previous call, the existing cache is reused to avoid reallocating buffers and
    rebuilding the OpenCL program.

    Returns:
        A tuple containing the generated tractogram and the updated cache.

    Raises:
        ValueError: If the FOD image, seeds, or configuration are incompatible with
            transport tracking or the cached OpenCL resources.
        np.linalg.LinAlgError: If an affine matrix cannot be inverted.
        RuntimeError: If OpenCL program compilation or kernel execution fails.

    """

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

        # Precompute the inverse affine.
        affine = fod.affine if fod.affine is not None else np.eye(4)
        iaffine = np.linalg.inv(affine).astype(np.float32)
        cache.fod_inverse_affine = cl.new_read_only_buffer(iaffine)

        # The FOD model is determined by the data shape. For example, if the last 
        # dimension has 45 coefficients, we assume a spherical harmonics model with 
        # 45 coefficients.
        model = LocalModel.from_shape(fod.shape)
        cache.fod = cl.new_read_only_buffer(fod.get_fdata().astype(np.float32))

        # The mask is where the FOD is non-zero.
        # We will use this to determine where streamlines can propagate.
        mask = (fod.get_fdata()[..., 0] > 0).astype(np.uint8)
        cache.mask = cl.new_read_only_buffer(mask)
        cache.mask_shape = cl.new_read_only_buffer(np.array(mask.shape + (0,), dtype=np.uint32))
        cache.mask_affine = cl.new_read_only_buffer(iaffine)

        # Create the seed buffer on the device. They are stored as two float4.
        seeds_array = np.empty((n_streamlines, 8), dtype=np.float32)
        cache.seeds = cl.new_read_only_buffer(seeds_array)

        # Reserve space for the streamlines on the device. The are stored as float4.
        streamlines = np.zeros((n_streamlines, config.n_steps, 4), dtype=np.float32)
        cache.streamlines = cl.new_write_only_buffer(streamlines.nbytes)

        # Reserve space for the length of the streamlines on the device.
        lengths = np.zeros((n_streamlines,), dtype=np.uint32)
        cache.lengths = cl.new_write_only_buffer(lengths.nbytes)

        # Set constants in the OpenCL code and build the program.
        values = {
            "nx": fod.shape[0],
            "ny": fod.shape[1],
            "nz": fod.shape[2],
            "n_coefficients": fod.shape[-1],
            "n_steps": config.n_steps,
            "n_streamlines": n_streamlines,
        }

        # The model and the field type will determine the correct OpenCL
        # implementation at compile time.
        cache.program = cl.build_program(
            values,
            "algorithms/transport/tractogram.cl",
            options=[f"-D {str(model)}", "-D FIELD_IMAGE"])

    # Type narrowing.
    assert cache is not None

    # Host buffers are always refreshed at call time.
    streamlines = np.zeros((n_streamlines, config.n_steps, 4), dtype=np.float32)
    lengths = np.zeros((n_streamlines,), dtype=np.uint32)

    # Move seeds to the device.
    seeds_array = seeds_module.to_array(seeds).astype(np.float32)
    cl.copy_to_buffer(cache.seeds, seeds_array)

    # Track streamlines.
    args = (
        cache.fod,
        cache.fod_inverse_affine,
        cache.mask,
        cache.mask_shape,
        cache.mask_affine,
        cache.seeds,
        np.float32(config.step_size),
        np.float32(config.save_at),
        np.float32(config.inverse_curvature),
        cache.streamlines,
        cache.lengths,
    )
    cl.run_tractogram(cache.program, args, n_streamlines)

    # Fetch the data.
    cl.copy_from_buffer(cache.streamlines, streamlines)
    cl.copy_from_buffer(cache.lengths, lengths)

    return Tractogram([streamlines[i, :n, :3] for i, n in enumerate(lengths)], affine_to_rasmm=np.eye(4)), cache


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

    Each streamline is seeded from the FOD, propagated using transport tracking,
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
        
    # Determine the local model based on the data shape.
    model = LocalModel.from_shape(fod.shape)

    fod_buffer = cl.new_read_only_buffer(fod.astype(np.float32))
    fod_inverse_affine_buffer = cl.new_read_only_buffer(
        np.linalg.inv(fod_affine).astype(np.float32)
    )
    seed_fod_buffer = cl.new_read_only_buffer(inline_fod.astype(np.float32))
    seed_fod_voxels_buffer = cl.new_read_only_buffer(voxels.astype(np.float32))
    seed_fod_affine_buffer = cl.new_read_only_buffer(seed_fod_affine.astype(np.float32))
    randoms_buffer = cl.new_buffer(randoms.astype(np.uint32))

    vertices_homogeneous = np.c_[vertices, np.zeros((len(vertices), 1))]
    vertices_buffer = cl.new_read_only_buffer(vertices_homogeneous.astype(np.float32))
    vertex_labels_buffer = cl.new_read_only_buffer(vertex_labels.astype(np.int32))

    conn_matrix = np.zeros((n_labels, n_labels), dtype=np.uint32)
    conn_matrix_buffer = cl.new_buffer(conn_matrix)

    values = {
        "model": str(model),
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
    program = cl.build_program(values, ["utils/seeds.cl", "algorithms/transport/connectome.cl"])

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
        np.float32(distance_upper_bound),
        np.uint32(n_seeds_per_thread),
        conn_matrix_buffer,
    )
    program.connectome(cl._queue, (n_threads,), None, *args)
    cl.copy_from_buffer(conn_matrix_buffer, conn_matrix)

    return conn_matrix


register(Algorithm.TRANSPORT, Configuration, tractogram, histogram, connectome)
