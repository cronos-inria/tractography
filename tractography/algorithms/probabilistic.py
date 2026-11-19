from dataclasses import dataclass
from typing import Tuple

from nibabel.nifti1 import Nifti1Image
from nibabel.streamlines.tractogram import Tractogram
import numpy as np
import pydantic

from ..seeds import Seed
from .. import core, seeds as seeds_module
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
        return super().load(Algorithm.PROBABILISTIC)


@dataclass
class Cache(BaseCache):
    """Cache for the OpenCL setup of the probabilistic algorithm.

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

    # The random number generator states, one per streamline.
    randoms: cl.Buffer | None = None


def tractogram(fod: Nifti1Image, seeds: list[Seed], config: Configuration, cache: Cache | None = None) -> Tuple[Tractogram, Cache]:
    """Generate streamlines using probabilistic tracking.

    Args:
        fod: The FOD image used to estimate the local tracking directions.
        seeds: Seed locations used to initialize each streamline.
        config: Probabilistic tracking configuration, including step size, maximum
            turning angle, and number of steps.
        cache: Optional OpenCL cache reused across calls when the input shape and
            configuration are unchanged.

    The cache stores the OpenCL buffers, random state, and compiled program used
    for tracking. When the input FOD shape, number of streamlines, and number of
    steps match the previous call, the existing cache is reused to avoid
    reallocating buffers and rebuilding the OpenCL program.

    Returns:
        A tuple containing the generated tractogram and the updated cache.

    Raises:
        ValueError: If the FOD image, seeds, or configuration are incompatible with
            probabilistic tracking or the cached OpenCL resources.
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

        fod_data = fod.get_fdata()
        affine = fod.affine if fod.affine is not None else np.eye(4)

        # Generate a set of orientations where the FODs are evaluated. On the
        # device the vertices are represented as float4.
        n_points = 300
        directions, fod_values = discretize_fod(fod_data, n_points)
        directions_homogeneous = np.c_[directions, np.zeros((n_points,))]
        cache.directions = cl.new_read_only_buffer(directions_homogeneous.astype(np.float32))

        # Precompute the inverse affine.
        iaffine = np.linalg.inv(affine).astype(np.float32)
        cache.fod_inverse_affine = cl.new_read_only_buffer(iaffine)

        # Use the discretized FOD values from the helper.
        cache.values = cl.new_read_only_buffer(fod_values.astype(np.float32))

        # Create the seed buffer on the device. They are stored as two float4.
        seeds_array = np.empty((n_streamlines, 8), dtype=np.float32)
        cache.seeds = cl.new_read_only_buffer(seeds_array)

        # Add the random number states.
        randoms_array = np.random.randint(4294967295, size=(n_streamlines, 2)).astype(np.uint32)
        cache.randoms = cl.new_read_only_buffer(randoms_array)

        # Reserve space for the streamlines on the device. They are stored as float4.
        streamlines = np.zeros((n_streamlines, config.n_steps, 4), dtype=np.float32)
        cache.streamlines = cl.new_write_only_buffer(streamlines.nbytes)

        # Reserve space for the length of the streamlines on the device.
        lengths = np.zeros((n_streamlines,), dtype=np.uint32)
        cache.lengths = cl.new_write_only_buffer(lengths.nbytes)

        # Build the OpenCL program that implements tractography.
        values = {
            "nx": fod_data.shape[0],
            "ny": fod_data.shape[1],
            "nz": fod_data.shape[2],
            "n_directions": len(directions),
            "n_steps": config.n_steps,
            "n_streamlines": n_streamlines,
        }
        cache.program = cl.build_program(values, "algorithms/probabilistic/tractogram.cl")

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
        cache.randoms,
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


register(Algorithm.PROBABILISTIC, Configuration, tractogram)
