from dataclasses import dataclass
from typing import Tuple

from nibabel.nifti1 import Nifti1Image
from nibabel.streamlines.tractogram import Tractogram
import numpy as np
import pydantic

from ..seeds import Seed
from .. import seeds as seeds_module
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
    noise_variance: pydantic.PositiveFloat

    @classmethod
    def load(cls):
        return super().load(Algorithm.DIFFUSION)


@dataclass
class Cache(BaseCache):
    """Cache for the OpenCL setup of the diffusion algorithm.

    This cache is used to store the OpenCL buffers and program, so that they can
    be reused across multiple tractography runs without needing to reallocate
    memory or rebuild the program.

    """

    # The model parameters modelling the local FOD (e.g. spherical harmonics coefficients).
    fod: cl.Buffer | None = None

    # The affine transform from world to voxel space.
    fod_inverse_affine: cl.Buffer | None = None

    # The random number generator states, one per streamline.
    randoms: cl.Buffer | None = None

    # The FOD data dimensions.
    dims: cl.Buffer | None = None


def tractogram(fod: Nifti1Image, seeds: list[Seed], config: Configuration, cache: Cache | None = None) -> Tuple[Tractogram, Cache]:
    """Generate streamlines using diffusion tracking.

    Args:
        fod: The FOD image used to estimate the local tracking directions.
        seeds: Seed locations used to initialize each streamline.
        config: Diffusion tracking configuration, including step size, maximum
            turning angle, inverse curvature, noise variance, and number of steps.
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
            diffusion tracking or the cached OpenCL resources.
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

        # Add the random number states.
        randoms_array = np.random.randint(4294967295, size=(n_streamlines, 2)).astype(np.uint32)
        cache.randoms = cl.new_read_only_buffer(randoms_array)

        # Create the seed buffer on the device. They are stored as two float4.
        seeds_array = np.empty((n_streamlines, 8), dtype=np.float32)
        cache.seeds = cl.new_read_only_buffer(seeds_array)

        # Create the buffer for data dimensions.
        dims = np.array(fod.shape, dtype=np.uint32)
        cache.dims = cl.new_read_only_buffer(dims)

        # Reserve space for the streamlines on the device. The are stored as float4.
        streamlines = np.zeros((n_streamlines, config.n_steps, 4), dtype=np.float32)
        cache.streamlines = cl.new_write_only_buffer(streamlines.nbytes)

        # Reserve space for the length of the streamlines on the device.
        lengths = np.zeros((n_streamlines,), dtype=np.uint32)
        cache.lengths = cl.new_write_only_buffer(lengths.nbytes)

        # Set constants in the OpenCL code and build the program.
        values = {
            "model": str(model),
            "nx": fod.shape[0],
            "ny": fod.shape[1],
            "nz": fod.shape[2],
            "n_coefficients": fod.shape[-1],
            "n_steps": config.n_steps,
            "n_streamlines": n_streamlines,
        }
        cache.program = cl.build_program(values, "algorithms/diffusion/tractogram.cl")

    # Type narrowing.
    assert cache is not None

    # Move seeds to the device.
    seeds_array = seeds_module.to_array(seeds).astype(np.float32)
    cl.copy_to_buffer(cache.seeds, seeds_array)

    # Track streamlines.
    streamlines = np.zeros((n_streamlines, config.n_steps, 4), dtype=np.float32)
    lengths = np.zeros((n_streamlines,), dtype=np.uint32)
    args = (
        cache.fod,
        cache.dims,
        cache.fod_inverse_affine,
        cache.seeds,
        cache.randoms,
        np.float32(config.step_size),
        np.float32(config.save_at),
        np.float32(config.inverse_curvature),
        np.float32(config.noise_variance),
        cache.streamlines,
        cache.lengths,
    )
    cl.run_tractogram(cache.program, args, n_streamlines)

    # Fetch the data.
    cl.copy_from_buffer(cache.streamlines, streamlines)
    cl.copy_from_buffer(cache.lengths, lengths)

    return Tractogram([streamlines[i, :n, :3] for i, n in enumerate(lengths)], affine_to_rasmm=np.eye(4)), cache


register(Algorithm.DIFFUSION, Configuration, tractogram)
