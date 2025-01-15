from pathlib import Path
from string import Template

import numpy as np
import pyopencl as cl
import trimesh

import tractography as tg


# Create the global OpenCL context.
_context = cl.create_some_context(interactive=False)
_queue = cl.CommandQueue(_context)

# Get the number of compute units for the selected device.
_device = _context.devices[0]
_nb_units = _device.max_compute_units

_OPENCL_DIR = Path(__file__).parents[2] / "src"


def _opencl_algorithm(name, fod_sampling, prepare_args):
    """Generates an OpenCL tracking algorithm"""

    def algorithm(fod, affine, seeds, step_size, n_steps, max_angle):
        """Generic OpenCL tractography algorithm"""

        # Generate a set of orientation where the FODs are evaluated.
        fod_values, vertices = fod_sampling(fod)

        # Prepare the arguments for the kernel and the required batch
        # data.
        args, batch_data = prepare_args(
            fod_values, affine, vertices, seeds, step_size, n_steps, max_angle
        )

        # Streamlines are generated in batches, allocate the required output
        # buffer and the corresponding one for the seeds.
        n_streamlines = len(seeds)
        n_batches = n_streamlines // tg.BATCH_SIZE
        streamlines = [
            np.empty((tg.BATCH_SIZE, n_steps, 3), dtype=np.float32)
            for _ in range(n_batches)
        ]
        streamline_cl = _write_buffer(streamlines[0])
        args += [streamline_cl]

        # The OpenCL program that implements tractography.
        with open(_OPENCL_DIR / f"{name}.cl") as f:
            kernel = f.read()
        template = Template(kernel)

        # Set constants in the OpenCL code.
        values = {
            "nx": fod.shape[0],
            "ny": fod.shape[1],
            "nz": fod.shape[2],
            "n_directions": len(vertices),
            "n_steps": n_steps,
            "n_streamlines": n_streamlines,
        }
        source = template.safe_substitute(values)
        program = cl.Program(_context, source).build()

        for s, *batch_datum in zip(streamlines, *batch_data):

            # Update the required buffers for every batch.
            for b, c in batch_datum:
                _update_buffer(b, c)

            # Track streamlines.
            program.tractography(_queue, (tg.BATCH_SIZE,), None, *args)
            cl.enqueue_copy(_queue, s, streamline_cl)

        return np.vstack(streamlines)

    return algorithm


def _sample_icosphere(fod, n_upsampling=2):
    """Sample the FODs on the vertices of an icosphere"""

    n_coefficients = fod.shape[-1]

    vertices = trimesh.creation.icosphere(n_upsampling).vertices
    azimuths, colatitudes, _ = tg.core.cart2sph(*vertices.T)
    matrix = tg.core.ishtmtx(azimuths, colatitudes, n_coefficients)

    # Convert the spherical harmonics to 1D probability mass functions.
    fod_values = np.maximum(np.dot(fod.reshape((-1, n_coefficients)), matrix.T), 0)
    fod_values = fod_values.reshape((*fod.shape[:3], -1))

    return fod_values, vertices


def _read_buffer(data):
    """Create a new read only OpenCL buffer from data"""
    flags = cl.mem_flags.READ_ONLY
    buffer = cl.Buffer(_context, flags, size=data.nbytes)
    cl.enqueue_copy(_queue, buffer, np.ascontiguousarray(data))
    return buffer


def _update_buffer(data, buffer):
    """Update a read only buffer with new data"""
    cl.enqueue_copy(_queue, buffer, np.ascontiguousarray(data))


def _write_buffer(data):
    """Create a new write OpenCL buffer from data"""
    flags = cl.mem_flags.READ_ONLY
    buffer = cl.Buffer(_context, flags, size=data.nbytes)
    return buffer


def _args_det(fod_values, affine, vertices, seeds, step_size, n_steps, max_angle):

    # Send all global data to the device except for the seed information.
    fod_values_cl = _read_buffer(fod_values.astype(np.float32))
    affine_cl = _read_buffer(np.linalg.inv(affine).astype(np.float32))
    vertices_cl = _read_buffer(vertices.astype(np.float32))

    seeds_array = tg.seeds.to_array(seeds).astype(np.float32)
    seeds_cl = _read_buffer(seeds_array[:tg.BATCH_SIZE])

    max_angle_ratio = np.cos(np.deg2rad(max_angle))
    args = [
        fod_values_cl,
        affine_cl,
        vertices_cl,
        seeds_cl,
        np.float32(step_size),
        np.float32(max_angle_ratio),
    ]

    n_batches = len(seeds) / tg.BATCH_SIZE
    batch_data = [
        [(d, seeds_cl) for d in np.split(seeds_array, n_batches)],
    ]

    return args, batch_data


def _args_prob(fod_values, affine, vertices, seeds, step_size, n_steps, max_angle):

    # The arguments are the same as for the deterministic case, but with
    # the random values added.
    args, batch_data = _args_det(
        fod_values, affine, vertices, seeds, step_size, n_steps, max_angle
    )

    randoms = np.random.rand(len(seeds), n_steps).astype(dtype=np.float32)
    randoms_cl = _read_buffer(randoms[:tg.BATCH_SIZE])
    args.append(randoms_cl)

    n_batches = len(seeds) / tg.BATCH_SIZE
    batch_data.append([(d, randoms_cl) for d in np.split(randoms, n_batches)])

    return args, batch_data


# Define two simple tractography algorithms using the generic implementation.
deterministic = _opencl_algorithm("deterministic", _sample_icosphere, _args_det)
probabilistic = _opencl_algorithm("probabilistic", _sample_icosphere, _args_prob)
