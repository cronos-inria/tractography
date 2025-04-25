from pathlib import Path
from string import Template

import numpy as np
import pyopencl as cl

import tractography as tg

# Create the global OpenCL context.
_context = cl.create_some_context(interactive=False)
_queue = cl.CommandQueue(_context)

# Get the number of compute units for the selected device.
_device = _context.devices[0]
_nb_units = _device.max_compute_units

_OPENCL_DIR = Path(__file__).parents[2] / "src"
_OPENCL_INCLUDE = f"-I {_OPENCL_DIR / 'include'}"


class Probabilistic:

    def __init__(self, fod, affine, n_streamlines, config):

        self._n_streamlines = n_streamlines
        self._config = config

        # Generate a set of orientation where the FODs are evaluated. On the
        # device the vertices are represented as float4.
        n_points = 400
        vertices = tg.core.fibonacci_sphere(n_points)
        device_vertices = np.c_[vertices, np.zeros((n_points,))]
        cl_vertices = _read_buffer(device_vertices.astype(np.float32))

        # Precompute the inverse affine.
        iaffine = np.linalg.inv(affine)
        cl_iaffine = _read_buffer(iaffine.astype(np.float32))

        # Convert the spherical harmonics to 1D probability mass functions.
        n_coefficients = fod.shape[-1]
        azimuths, colatitudes, _ = tg.core.cart2sph(*vertices.T)
        matrix, _ = tg.core.ishtmtx(azimuths, colatitudes, n_coefficients)
        fod_values = np.maximum(np.dot(fod.reshape((-1, n_coefficients)), matrix.T), 0)
        fod_values = fod_values.reshape((*fod.shape[:3], -1))
        cl_fod_values = _read_buffer(fod_values.astype(np.float32))

        # Create the seed buffer on the device. They are stored as two float4.
        seeds_array = np.empty((n_streamlines, 8), dtype=np.float32)
        cl_seeds = _read_buffer(seeds_array)

        # Pregenerate the random values.
        randoms = np.random.rand(n_streamlines, config.n_steps).astype(dtype=np.float32)
        cl_randoms = _read_buffer(randoms)

        # Reserve space for the streamlines on the device. The are
        # stored as float4.
        streamlines = np.zeros((n_streamlines, config.n_steps, 4), dtype=np.float32)
        flags = cl.mem_flags.WRITE_ONLY
        cl_streamlines = cl.Buffer(_context, flags, size=streamlines.nbytes)

        # Reserve space for the length of the streamlines on the device.
        lengths = np.zeros((n_streamlines,), dtype=np.uint32)
        flags = cl.mem_flags.WRITE_ONLY
        cl_lengths = cl.Buffer(_context, flags, size=lengths.nbytes)

        # Build the OpenCL program that implements tractography.
        with open(_OPENCL_DIR / "probabilistic.cl") as f:
            kernel = f.read()
        template = Template(kernel)

        # Set constants in the OpenCL code.
        values = {
            "nx": fod.shape[0],
            "ny": fod.shape[1],
            "nz": fod.shape[2],
            "n_directions": len(vertices),
            "n_steps": config.n_steps,
            "n_streamlines": n_streamlines,
        }
        source = template.safe_substitute(values)
        program = cl.Program(_context, source).build(_OPENCL_INCLUDE)

        # Keep what is needed to run the algorithm.
        self._values = cl_fod_values
        self._iaffine = cl_iaffine
        self._vertices = cl_vertices
        self._seeds = cl_seeds
        self._randoms = cl_randoms
        self._streamlines = cl_streamlines
        self._lengths = cl_lengths
        self._program = program

    def run(self, seeds):

        # Transfer the seeds to the buffer.
        seeds_array = tg.seeds.to_array(seeds).astype(np.float32)
        fillo = np.ones(self._n_streamlines, dtype=np.float32)
        fillz = np.zeros(self._n_streamlines, dtype=np.float32)
        seeds_array = np.c_[seeds_array[:, :3], fillo, seeds_array[:, 3:], fillz]
        cl.enqueue_copy(_queue, self._seeds, np.ascontiguousarray(seeds_array))

        # Track streamlines.
        args = (
            self._values,
            self._iaffine,
            self._vertices,
            self._seeds,
            np.float32(self._config.step_size),
            np.float32(
                np.cos(np.deg2rad(self._config.algorithms.probabilistic.maximum_angle))
            ),
            self._randoms,
            self._streamlines,
            self._lengths,
        )
        self._program.tractography(_queue, (self._n_streamlines,), None, *args)
        streamlines = np.zeros(
            (self._n_streamlines, self._config.n_steps, 4), dtype=np.float32
        )
        cl.enqueue_copy(_queue, streamlines, self._streamlines)
        lengths = np.zeros((self._n_streamlines,), dtype=np.uint32)
        cl.enqueue_copy(_queue, lengths, self._lengths)

        return [streamlines[i, :n, :3] for i, n in enumerate(lengths)]


def _read_buffer(data):
    """Create a new read only OpenCL buffer from data"""
    flags = cl.mem_flags.READ_ONLY
    buffer = cl.Buffer(_context, flags, size=data.nbytes)
    cl.enqueue_copy(_queue, buffer, np.ascontiguousarray(data))
    return buffer
