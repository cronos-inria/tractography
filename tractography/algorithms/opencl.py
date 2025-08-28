from pathlib import Path
from string import Template

import numpy as np
import pyopencl as cl

# Create the global OpenCL context.
_context = cl.create_some_context(interactive=False)
_queue = cl.CommandQueue(_context)

# Get the number of compute units for the selected device.
_device = _context.devices[0]
_nb_units = _device.max_compute_units

_OPENCL_DIR = Path(__file__).parents[2] / "src"
_OPENCL_INCLUDE = f"-I {_OPENCL_DIR / 'include'}"


def new_read_only_buffer(data):
    """Create a new read only OpenCL buffer from data"""
    buffer = cl.Buffer(_context, cl.mem_flags.READ_ONLY, size=data.nbytes)
    cl.enqueue_copy(_queue, buffer, np.ascontiguousarray(data), is_blocking=False)
    return buffer


def build_program(values, name):

    # Compile the OpenCL program that implements Boltzmann tractography.
    with open(_OPENCL_DIR / name) as f:
        kernel = f.read()
    template = Template(kernel)

    # Set constants in the OpenCL code.
    source = template.safe_substitute(values)
    return cl.Program(_context, source).build(_OPENCL_INCLUDE)


def new_write_only_buffer(size):
    return cl.Buffer(_context, cl.mem_flags.WRITE_ONLY, size=size)


def copy_to_buffer(buffer, data):
    return cl.enqueue_copy(
        _queue, buffer, np.ascontiguousarray(data), is_blocking=False
    )


def copy_from_buffer(buffer, data):
    return cl.enqueue_copy(_queue, data, buffer, is_blocking=False)


def run_program(program, args, n_threads):
    return program.tractography(_queue, (n_threads,), None, *args)


def run_histogram(program, args, n_threads):
    return program.histogram(_queue, (n_threads,), None, *args)
