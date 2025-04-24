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


def opencl(peaks, affine, seeds, config):

    # Normalize the peaks.
    n_peaks = peaks.shape[-1] // 3
    peaks = peaks.reshape(peaks.shape[:3] + (3, n_peaks), order="F")
    peaks = np.transpose(peaks, (0, 1, 2, 4, 3))
    norms = np.linalg.norm(peaks, axis=-1, keepdims=True)
    peaks = np.divide(peaks, norms, where=norms != 0)

    # Precompute the inverse affine.
    iaffine = np.linalg.inv(affine)

    # Send readonly data to the device.
    flags = cl.mem_flags.READ_ONLY

    iaffine = iaffine.astype(np.float32)
    affine_buffer = cl.Buffer(_context, flags, size=iaffine.nbytes)
    cl.enqueue_copy(_queue, affine_buffer, np.ascontiguousarray(iaffine))

    # Augment the vectors to 4 elements because OpenCL only has float4.
    peaks = np.pad(peaks.astype(np.float32), ((0, 0),) * 4 + ((0, 1),))
    peaks_buffer = cl.Buffer(_context, flags, size=peaks.nbytes)
    cl.enqueue_copy(_queue, peaks_buffer, np.ascontiguousarray(peaks))

    # Also augment the seeds array to two float4 per seed.
    n_streamlines = len(seeds)
    seeds_array = tg.seeds.to_array(seeds).astype(np.float32)
    fill = np.ones(n_streamlines, dtype=np.float32)
    seeds_array = np.c_[seeds_array[:, :3], fill, seeds_array[:, 3:], fill]
    seeds_buffer = cl.Buffer(_context, flags, size=seeds_array.nbytes)
    cl.enqueue_copy(_queue, seeds_buffer, np.ascontiguousarray(seeds_array))

    # Reserve space for the streamlines on the device. Streamlines are also float4.
    streamlines = np.zeros((n_streamlines, config.n_steps, 4), dtype=np.float32)
    flags = cl.mem_flags.WRITE_ONLY
    streamlines_buffer = cl.Buffer(_context, flags, size=streamlines.nbytes)

    # Compile the OpenCL program that implements Boltzmann tractography.
    with open(_OPENCL_DIR / "fact.cl") as f:
        kernel = f.read()
    template = Template(kernel)

    # Set constants in the OpenCL code.
    values = {
        "nx": peaks.shape[0],
        "ny": peaks.shape[1],
        "nz": peaks.shape[2],
        "n_peaks": n_peaks,
        "n_steps": config.n_steps,
        "n_streamlines": n_streamlines,
    }
    source = template.safe_substitute(values)
    program = cl.Program(_context, source).build(_OPENCL_INCLUDE)

    # Track streamlines.
    args = (
        peaks_buffer,
        affine_buffer,
        seeds_buffer,
        np.float32(config.step_size),
        np.float32(np.cos(np.deg2rad(config.algorithms.fact.maximum_angle))),
        streamlines_buffer,
    )
    program.tractography(_queue, (n_streamlines,), None, *args)
    cl.enqueue_copy(_queue, streamlines, streamlines_buffer)

    return streamlines[..., :3]
