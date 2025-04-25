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


class FACT:

    def __init__(self, peaks, affine, n_streamlines, config):

        self._config = config
        self._n_streamlines = n_streamlines

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
        self._iaffine = cl.Buffer(_context, flags, size=iaffine.nbytes)
        cl.enqueue_copy(_queue, self._iaffine, np.ascontiguousarray(iaffine))

        # Augment the vectors to 4 elements because OpenCL only has float4.
        peaks = np.pad(peaks.astype(np.float32), ((0, 0),) * 4 + ((0, 1),))
        self._peaks = cl.Buffer(_context, flags, size=peaks.nbytes)
        cl.enqueue_copy(_queue, self._peaks, np.ascontiguousarray(peaks))

        # Also augment the seeds array to two float4 per seed.
        seeds_array = np.ones((n_streamlines, 8), dtype=np.float32)
        self._seeds = cl.Buffer(_context, flags, size=seeds_array.nbytes)
        cl.enqueue_copy(_queue, self._seeds, np.ascontiguousarray(seeds_array))

        # Reserve space for the streamlines on the device. Streamlines are also float4.
        streamlines = np.zeros((n_streamlines, config.n_steps, 4), dtype=np.float32)
        flags = cl.mem_flags.WRITE_ONLY
        self._streamlines = cl.Buffer(_context, flags, size=streamlines.nbytes)

        # Reserve space for the length of the streamlines on the device.
        lengths = np.zeros((n_streamlines,), dtype=np.uint32)
        flags = cl.mem_flags.WRITE_ONLY
        self._lengths = cl.Buffer(_context, flags, size=lengths.nbytes)

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
        self._program = cl.Program(_context, source).build(_OPENCL_INCLUDE)

    def run(self, seeds):

        # Transfer the seeds to the buffer.
        seeds_array = tg.seeds.to_array(seeds).astype(np.float32)
        fillo = np.ones(self._n_streamlines, dtype=np.float32)
        fillz = np.zeros(self._n_streamlines, dtype=np.float32)
        seeds_array = np.c_[seeds_array[:, :3], fillo, seeds_array[:, 3:], fillz]
        cl.enqueue_copy(_queue, self._seeds, np.ascontiguousarray(seeds_array))

        # Track streamlines.
        args = (
            self._peaks,
            self._iaffine,
            self._seeds,
            np.float32(self._config.step_size),
            np.float32(np.cos(np.deg2rad(self._config.algorithms.fact.maximum_angle))),
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
