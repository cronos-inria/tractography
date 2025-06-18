import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyopencl as cl
import scipy.interpolate as si

import tractography as tg


_OPENCL_DIR = Path(__file__).parents[2] / "src"
_DATA_DIR = Path(__file__).parents[1] / "data"


class TestCore(unittest.TestCase):
    """Test the OpenCL implementation of core functions"""

    def test_randu(self):
        """Test the randu function"""


        # Create the global OpenCL context.
        _context = tg.algorithms.opencl._context
        _queue = tg.algorithms.opencl._queue

        flags = cl.mem_flags.WRITE_ONLY
        values = np.zeros((100000,)).astype(np.float32)
        values_buffer = cl.Buffer(_context, flags, size=values.nbytes)

        # Compile the OpenCL program that implements Boltzmann tractography.
        program = tg.algorithms.opencl.build_program(dict(), "test-boltzmann.cl")

        program.test_randu(
            _queue, (1,), None, values_buffer, np.uint32(len(values)),
        )
        cl.enqueue_copy(_queue, values, values_buffer)

        plt.hist(values, bins=100)
        plt.title("Uniform distribution")
        plt.savefig(_DATA_DIR / "uniform.png")

    def test_randn(self):
        """Test the randn function"""


        # Create the global OpenCL context.
        _context = tg.algorithms.opencl._context
        _queue = tg.algorithms.opencl._queue

        flags = cl.mem_flags.WRITE_ONLY
        values = np.zeros((100000,)).astype(np.float32)
        values_buffer = cl.Buffer(_context, flags, size=values.nbytes)

        # Compile the OpenCL program that implements Boltzmann tractography.
        program = tg.algorithms.opencl.build_program(dict(), "test-boltzmann.cl")

        program.test_randn(
            _queue, (1,), None, values_buffer, np.uint32(len(values)),
        )
        cl.enqueue_copy(_queue, values, values_buffer)
        
        plt.hist(values, bins=100)
        plt.title("Normal distribution")
        plt.savefig(_DATA_DIR / "normal.png")

