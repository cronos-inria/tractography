import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyopencl as cl
import scipy.interpolate as si

import tractography as tg


_OPENCL_DIR = Path(__file__).parents[2] / "src"
_TEST_RESULTS_DIR = Path(__file__).parents[2] / "test-results" / "algorithms" / "core"


class TestCore(unittest.TestCase):
    """Test the OpenCL implementation of core functions"""

    def setUp(self):
        _TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def test_cart2sph(self):
        """Test the cart2sph function"""

        azimuths = np.linspace(0, 2 * np.pi - 2 * np.pi / 1000, 1000).astype(np.float32)
        colatitudes = np.linspace(0, np.pi - np.pi / 1000, 1000).astype(np.float32)
        ex, ey, ez = zip(
            *[tg.core.sph2cart(a, c, 1) for a, c in zip(azimuths, colatitudes)]
        )
        cart = np.c_[ex, ey, ez, np.zeros_like(ex)].astype(np.float32)
        sph = np.c_[azimuths, colatitudes].astype(np.float32)

        # Create the global OpenCL context.
        _context = tg.algorithms.opencl._context
        _queue = tg.algorithms.opencl._queue

        flags = cl.mem_flags.READ_ONLY
        cart_buffer = cl.Buffer(_context, flags, size=cart.nbytes)
        cl.enqueue_copy(_queue, cart_buffer, cart)

        flags = cl.mem_flags.WRITE_ONLY
        sph_buffer = cl.Buffer(_context, flags, size=sph.nbytes)

        # Compile the OpenCL program that implements Boltzmann tractography.
        program = tg.algorithms.opencl.build_program(dict(), "test-boltzmann.cl")

        program.test_cart2sph(
            _queue,
            (1,),
            None,
            cart_buffer,
            np.int32(len(cart)),
            sph_buffer,
        )
        new_sph = np.zeros_like(sph)
        cl.enqueue_copy(_queue, new_sph, sph_buffer)
        np.testing.assert_array_almost_equal(new_sph, sph, 3)

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
            _queue,
            (1,),
            None,
            values_buffer,
            np.uint32(len(values)),
        )
        cl.enqueue_copy(_queue, values, values_buffer)

        plt.hist(values, bins=100)
        plt.title("Uniform distribution")
        plt.savefig(_TEST_RESULTS_DIR / "uniform.png")
        plt.close()

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
            _queue,
            (1,),
            None,
            values_buffer,
            np.uint32(len(values)),
        )
        cl.enqueue_copy(_queue, values, values_buffer)

        plt.hist(values, bins=100)
        plt.title("Normal distribution")
        plt.savefig(_TEST_RESULTS_DIR / "normal.png")
        plt.close()
