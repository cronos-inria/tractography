import unittest
from pathlib import Path

import numpy as np
import pyopencl as cl

import tractography as tg

_OPENCL_DIR = Path(__file__).parents[2] / "src"
_DATA_DIR = Path(__file__).parents[1] / "data"


class TestCore(unittest.TestCase):
    """Test the OpenCL implementation of core functions"""

    def test_randu(self):
        """Test OpenCL generation of uniform floats"""

        # Compile the OpenCL program that implements Boltzmann tractography.
        program = tg.algorithms.opencl.build_program(dict(), "utils/core.cl")

        n_values = 1000000
        values = np.zeros((n_values,), dtype=np.float32)
        values_buffer = tg.algorithms.opencl.new_write_only_buffer(values.nbytes)
        program.randus(
            tg.algorithms.opencl._queue,
            (1,),
            None,
            values_buffer,
            np.uint32(n_values),
        )
        tg.algorithms.opencl.copy_from_buffer(values_buffer, values)
        self.assertTrue(values.min() > 0)
        self.assertTrue(values.max() < 1)

    def test_randi(self):
        """Test OpenCL generation of uniform integers"""

        # Compile the OpenCL program that implements Boltzmann tractography.
        program = tg.algorithms.opencl.build_program(dict(), "utils/core.cl")

        n_values = 1000000
        values = np.zeros((n_values,), dtype=np.uint32)
        values_buffer = tg.algorithms.opencl.new_write_only_buffer(values.nbytes)
        program.randis(
            tg.algorithms.opencl._queue,
            (1,),
            None,
            values_buffer,
            np.uint32(n_values),
            np.uint32(100),
        )
        tg.algorithms.opencl.copy_from_buffer(values_buffer, values)
        self.assertTrue(values.min() >= 0)
        self.assertTrue(values.max() <= 99)


class TestSpharm(unittest.TestCase):
    """Test the OpenCL implementation of spherical harmonic functions"""

    def test_ishtmtx(self):

        n = 100
        azimuths = np.linspace(0, 2 * np.pi - 2 * np.pi / n, n).astype(np.float32)
        colatitudes = np.linspace(0, np.pi - np.pi / n, n).astype(np.float32)
        values = tg.core.ishtmtx(azimuths, colatitudes, 45)[0].astype(np.float32)

        # Create the global OpenCL context.
        _context = tg.algorithms.opencl._context
        _queue = tg.algorithms.opencl._queue

        flags = cl.mem_flags.READ_ONLY
        azimuths_buffer = cl.Buffer(_context, flags, size=azimuths.nbytes)
        cl.enqueue_copy(_queue, azimuths_buffer, azimuths)

        flags = cl.mem_flags.READ_ONLY
        colatitudes_buffer = cl.Buffer(_context, flags, size=colatitudes.nbytes)
        cl.enqueue_copy(_queue, colatitudes_buffer, colatitudes)

        flags = cl.mem_flags.READ_WRITE
        values_buffer = cl.Buffer(_context, flags, size=values.nbytes)

        # Compile the OpenCL program that implements Boltzmann tractography.
        program = tg.algorithms.opencl.build_program(dict(), "utils/spharm.cl")

        program.test_ishtmtx(
            _queue,
            (1,),
            None,
            azimuths_buffer,
            colatitudes_buffer,
            np.uint32(len(azimuths)),
            values_buffer,
        )
        new_values = np.zeros(values.shape, np.float32)
        cl.enqueue_copy(_queue, new_values, values_buffer)
        np.testing.assert_array_almost_equal(new_values, values, 5)

    def test_ishtmtx_dt(self):

        n = 100
        azimuths = np.linspace(0, 2 * np.pi - 2 * np.pi / n, n).astype(np.float32)
        colatitudes = np.linspace(0, np.pi - np.pi / n, n).astype(np.float32)
        values = tg.core.ishtmtx(azimuths, colatitudes, 45)[1][0].astype(np.float32)

        # Create the global OpenCL context.
        _context = tg.algorithms.opencl._context
        _queue = tg.algorithms.opencl._queue

        flags = cl.mem_flags.READ_ONLY
        azimuths_buffer = cl.Buffer(_context, flags, size=azimuths.nbytes)
        cl.enqueue_copy(_queue, azimuths_buffer, azimuths)

        flags = cl.mem_flags.READ_ONLY
        colatitudes_buffer = cl.Buffer(_context, flags, size=colatitudes.nbytes)
        cl.enqueue_copy(_queue, colatitudes_buffer, colatitudes)

        flags = cl.mem_flags.READ_WRITE
        values_buffer = cl.Buffer(_context, flags, size=values.nbytes)

        # Compile the OpenCL program that implements Boltzmann tractography.
        program = tg.algorithms.opencl.build_program(dict(), "utils/spharm.cl")

        program.test_ishtmtx_dt(
            _queue,
            (1,),
            None,
            azimuths_buffer,
            colatitudes_buffer,
            np.uint32(len(azimuths)),
            values_buffer,
        )
        new_values = np.zeros(values.shape, np.float32)
        cl.enqueue_copy(_queue, new_values, values_buffer)
        np.testing.assert_array_almost_equal(new_values, values, 5)

    def test_ishtmtx_dp(self):

        n = 100
        azimuths = np.linspace(0, 2 * np.pi - 2 * np.pi / n, n).astype(np.float32)
        colatitudes = np.linspace(0, np.pi - np.pi / n, n).astype(np.float32)
        values = tg.core.ishtmtx(azimuths, colatitudes, 45)[1][1].astype(np.float32)

        # Create the global OpenCL context.
        _context = tg.algorithms.opencl._context
        _queue = tg.algorithms.opencl._queue

        flags = cl.mem_flags.READ_ONLY
        azimuths_buffer = cl.Buffer(_context, flags, size=azimuths.nbytes)
        cl.enqueue_copy(_queue, azimuths_buffer, azimuths)

        flags = cl.mem_flags.READ_ONLY
        colatitudes_buffer = cl.Buffer(_context, flags, size=colatitudes.nbytes)
        cl.enqueue_copy(_queue, colatitudes_buffer, colatitudes)

        flags = cl.mem_flags.READ_WRITE
        values_buffer = cl.Buffer(_context, flags, size=values.nbytes)

        # Compile the OpenCL program that implements Boltzmann tractography.
        program = tg.algorithms.opencl.build_program(dict(), "utils/spharm.cl")

        program.test_ishtmtx_dp(
            _queue,
            (1,),
            None,
            azimuths_buffer,
            colatitudes_buffer,
            np.uint32(len(azimuths)),
            values_buffer,
        )
        new_values = np.zeros(values.shape, np.float32)
        cl.enqueue_copy(_queue, new_values, values_buffer)
        np.testing.assert_array_almost_equal(new_values, values, 5)
