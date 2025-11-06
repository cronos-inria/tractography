import unittest

import numpy as np

import tractography as tg


class TestCore(unittest.TestCase):

    def test_randu(self):

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
