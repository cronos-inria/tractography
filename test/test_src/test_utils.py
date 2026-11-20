import unittest

import numpy as np

import tractography as tg
from tractography.opencl import BufferAccess, OpenCLRuntime


class TestCore(unittest.TestCase):
    """Test the OpenCL implementation of core functions"""

    @classmethod
    def setUpClass(cls):
        cls.runtime = OpenCLRuntime()

    def test_randu(self):
        """Test OpenCL generation of uniform floats"""

        # Compile the OpenCL program that implements Boltzmann tractography.
        program = self.runtime.build_kernel_program("utils/core.cl")

        n_values = 1000000
        values = np.zeros((n_values,), dtype=np.float32)
        values_buffer = self.runtime.allocate_buffer(values.nbytes)
        program.randus(
            self.runtime.queue,
            (1,),
            None,
            values_buffer,
            np.uint32(n_values),
        )
        self.runtime.copy_from_buffer(values_buffer, values, is_blocking=True)
        self.assertTrue(values.min() > 0)
        self.assertTrue(values.max() < 1)

    def test_randi(self):
        """Test OpenCL generation of uniform integers"""

        # Compile the OpenCL program that implements Boltzmann tractography.
        program = self.runtime.build_kernel_program("utils/core.cl")

        n_values = 1000000
        values = np.zeros((n_values,), dtype=np.uint32)
        values_buffer = self.runtime.allocate_buffer(values.nbytes)
        program.randis(
            self.runtime.queue,
            (1,),
            None,
            values_buffer,
            np.uint32(n_values),
            np.uint32(100),
        )
        self.runtime.copy_from_buffer(values_buffer, values, is_blocking=True)
        self.assertTrue(values.min() >= 0)
        self.assertTrue(values.max() <= 99)


class TestSpharm(unittest.TestCase):
    """Test the OpenCL implementation of spherical harmonic functions"""

    @classmethod
    def setUpClass(cls):
        cls.runtime = OpenCLRuntime()

    def test_ishtmtx_antipodal_symmetry(self):

        n = 20
        directions = tg.core.fibonacci_sphere(n)
        azimuths, colatitudes, _ = tg.core.cart2sph(*directions.T)

        antipodal_directions = -directions
        antipodal_azimuths, antipodal_colatitudes, _ = tg.core.cart2sph(
            *antipodal_directions.T
        )

        values = np.zeros((n, 45), np.float32)
        antipodal_values = np.zeros((n, 45), np.float32)

        azimuths_buffer = self.runtime.buffer_from_array(
            azimuths.astype(np.float32), BufferAccess.READ_ONLY
        )
        colatitudes_buffer = self.runtime.buffer_from_array(
            colatitudes.astype(np.float32), BufferAccess.READ_ONLY
        )
        antipodal_azimuths_buffer = self.runtime.buffer_from_array(
            antipodal_azimuths.astype(np.float32), BufferAccess.READ_ONLY
        )
        antipodal_colatitudes_buffer = self.runtime.buffer_from_array(
            antipodal_colatitudes.astype(np.float32), BufferAccess.READ_ONLY
        )
        values_buffer = self.runtime.allocate_buffer(
            values.nbytes, BufferAccess.READ_WRITE
        )
        antipodal_values_buffer = self.runtime.allocate_buffer(
            antipodal_values.nbytes, BufferAccess.READ_WRITE
        )

        # Compile the OpenCL program that implements spherical harmonics.
        program = self.runtime.build_kernel_program("utils/spharm.cl")

        program.test_ishtmtx(
            self.runtime.queue,
            (1,),
            None,
            azimuths_buffer,
            colatitudes_buffer,
            np.uint32(len(azimuths)),
            values_buffer,
        )
        program.test_ishtmtx(
            self.runtime.queue,
            (1,),
            None,
            antipodal_azimuths_buffer,
            antipodal_colatitudes_buffer,
            np.uint32(len(antipodal_azimuths)),
            antipodal_values_buffer,
        )

        self.runtime.copy_from_buffer(values_buffer, values, is_blocking=True)
        self.runtime.copy_from_buffer(
            antipodal_values_buffer, antipodal_values, is_blocking=True
        )

        np.testing.assert_array_almost_equal(values, antipodal_values, 5)

    def test_ishtmtx(self):

        n = 100
        azimuths = np.linspace(0, 2 * np.pi - 2 * np.pi / n, n).astype(np.float32)
        colatitudes = np.linspace(0, np.pi - np.pi / n, n).astype(np.float32)
        values = tg.core.ishtmtx(azimuths, colatitudes, 45)[0].astype(np.float32)

        azimuths_buffer = self.runtime.buffer_from_array(
            azimuths, BufferAccess.READ_ONLY
        )
        colatitudes_buffer = self.runtime.buffer_from_array(
            colatitudes, BufferAccess.READ_ONLY
        )
        values_buffer = self.runtime.allocate_buffer(
            values.nbytes, BufferAccess.READ_WRITE
        )

        # Compile the OpenCL program that implements Boltzmann tractography.
        program = self.runtime.build_kernel_program("utils/spharm.cl")

        program.test_ishtmtx(
            self.runtime.queue,
            (1,),
            None,
            azimuths_buffer,
            colatitudes_buffer,
            np.uint32(len(azimuths)),
            values_buffer,
        )
        new_values = np.zeros(values.shape, np.float32)
        self.runtime.copy_from_buffer(values_buffer, new_values, is_blocking=True)
        np.testing.assert_array_almost_equal(new_values, values, 5)

    def test_ishtmtx_dt(self):

        n = 100
        azimuths = np.linspace(0, 2 * np.pi - 2 * np.pi / n, n).astype(np.float32)
        colatitudes = np.linspace(0, np.pi - np.pi / n, n).astype(np.float32)
        values = tg.core.ishtmtx(azimuths, colatitudes, 45)[1][0].astype(np.float32)

        azimuths_buffer = self.runtime.buffer_from_array(
            azimuths, BufferAccess.READ_ONLY
        )
        colatitudes_buffer = self.runtime.buffer_from_array(
            colatitudes, BufferAccess.READ_ONLY
        )
        values_buffer = self.runtime.allocate_buffer(
            values.nbytes, BufferAccess.READ_WRITE
        )

        # Compile the OpenCL program that implements Boltzmann tractography.
        program = self.runtime.build_kernel_program("utils/spharm.cl")

        program.test_ishtmtx_dt(
            self.runtime.queue,
            (1,),
            None,
            azimuths_buffer,
            colatitudes_buffer,
            np.uint32(len(azimuths)),
            values_buffer,
        )
        new_values = np.zeros(values.shape, np.float32)
        self.runtime.copy_from_buffer(values_buffer, new_values, is_blocking=True)
        np.testing.assert_array_almost_equal(new_values, values, 5)

    def test_ishtmtx_dp(self):

        n = 100
        azimuths = np.linspace(0, 2 * np.pi - 2 * np.pi / n, n).astype(np.float32)
        colatitudes = np.linspace(0, np.pi - np.pi / n, n).astype(np.float32)
        values = tg.core.ishtmtx(azimuths, colatitudes, 45)[1][1].astype(np.float32)

        azimuths_buffer = self.runtime.buffer_from_array(
            azimuths, BufferAccess.READ_ONLY
        )
        colatitudes_buffer = self.runtime.buffer_from_array(
            colatitudes, BufferAccess.READ_ONLY
        )
        values_buffer = self.runtime.allocate_buffer(
            values.nbytes, BufferAccess.READ_WRITE
        )

        # Compile the OpenCL program that implements Boltzmann tractography.
        program = self.runtime.build_kernel_program("utils/spharm.cl")

        program.test_ishtmtx_dp(
            self.runtime.queue,
            (1,),
            None,
            azimuths_buffer,
            colatitudes_buffer,
            np.uint32(len(azimuths)),
            values_buffer,
        )
        new_values = np.zeros(values.shape, np.float32)
        self.runtime.copy_from_buffer(values_buffer, new_values, is_blocking=True)
        np.testing.assert_array_almost_equal(new_values, values, 5)
