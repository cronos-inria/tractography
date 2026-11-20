import numpy as np

import tractography as tg
from tractography.opencl import BufferAccess, OpenCLRuntime
from test.opencl_helpers import build_test_kernel_program


class TestCore:
    """Test the OpenCL implementation of core functions"""

    def test_cart2sph(self):
        """Test the cart2sph function"""

        azimuths = np.linspace(0, 2 * np.pi - 2 * np.pi / 1000, 1000).astype(np.float32)
        colatitudes = np.linspace(0, np.pi - np.pi / 1000, 1000).astype(np.float32)
        radii = np.linspace(0.25, 2.0, len(azimuths), dtype=np.float32)
        ex, ey, ez = tg.core.sph2cart(azimuths, colatitudes, radii)
        cart = np.c_[ex, ey, ez, np.zeros_like(ex)].astype(np.float32)
        sph = np.c_[azimuths, colatitudes].astype(np.float32)
        cart = np.concatenate(
            (cart, np.array([[0, 0, -2, 0], [0, 0, 0, 0]], dtype=np.float32))
        )
        sph = np.concatenate(
            (sph, np.array([[0, np.pi], [0, 0]], dtype=np.float32))
        )

        runtime = OpenCLRuntime()
        cart_buffer = runtime.allocate_buffer(cart.nbytes, BufferAccess.READ_ONLY)
        runtime.copy_to_buffer(cart_buffer, cart)
        sph_buffer = runtime.allocate_buffer(sph.nbytes)

        # Compile the OpenCL program that implements Boltzmann tractography.
        program = build_test_kernel_program(runtime, "test-utils.cl")

        program.test_cart2sph(
            runtime.queue,
            (1,),
            None,
            cart_buffer,
            np.int32(len(cart)),
            sph_buffer,
        )
        new_sph = np.zeros_like(sph)
        runtime.copy_from_buffer(sph_buffer, new_sph, is_blocking=True)
        np.testing.assert_array_almost_equal(new_sph, sph, 3)

    def test_randu(self):
        """Test the randu function"""

        runtime = OpenCLRuntime()
        values = np.zeros((100000,)).astype(np.float32)
        values_buffer = runtime.allocate_buffer(values.nbytes)

        # Compile the OpenCL program that implements Boltzmann tractography.
        program = build_test_kernel_program(runtime, "test-utils.cl")

        program.test_randu(
            runtime.queue,
            (1,),
            None,
            values_buffer,
            np.uint32(len(values)),
        )
        runtime.copy_from_buffer(values_buffer, values, is_blocking=True)

    def test_randn(self):
        """Test the randn function"""

        runtime = OpenCLRuntime()
        values = np.zeros((100000,)).astype(np.float32)
        values_buffer = runtime.allocate_buffer(values.nbytes)

        # Compile the OpenCL program that implements Boltzmann tractography.
        program = build_test_kernel_program(runtime, "test-utils.cl")

        program.test_randn(
            runtime.queue,
            (1,),
            None,
            values_buffer,
            np.uint32(len(values)),
        )
        runtime.copy_from_buffer(values_buffer, values, is_blocking=True)
