import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import pyopencl as cl
import scipy.interpolate as si

import tractography as tg
import test
import test.data.tensor


_OPENCL_DIR = Path(__file__).parents[2] / "src"
_TEST_RESULTS_DIR = (
    Path(__file__).parents[2] / "test-results" / "algorithms" / "transport"
)


class TestExactSolutions(unittest.TestCase):
    """Test the transport tractography kernel against numerically integrable FODs.

    The transport equation propagates a fibre orientation U along the gradient
    flow of log F on the unit sphere:

        dU/dt = gamma * (I - U*U^T) * grad_F(U) / F(U)

    where F : S^2 -> R+ is the FOD and gamma = inverse_curvature.  For simple
    analytic FODs the right-hand side can be evaluated exactly, allowing a
    high-precision reference trajectory to be computed in Python and compared
    against the OpenCL kernel output.
    """

    def test_u0u3(self):
        """Test against F(u) = 1 + eps * u_x * u_z (cross-term in the xz-plane).

        The FOD F(u) = 1 + eps * u_x * u_z is antipodally symmetric and
        representable in real spherical harmonics up to l=2.  Its Cartesian
        gradient is grad_F = eps * (u_z, 0, u_x), which gives the drift

            dU/dt = eps/F(U) * (I - U*U^T) * (U_z, 0, U_x)^T

        This ODE is integrated by `euler_intrinsic` below and the resulting
        positions are compared against the kernel output.
        """

        epsilon = 0.3
        n_coefficients = 45
        shape = (1, 1, 1)

        # Build a spatially constant FOD image with F(u) = 1 + eps * u_x * u_z.
        # The SH coefficients are found by least-squares fitting over a dense
        # Fibonacci grid on the sphere.
        bvectors = tg.core.fibonacci_sphere(5000)
        fod_values = [1 + epsilon * u[2] * u[0] for u in bvectors]

        azimuths, colatitudes, _ = tg.core.cart2sph(*bvectors.T)
        ylm, _ = tg.core.ishtmtx(azimuths, colatitudes, n_coefficients)
        fod_constant = np.dot(np.linalg.pinv(ylm), fod_values)

        fod = np.zeros(shape + (n_coefficients,))
        fod[..., :] = fod_constant

        # A single voxel 1000 mm wide keeps the streamline inside the image
        # throughout the 1 mm integration window.
        affine = np.diag((1000, 1000, 1000, 1))
        affine[:3, 3] = -0.5
        fod = nib.nifti1.Nifti1Image(fod, affine=affine)

        def euler_intrinsic(dt, t0, p0, epsilon, N):
            """Integrate dU/dt = eps/F(U) * (I - U*U^T) * (U_z, 0, U_x)^T via forward Euler.

            Args:
                dt: Integration step size (same units as config.save_at).
                t0: Initial colatitude theta_0 (radians).
                p0: Initial azimuth phi_0 (radians).
                epsilon: Anisotropy amplitude of the FOD.
                N: Total number of orientation vectors to return.

            Returns:
                Array of shape (N, 3) where entry k is the unit orientation
                vector after k update steps, i.e. [U_0, U_1, ..., U_{N-1}].
                U_0 is the initial orientation; U_k for k >= 1 is obtained by
                applying k Euler steps from U_0.
            """
            u1, u2, u3 = tg.core.sph2cart(p0, t0, 1.0)
            U = np.array([u1, u2, u3])
            sol = [U]
            I = np.eye(3)
            for _ in range(N - 1):
                F = 1 + epsilon * U[0] * U[2]
                # Cartesian gradient of F projected onto the tangent plane of
                # the sphere at U: (I - U*U^T) * grad_F / F.
                target_vec = np.array([U[2], 0.0, U[0]])  # grad_F / eps
                outer_U = np.outer(U, U)
                dU = np.dot((I - outer_U), target_vec)
                dU = dU * epsilon / F
                U = U + dt * dU
                U /= np.linalg.norm(U)  # re-normalise after Euler step
                sol.append(U)

            return np.array(sol)

        # Seed at an off-axis orientation so the drift is non-trivial.
        theta_0 = 0.12
        phi_0 = 0.5
        orientation = np.array(tg.core.sph2cart(phi_0, theta_0, 1.0))
        seeds = [tg.seeds.Seed(np.array([10., 10, 10]), orientation)]

        # Use step_size == save_at so every kernel step is saved, making
        # the reference integration straightforward.  inverse_curvature=1.0
        # matches the gamma=1 assumption in the analytic drift above.
        config = tg.configuration.load(tg.Algorithm.TRANSPORT)
        config.batch_size = 1
        config.step_size = 0.001
        config.save_at = 0.001
        config.inverse_curvature = 1.0
        config.streamline.length.minimum = 0.0
        config.streamline.length.maximum = 1.0
        streamline = np.array(tg.tractogram(fod, seeds, config).streamlines[0])

        # Build the reference trajectory by integrating with euler_intrinsic.
        # The kernel updates orientation *before* advancing position, so the
        # step from P_k to P_{k+1} uses the already-updated U_{k+1}:
        #
        #   P_{k+1} = P_k + dt * U_{k+1}
        #
        # euler_intrinsic returns [U_0, U_1, ..., U_{n_steps-1}], so index k
        # (one-based Euler step) corresponds to u_ref[k].
        u_ref = euler_intrinsic(config.save_at, theta_0, phi_0, epsilon, config.n_steps)
        seed_pos = np.array([10., 10., 10.])
        expected = np.zeros((config.n_steps, 3))
        expected[0] = seed_pos
        for k in range(1, config.n_steps):
            expected[k] = expected[k - 1] + config.save_at * u_ref[k]

        # decimal=3 (tolerance +-5e-4 mm) comfortably covers the ~1.2e-3 mm
        # accumulated float32 rounding error over ~1000 steps at position ~10 mm
        # (float32 ULP at 10 mm is ~9.5e-7 mm; errors grow as O(sqrt(N) * ULP)).
        np.testing.assert_array_almost_equal(streamline, expected, decimal=3)


class TestTransportHistogram(unittest.TestCase):

    def setUp(self):
        _TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def test_cross(self):

        fod, wm, _ = test.data.cross()

        config = tg.configuration.load(tg.Algorithm.TRANSPORT)
        nib.save(wm, _TEST_RESULTS_DIR / "histogram-cross-wm.nii.gz")
        nib.save(fod, _TEST_RESULTS_DIR / "histogram-cross-fod.nii.gz")

        histogram = tg.algorithms.transport.histogram(
            fod.get_fdata(),
            fod.affine,
            fod.get_fdata(),
            fod.affine,
            10000,
            config,
        )
        nib.save(
            nib.Nifti1Image(histogram, fod.affine),
            _TEST_RESULTS_DIR / "histogram-cross-histogram.nii.gz",
        )

        # The histogram and the FOD should be very similar. The value of 0.5 is
        # arbitrary.
        self.assertTrue(
            np.linalg.norm(histogram - fod.get_fdata())
            / wm.get_fdata().size < 0.5
        )


class TestTractogram(unittest.TestCase):
    """Test the OpenCL implementation of Transport tractography"""

    def test_uniform_isotropic(self):
        """Test transport tractography on a uniform isotropic fOD field"""

        # Prepare the data.
        fod = test.data.uniform_isotropic()
        affine = np.eye(4)
        nii = nib.nifti1.Nifti1Image(fod, affine)
        seeds = tg.seeds.from_fod(nii, 1000)

        # Generate the tractogram.
        config = tg.configuration.load(tg.Algorithm.TRANSPORT)
        tractogram, _ = tg.algorithms.transport.tractogram(nii, seeds, config)

        # In an isotropic field, transport tractography should produce only
        # straight lines.
        for streamline in tractogram.streamlines:
            d = np.diff(streamline, n=2, axis=0)
            np.testing.assert_array_less(d, 1e-3)

    def test_cross(self):
        """Test tractography on the cross dataset"""

        # Prepare the data.
        fod, wm, _ = test.data.cross()
        seeds = tg.seeds.from_mask(wm, 10000)

        # Generate the tractogram.
        config = tg.configuration.load(tg.Algorithm.TRANSPORT)
        tractogram, _ = tg.algorithms.transport.tractogram(tg.nifti.multiply(fod, wm), seeds, config)

        # Streamlines should cover the whole cross, but not go outside of it.
        points = np.vstack([s[:-1] for s in tractogram.streamlines])
        voxels = np.round(nib.affines.apply_affine(np.linalg.inv(fod.affine), points)).astype(int)
        mask = np.zeros(wm.shape, dtype=np.uint8)
        mask[*voxels.T] = 1
        np.testing.assert_array_equal(mask, wm.get_fdata().astype(np.uint8))
        

    def test_circle_dti(self):
        """Test tractography on the circle dataset, with DTI data"""

        # Prepare the data.
        fod, wm, _ = test.data.tensor.circle((20, 20, 1), radius=5, width=2)
        seeds = [tg.seeds.Seed([19.0, 22.5, 0.0], [-1.0, 0.0, 0.0])]

        # Generate the tractogram.
        config = tg.configuration.load(tg.Algorithm.TRANSPORT)
        config.inverse_curvature = 10.0
        tractogram, _ = tg.algorithms.transport.tractogram(fod, seeds, config)
        streamlines = tractogram.streamlines

    def test_circle(self):
        """Test tractography on the circle dataset"""

        # Prepare the data.
        shape = (10, 10, 1)
        radius = 2
        fod = test.data.circle(shape=shape, radius=radius)
        affine = np.eye(4)
        fod = nib.nifti1.Nifti1Image(fod, affine)
        wm = fod.get_fdata()[..., 0] > 0
        wm = nib.nifti1.Nifti1Image(wm.astype(np.uint8), affine)
        seeds = [
            tg.seeds.Seed(
                [(shape[0] - 1) / 2, (shape[1] - 1) / 2 + radius, 0.0], [1.0, 0.0, 0.0]
            )
        ]

        # Generate the tractogram. We set a few specific parameters due to the
        # small size of the circle.
        config = tg.configuration.load(tg.Algorithm.TRANSPORT)
        config.streamline.length.maximum = 100
        config.step_size = 1e-4
        config.inverse_curvature = 50.0
        tractogram, _ = tg.algorithms.transport.tractogram(fod, seeds, config)
        streamlines = tractogram.streamlines

        # The streamlines should run until the maximum lenght is reached.
        length = len(streamlines[0]) * config.save_at
        self.assertAlmostEqual(length, config.streamline.length.maximum)

    def test_mod(self):
        """Test the modulo operation"""

        angles = np.linspace(-4 * np.pi, 4 * np.pi, 1000).astype(np.float32)
        results = np.empty(angles.shape).astype(np.float32)

        # Create the global OpenCL context.
        _context = tg.algorithms.opencl._context
        _queue = tg.algorithms.opencl._queue

        flags = cl.mem_flags.READ_ONLY
        angles_buffer = cl.Buffer(_context, flags, size=angles.nbytes)
        cl.enqueue_copy(_queue, angles_buffer, np.ascontiguousarray(angles))

        flags = cl.mem_flags.WRITE_ONLY
        results_buffer = cl.Buffer(_context, flags, size=results.nbytes)

        # Compile the OpenCL program that implements Transport tractography.
        program = tg.algorithms.opencl.build_program(dict(), "test-boltzmann.cl")

        program.test_modulus(
            _queue, (1,), None, angles_buffer, np.int32(len(angles)), results_buffer
        )
        cl.enqueue_copy(_queue, results, results_buffer)

        np.testing.assert_array_almost_equal(results, np.mod(angles, 2 * np.pi))

    def test_wrap(self):
        """Test the wrap function"""

        azimuths = np.linspace(-4 * np.pi, 4 * np.pi, 1000).astype(np.float32)
        colatitudes = np.linspace(-4 * np.pi, 4 * np.pi, 1000).astype(np.float32)
        wa, wc = zip(*[tg.utils.wrap(a, c) for a, c in zip(azimuths, colatitudes)])

        # Create the global OpenCL context.
        _context = tg.algorithms.opencl._context
        _queue = tg.algorithms.opencl._queue

        flags = cl.mem_flags.READ_WRITE
        azimuths_buffer = cl.Buffer(_context, flags, size=azimuths.nbytes)
        cl.enqueue_copy(_queue, azimuths_buffer, azimuths)

        flags = cl.mem_flags.READ_WRITE
        colatitudes_buffer = cl.Buffer(_context, flags, size=colatitudes.nbytes)
        cl.enqueue_copy(_queue, colatitudes_buffer, colatitudes)

        # Compile the OpenCL program that implements Transport tractography.
        program = tg.algorithms.opencl.build_program(dict(), "test-boltzmann.cl")

        program.test_wrap(
            _queue,
            (1,),
            None,
            azimuths_buffer,
            colatitudes_buffer,
            np.int32(len(azimuths)),
        )
        cl.enqueue_copy(_queue, azimuths, azimuths_buffer)
        cl.enqueue_copy(_queue, colatitudes, colatitudes_buffer)
        np.testing.assert_array_almost_equal(wa, azimuths)
        np.testing.assert_array_almost_equal(wc, colatitudes)

    def test_sph2cart(self):
        """Test the sph2cart function"""

        azimuths = np.linspace(0, 2 * np.pi, 1000).astype(np.float32)
        colatitudes = np.linspace(0, np.pi, 1000).astype(np.float32)
        ex, ey, ez = zip(
            *[tg.core.sph2cart(a, c, 1) for a, c in zip(azimuths, colatitudes)]
        )

        # Create the global OpenCL context.
        _context = tg.algorithms.opencl._context
        _queue = tg.algorithms.opencl._queue

        flags = cl.mem_flags.READ_ONLY
        azimuths_buffer = cl.Buffer(_context, flags, size=azimuths.nbytes)
        cl.enqueue_copy(_queue, azimuths_buffer, azimuths)

        flags = cl.mem_flags.READ_ONLY
        colatitudes_buffer = cl.Buffer(_context, flags, size=colatitudes.nbytes)
        cl.enqueue_copy(_queue, colatitudes_buffer, colatitudes)

        flags = cl.mem_flags.WRITE_ONLY
        x = np.empty(azimuths.shape, dtype=np.float32)
        x_buffer = cl.Buffer(_context, flags, size=x.nbytes)
        cl.enqueue_copy(_queue, x_buffer, x)

        flags = cl.mem_flags.WRITE_ONLY
        y = np.empty(azimuths.shape, dtype=np.float32)
        y_buffer = cl.Buffer(_context, flags, size=y.nbytes)
        cl.enqueue_copy(_queue, y_buffer, y)

        flags = cl.mem_flags.WRITE_ONLY
        z = np.empty(azimuths.shape, dtype=np.float32)
        z_buffer = cl.Buffer(_context, flags, size=z.nbytes)
        cl.enqueue_copy(_queue, z_buffer, z)

        # Compile the OpenCL program that implements Transport tractography.
        program = tg.algorithms.opencl.build_program(dict(), "test-boltzmann.cl")

        program.test_sph2cart(
            _queue,
            (1,),
            None,
            azimuths_buffer,
            colatitudes_buffer,
            np.int32(len(azimuths)),
            x_buffer,
            y_buffer,
            z_buffer,
        )
        cl.enqueue_copy(_queue, x, x_buffer)
        cl.enqueue_copy(_queue, y, y_buffer)
        cl.enqueue_copy(_queue, z, z_buffer)
        np.testing.assert_array_almost_equal(x, ex)
        np.testing.assert_array_almost_equal(y, ey)
        np.testing.assert_array_almost_equal(z, ez)

    def test_sample_fod(self):
        """Test the sample_fod function"""

        # Create the global OpenCL context.
        _context = tg.algorithms.opencl._context
        _queue = tg.algorithms.opencl._queue

        flags = cl.mem_flags.READ_ONLY
        fod = np.random.rand(3, 4, 5, 6).astype(np.float32)
        fod_buffer = cl.Buffer(_context, flags, size=fod.nbytes)
        cl.enqueue_copy(_queue, fod_buffer, fod)

        flags = cl.mem_flags.READ_ONLY
        voxel = np.zeros(3).astype(np.float32)
        voxel_buffer = cl.Buffer(_context, flags, size=voxel.nbytes)
        cl.enqueue_copy(_queue, voxel_buffer, voxel)

        flags = cl.mem_flags.WRITE_ONLY
        coefficients = np.empty((6,), dtype=np.float32)
        coefficients_buffer = cl.Buffer(_context, flags, size=coefficients.nbytes)

        # Compile the OpenCL program that implements Transport tractography.
        program = tg.algorithms.opencl.build_program(dict(), "test-boltzmann.cl")

        x = np.arange(fod.shape[0])
        y = np.arange(fod.shape[1])
        z = np.arange(fod.shape[2])
        ifod = si.RegularGridInterpolator(
            (x, y, z), fod, method="nearest", bounds_error=False, fill_value=0
        )

        kernel = program.test_sample_fod
        for _ in range(1000):
            voxel[:] = [
                np.random.rand() * 2,
                np.random.rand() * 3,
                np.random.rand() * 4,
            ]
            cl.enqueue_copy(_queue, voxel_buffer, voxel)

            ec = ifod(voxel)[0]
            kernel(
                _queue, (1,), (1,), fod_buffer, voxel_buffer, coefficients_buffer
            )
            cl.enqueue_copy(_queue, coefficients, coefficients_buffer)
            np.testing.assert_array_almost_equal(coefficients, ec)

        voxel[:] = [-0.2, 0, 0]
        cl.enqueue_copy(_queue, voxel_buffer, voxel)
        ec = ifod(voxel)[0]
        kernel(
            _queue, (1,), None, fod_buffer, voxel_buffer, coefficients_buffer
        )
        cl.enqueue_copy(_queue, coefficients, coefficients_buffer)
        np.testing.assert_array_almost_equal(coefficients, ec)