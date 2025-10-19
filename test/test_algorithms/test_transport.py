import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import pyopencl as cl
import scipy.interpolate as si

import tractography as tg
import test


_OPENCL_DIR = Path(__file__).parents[2] / "src"
_TEST_RESULTS_DIR = (
    Path(__file__).parents[2] / "test-results" / "algorithms" / "transport"
)


class TestTransport(unittest.TestCase):
    """Test the OpenCL implementation of Transport tractography"""

    def setUp(self):
        _TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def test_uniform_isotropic(self):
        """Test transport tractography on a uniform isotropic fOD field"""

        # Prepare the data.
        fod = test.data.uniform_isotropic()
        affine = np.eye(4)
        seeds = tg.seeds.from_fod(fod, affine, 1000)
        nib.save(nib.Nifti1Image(fod, affine), _TEST_RESULTS_DIR / "uniform-fod.nii.gz")

        # Generate the tractogram.
        config = tg.configuration.load()
        algorithm = tg.algorithms.Transport(fod, affine, len(seeds), config)
        streamlines = algorithm.run(seeds)

        # In an isotropic field, transport tractography should produce only
        # straight lines.
        for streamline in streamlines:
            d = np.diff(streamline, n=2, axis=0)
            np.testing.assert_array_less(d, 1e-3)

        # Save the streamlines for QA.
        tractogram = nib.streamlines.Tractogram(streamlines, affine_to_rasmm=np.eye(4))
        tck = nib.streamlines.TckFile(tractogram)
        tck.save(_TEST_RESULTS_DIR / "uniform-streamlines.tck")

    def test_cross(self):
        """Test tractography on the cross dataset"""

        # Prepare the data.
        fod = test.data.cross()
        affine = np.eye(4)
        nib.save(nib.Nifti1Image(fod, affine), _TEST_RESULTS_DIR / "cross-fod.nii.gz")
        wm = fod[..., 0] > 0
        nib.save(
            nib.Nifti1Image(wm.astype(np.uint8), affine),
            _TEST_RESULTS_DIR / "cross-wm.nii.gz",
        )
        seeds = tg.seeds.from_mask(wm, affine, 1000)

        # Generate the tractogram.
        config = tg.configuration.load()
        algorithm = tg.algorithms.Transport(fod, affine, len(seeds), config)
        streamlines = algorithm.run(seeds)

        # Save the streamlines for QA.
        tractogram = nib.streamlines.Tractogram(streamlines, affine_to_rasmm=np.eye(4))
        tck = nib.streamlines.TckFile(tractogram)
        tck.save(_TEST_RESULTS_DIR / "cross-streamlines.tck")

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

        for _ in range(1000):
            voxel[:] = [
                np.random.rand() * 2,
                np.random.rand() * 3,
                np.random.rand() * 4,
            ]
            cl.enqueue_copy(_queue, voxel_buffer, voxel)

            ec = ifod(voxel)[0]
            program.test_sample_fod(
                _queue, (1,), (1,), fod_buffer, voxel_buffer, coefficients_buffer
            )
            cl.enqueue_copy(_queue, coefficients, coefficients_buffer)
            np.testing.assert_array_almost_equal(coefficients, ec)

        voxel[:] = [-0.2, 0, 0]
        cl.enqueue_copy(_queue, voxel_buffer, voxel)
        ec = ifod(voxel)[0]
        program.test_sample_fod(
            _queue, (1,), None, fod_buffer, voxel_buffer, coefficients_buffer
        )
        cl.enqueue_copy(_queue, coefficients, coefficients_buffer)
        np.testing.assert_array_almost_equal(coefficients, ec)
