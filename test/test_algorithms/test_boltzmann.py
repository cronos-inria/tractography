import unittest
from pathlib import Path
from string import Template

import nibabel as nib
import numpy as np
import pyopencl as cl
import scipy.interpolate as si

import tractography as tg
import test.test_algorithms


_OPENCL_DIR = Path(__file__).parents[2] / "src"
_DATA_DIR = Path(__file__).parents[1] / "data"


class TestBoltzmann(unittest.TestCase):
    """Test the OpenCL implementation of Boltzmann tractography"""

    def setUp(self):
        test.test_algorithms.generate_cross_test_data()

    def test_cross_tractography(self):
        """Test tractography on the cross dataset"""

        fod = nib.load(_DATA_DIR / "cross-fod.nii.gz")
        wm_mask = nib.load(_DATA_DIR / "cross-wm.nii.gz")
        seed_mask = nib.load(_DATA_DIR / "cross-seed.nii.gz")
        seeds = tg.seeds.from_mask(seed_mask.get_fdata(), seed_mask.affine, 1000)

        fod_data = tg.core.apply_mask(
            fod.get_fdata(), fod.affine, wm_mask.get_fdata(), wm_mask.affine
        )

        config = tg.configuration.load()
        config.step_size = 0.1
        config.streamline.length.minimum = 1
        config.streamline.length.maximum = 20
        algorithm = tg.algorithms.Boltzmann(fod_data, fod.affine, len(seeds), config)
        streamlines = algorithm.run(seeds)

        tractogram = nib.streamlines.Tractogram(streamlines, affine_to_rasmm=np.eye(4))
        tck = nib.streamlines.TckFile(tractogram)
        tck.save(_DATA_DIR / "cross-streamlines-boltzmann.tck")

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

        # Compile the OpenCL program that implements Boltzmann tractography.
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

        # Compile the OpenCL program that implements Boltzmann tractography.
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

        # Compile the OpenCL program that implements Boltzmann tractography.
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

        # Compile the OpenCL program that implements Boltzmann tractography.
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
