import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

import tractography as tg
import test
import test.data.tensor


_OPENCL_DIR = Path(__file__).parents[2] / "src"
_TEST_RESULTS_DIR = (
    Path(__file__).parents[2] / "test-results" / "algorithms" / "diffusion"
)


class TestDiffusionHistogram(unittest.TestCase):

    def setUp(self):
        _TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def test_cross(self):

        fod, wm, _ = test.data.cross()
        fod = tg.nifti.multiply(fod, wm)
        config = tg.configuration.load(tg.Algorithm.DIFFUSION)

        nib.save(wm, _TEST_RESULTS_DIR / "histogram-cross-wm.nii.gz")
        nib.save(fod, _TEST_RESULTS_DIR / "histogram-cross-fod.nii.gz")

        histogram = tg.algorithms.diffusion.histogram(
            fod.get_fdata(), fod.affine, fod.get_fdata(), fod.affine, 10000, config
        )
        nib.save(
            nib.Nifti1Image(histogram, fod.affine),
            _TEST_RESULTS_DIR / "histogram-cross-histogram.nii.gz",
        )

        # The histogram and the FOD should be very similar.
        self.assertTrue(
            np.linalg.norm(histogram - fod.get_fdata())
            / wm.get_fdata().size < 0.005
        )

    def test_cross_dti(self):
        """Test histogram generation using DTI model on the cross dataset"""

        tensor, wm, _ = test.data.tensor.cross((20, 20, 1))
        config = tg.configuration.load(tg.Algorithm.DIFFUSION)

        nib.save(tensor, _TEST_RESULTS_DIR / "histogram-cross-dti-tensor.nii.gz")
        nib.save(wm, _TEST_RESULTS_DIR / "histogram-cross-dti-wm.nii.gz")

        histogram = tg.algorithms.diffusion.histogram(
            tensor.get_fdata(),
            tensor.affine,
            tensor.get_fdata(),
            tensor.affine,
            1000,
            config,
        )
        nib.save(
            nib.Nifti1Image(histogram, tensor.affine),
            _TEST_RESULTS_DIR / "histogram-cross-dti-histogram.nii.gz",
        )

        # Histogram should have same shape as input tensor and be finite
        self.assertEqual(histogram.shape, tensor.get_fdata().shape)
        self.assertTrue(np.isfinite(histogram).all())
        
        # In voxels with tensor data, histogram should be non-zero
        mask = tensor.get_fdata()[..., 0] > 0
        self.assertTrue((histogram[mask, 0] > 0).any())


class TestDiffusion(unittest.TestCase):
    """Test the OpenCL implementation of Diffusion tractography"""

    def setUp(self):
        _TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def test_uniform_isotropic(self):
        """Test diffusion tractography on a uniform isotropic fOD field"""

        # Prepare the data.
        fod = test.data.uniform_isotropic()
        affine = np.eye(4)
        seeds = tg.seeds.from_fod(fod, affine, 1000)
        nib.save(nib.Nifti1Image(fod, affine), _TEST_RESULTS_DIR / "uniform-fod.nii.gz")

        # Generate the tractogram.
        config = tg.configuration.load(tg.Algorithm.DIFFUSION)
        algorithm = tg.algorithms.Diffusion(fod, affine, len(seeds), config)
        streamlines = algorithm.run(seeds)

        # Save the streamlines for QA.
        tractogram = nib.streamlines.Tractogram(streamlines, affine_to_rasmm=np.eye(4))
        tck = nib.streamlines.TckFile(tractogram)
        tck.save(_TEST_RESULTS_DIR / "uniform-streamlines.tck")

    def test_cross(self):
        """Test tractography on the cross dataset"""

        # Prepare the data.
        fod, wm, _ = test.data.cross()
        nib.save(fod, _TEST_RESULTS_DIR / "cross-fod.nii.gz")
        nib.save(wm, _TEST_RESULTS_DIR / "cross-wm.nii.gz")
        seeds = tg.seeds.from_mask(wm.get_fdata(), wm.affine, 1000)

        # Generate the tractogram.
        config = tg.configuration.load(tg.Algorithm.DIFFUSION)
        algorithm = tg.algorithms.Diffusion(fod.get_fdata(), fod.affine, len(seeds), config)
        streamlines = algorithm.run(seeds)

        # Save the streamlines for QA.
        tractogram = nib.streamlines.Tractogram(streamlines, affine_to_rasmm=np.eye(4))
        tck = nib.streamlines.TckFile(tractogram)
        tck.save(_TEST_RESULTS_DIR / "cross-streamlines.tck")

    def test_circle_dti(self):
        """Test diffusion tractography on the circle dataset, with DTI data"""

        # Prepare the data.
        tensor, wm, _ = test.data.tensor.circle((20, 20, 1), radius=5, width=2)
        nib.save(tensor, _TEST_RESULTS_DIR / "circle-dti-tensor.nii.gz")
        nib.save(wm, _TEST_RESULTS_DIR / "circle-dti-wm.nii.gz")
        seeds = [tg.seeds.Seed([19.0, 22.5, 0.0], [-1.0, 0.0, 0.0])] * 10

        # Generate the tractogram.
        config = tg.configuration.load(tg.Algorithm.DIFFUSION)
        config.inverse_curvature = 10.0
        config.noise_variance = 0.05
        algorithm = tg.algorithms.Diffusion(
            tensor.get_fdata(), tensor.affine, len(seeds), config
        )
        streamlines = algorithm.run(seeds)

        # Save the streamlines for QA.
        tractogram = nib.streamlines.Tractogram(streamlines, affine_to_rasmm=np.eye(4))
        tck = nib.streamlines.TckFile(tractogram)
        tck.save(_TEST_RESULTS_DIR / "circle-dti-streamlines.tck")

        self.assertEqual(len(streamlines), len(seeds))
        self.assertTrue(any(len(streamline) > 1 for streamline in streamlines))
        for streamline in streamlines:
            self.assertTrue(np.isfinite(streamline).all())

    def test_circle(self):
        """Test tractography on the circle dataset"""

        # Prepare the data.
        shape = (10, 10, 1)
        radius = 2
        fod = test.data.circle(shape=shape, radius=radius)
        affine = np.eye(4)
        nib.save(nib.Nifti1Image(fod, affine), _TEST_RESULTS_DIR / "circle-fod.nii.gz")
        wm = fod[..., 0] > 0
        nib.save(
            nib.Nifti1Image(wm.astype(np.uint8), affine),
            _TEST_RESULTS_DIR / "circle-wm.nii.gz",
        )
        seeds = [
            tg.seeds.Seed(
                [(shape[0] - 1) / 2, (shape[1] - 1) / 2 + radius, 0.0], [1.0, 0.0, 0.0]
            )
            for _ in range(100)
        ]

        # Generate the tractogram. We set a few specific parameters due to the
        # small size of the circle.
        config = tg.configuration.load(tg.Algorithm.DIFFUSION)
        config.streamline.length.maximum = 100
        config.step_size = 1e-3
        config.inverse_curvature = 50.0
        config.noise_variance = 0.1
        algorithm = tg.algorithms.Diffusion(fod, affine, len(seeds), config)
        streamlines = algorithm.run(seeds)

        # Save the streamlines for QA.
        tractogram = nib.streamlines.Tractogram(streamlines, affine_to_rasmm=affine)
        tck = nib.streamlines.TckFile(tractogram)
        tck.save(_TEST_RESULTS_DIR / "circle-streamlines.tck")

        # The streamlines should run until the maximum lenght is reached.
        for streamline in streamlines:
            length = len(streamline) * config.save_at
            self.assertAlmostEqual(length, config.streamline.length.maximum)
