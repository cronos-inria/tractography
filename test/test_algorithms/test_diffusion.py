import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

import tractography as tg
import test


_OPENCL_DIR = Path(__file__).parents[2] / "src"
_TEST_RESULTS_DIR = (
    Path(__file__).parents[2] / "test-results" / "algorithms" / "diffusion"
)


# class TestDiffusionHistogram(unittest.TestCase):
#
#     def setUp(self):
#         _TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
#
#     def test_cross(self):
#         fod = test.data.cross()
#         fod = tg.utils.normalize_odf(fod)
#         wm = fod[..., 0] > 0
#         affine = np.eye(4)
#         config = tg.configuration.load(tg.Algorithm.DIFFUSION)
#         nib.save(
#             nib.Nifti1Image(wm.astype(np.uint8), affine),
#             _TEST_RESULTS_DIR / "histogram-cross-wm.nii.gz",
#         )
#         nib.save(
#             nib.Nifti1Image(fod, affine),
#             _TEST_RESULTS_DIR / "histogram-cross-fod.nii.gz",
#         )
#
#         histogram = tg.algorithms.diffusion.histogram(
#             fod, affine, fod, affine, 10000, config
#         )
#         nib.save(
#             nib.Nifti1Image(histogram, affine),
#             _TEST_RESULTS_DIR / "histogram-cross-histogram.nii.gz",
#         )
#
#         # The histogram and the FOD should be very similar.
#         self.assertTrue(np.linalg.norm(histogram - fod) / wm.size < 0.005)


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
        config = tg.configuration.load(tg.Algorithm.DIFFUSION)
        algorithm = tg.algorithms.Diffusion(fod, affine, len(seeds), config)
        streamlines = algorithm.run(seeds)

        # Save the streamlines for QA.
        tractogram = nib.streamlines.Tractogram(streamlines, affine_to_rasmm=np.eye(4))
        tck = nib.streamlines.TckFile(tractogram)
        tck.save(_TEST_RESULTS_DIR / "cross-streamlines.tck")

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
