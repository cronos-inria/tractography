import subprocess
import unittest
from pathlib import Path

import nibabel as nib
import nimesh
import numpy as np

import tractography as tg
import tractography.cli.seeds
import test


_TEST_RESULTS_DIR = Path(__file__).parents[2] / "test-results" / "cli" / "seeds"


class TestSeeds(unittest.TestCase):
    """Test the generation of seeds using the CLI"""

    def setUp(self):
        _TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        # Prepare the data.
        fod = test.data.cross()
        affine = np.eye(4)
        fod_path = _TEST_RESULTS_DIR / "cross-fod.nii.gz"
        nib.save(nib.Nifti1Image(fod, affine), fod_path)
        self.fod_path = fod_path
        self.n_seeds = 1000

    def test_from_fod(self):
        """Test generating seeds from FODs"""

        # Generate the seeds from the API.
        seeds_path = _TEST_RESULTS_DIR / "cross-seeds-fod-api.tck"
        tg.cli.seeds.from_fod(self.fod_path, self.n_seeds, seeds_path)
        seeds = tg.seeds.load(seeds_path)
        self.assertEqual(len(seeds), self.n_seeds)

        # Generate the seeds from the CLI.
        seeds_path = _TEST_RESULTS_DIR / "cross-seeds-fod-cli.tck"
        subprocess.run(["tractography", "seeds", "from-fod", self.fod_path, str(self.n_seeds), seeds_path])
        seeds = tg.seeds.load(seeds_path)
        self.assertEqual(len(seeds), self.n_seeds)

    def test_from_surface(self):
        """Test generating seeds from FODs"""

        # Generate a simple surface from the cross.
        vertices = np.array([
            [-0.5, 5.0, -0.5],
            [9.5, 5.0, -0.5],
            [9.5, 5.0, 0.5],
        ])
        triangles = np.array([
            [0, 1, 2],
        ])
        surface = nimesh.Mesh(vertices, triangles)
        surface_path = _TEST_RESULTS_DIR / "cross.gii"
        nimesh.io.save(surface_path, surface)

        # Generate the seeds from the API.
        seeds_path = _TEST_RESULTS_DIR / "cross-seeds-surface-api.tck"
        tg.cli.seeds.from_surface(surface_path, self.n_seeds, seeds_path)
        seeds = tg.seeds.load(seeds_path)
        self.assertEqual(len(seeds), self.n_seeds)

        # Generate the seeds from the CLI.
        seeds_path = _TEST_RESULTS_DIR / "cross-seeds-surface-cli.tck"
        subprocess.run(["tractography", "seeds", "from-surface", surface_path, str(self.n_seeds), seeds_path])
        seeds = tg.seeds.load(seeds_path)
        self.assertEqual(len(seeds), self.n_seeds)
