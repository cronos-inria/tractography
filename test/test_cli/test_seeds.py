import subprocess

import nibabel as nib
import nimesh
import numpy as np
import pytest

import tractography as tg
import tractography.cli.seeds
import test


class TestSeeds:
    """Test the generation of seeds using the CLI"""

    @pytest.fixture(autouse=True)
    def setup_inputs(self, tmp_path):
        # Prepare the data.
        fod, _, _ = test.data.cross()
        fod_path = tmp_path / "cross-fod.nii.gz"
        nib.save(fod, fod_path)
        self.fod_path = fod_path
        self.n_seeds = 1000

    def test_from_fod(self):
        """Test generating seeds from FODs"""

        # Generate the seeds from the API.
        seeds_path = self.fod_path.parent / "cross-seeds-fod-api.tck"
        tg.cli.seeds.from_fod(self.fod_path, self.n_seeds, seeds_path)
        seeds = tg.seeds.load(seeds_path)
        assert len(seeds) == self.n_seeds

        # Generate the seeds from the CLI.
        seeds_path = self.fod_path.parent / "cross-seeds-fod-cli.tck"
        subprocess.run(
            ["tractography", "seeds", "from-fod", str(self.fod_path), str(self.n_seeds), str(seeds_path)],
            check=True,
        )
        seeds = tg.seeds.load(seeds_path)
        assert len(seeds) == self.n_seeds

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
        surface_path = self.fod_path.parent / "cross.gii"
        nimesh.io.save(surface_path, surface)

        # Generate the seeds from the API.
        seeds_path = self.fod_path.parent / "cross-seeds-surface-api.tck"
        tg.cli.seeds.from_surface(surface_path, self.n_seeds, seeds_path)
        seeds = tg.seeds.load(seeds_path)
        assert len(seeds) == self.n_seeds

        # Generate the seeds from the CLI.
        seeds_path = self.fod_path.parent / "cross-seeds-surface-cli.tck"
        subprocess.run(
            ["tractography", "seeds", "from-surface", str(surface_path), str(self.n_seeds), str(seeds_path)],
            check=True,
        )
        seeds = tg.seeds.load(seeds_path)
        assert len(seeds) == self.n_seeds
