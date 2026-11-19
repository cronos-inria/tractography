"""Opt-in end-to-end tests for examples included in the documentation."""

import subprocess
import sys
import unittest
from pathlib import Path

import nibabel as nib


_REPOSITORY_ROOT = Path(__file__).parents[2]
_EXAMPLE = _REPOSITORY_ROOT / "examples" / "iflow_tractogram.py"


class TestDocumentedExamples(unittest.TestCase):
    def test_iflow_tractogram_example(self):
        output_directory = _REPOSITORY_ROOT / "test-integration"
        output_directory.mkdir(parents=True, exist_ok=True)
        output_path = output_directory / "iflow-tractogram.tck"
        subprocess.run(
            [sys.executable, str(_EXAMPLE), "--output", str(output_path)],
            cwd=_REPOSITORY_ROOT,
            check=True,
            timeout=1800,
        )

        tractogram = nib.streamlines.load(output_path)
        self.assertGreater(len(tractogram.streamlines), 0)


if __name__ == "__main__":
    unittest.main()
