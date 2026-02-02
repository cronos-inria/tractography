import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

import tractography as tg
import tractography.cli
import test


class TestHistogram(unittest.TestCase):

    def test_simple(self):
        """Test the histogram subcommand of the CLI"""

        # Generate the data.
        fod, seed_mask, _ = test.data.cross()

        with tempfile.TemporaryDirectory() as d:

            # Save all the data to temporary files.
            fod_path = Path(d) / "fod.nii.gz"
            nib.save(fod, fod_path)
            seed_mask_path = Path(d) / "mask.nii.gz"
            nib.save(seed_mask, seed_mask_path)
            histogram_path = Path(d) / "histogram.nii.gz"

            # Run the CLI.
            args = tg.cli.parse_arguments([
                "histogram",
                str(fod_path),
                str(seed_mask_path),
                str(Path(d) / "histogram.nii.gz"),
                "--number-of-seeds", "1000"
            ])
            tg.cli.main(args)

            # We only verify that the histogram was created. Its quality
            # is tested elsewhere.
            self.assertTrue(histogram_path.is_file())
