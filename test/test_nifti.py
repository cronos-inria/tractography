import unittest

import nibabel as nib
import numpy as np

import tractography as tg


class TestMultiply(unittest.TestCase):

    def create_nifti(self, shape, fill_value=1, affine=None):
        """Helper to create synthetic NIfTI images."""
        data = np.full(shape, fill_value, dtype=float)
        if affine is None:
            affine = np.eye(4)
        return nib.Nifti1Image(data, affine)

    def test_basic_multiplication_aligned(self):
        """Test multiplication of two aligned images."""
        shape = (10, 10, 10)
        img1 = self.create_nifti(shape, fill_value=2)
        img2 = self.create_nifti(shape, fill_value=3)

        result = tg.nifti.multiply(img1, img2)

        # 2 * 3 = 6
        np.testing.assert_array_equal(result.get_fdata(), np.full(shape, 6.0))
        np.testing.assert_array_equal(result.affine, img1.affine)

    def test_resampling_with_shift(self):
        """
        Test that the function correctly pulls data from a shifted image.
        If 'right' is shifted by +1 on x-axis, 'left' at x=0 should see
        the value that was at x=1 in 'right' (conceptually).
        """
        shape = (5, 5, 5)

        # Left image is all 1s
        left = self.create_nifti(shape, fill_value=1)

        # Right image has specific data
        right_data = np.zeros(shape)
        right_data[2, 2, 2] = 100  # The "target" value

        # Shift right image's physical space by -1 in x, y, z
        # This means the voxel at (2,2,2) is physically at (1,1,1)
        right_affine = np.eye(4)
        right_affine[:3, 3] = -1
        right = nib.Nifti1Image(right_data, right_affine)

        # We inspect Left at (1,1,1).
        # Left(1,1,1) is at world (1,1,1).
        # Right's voxel (2,2,2) is at world (1,1,1).
        # Therefore, result at (1,1,1) should pick up the 100.
        result = tg.nifti.multiply(left, right, order=0)

        self.assertEqual(result.get_fdata()[1, 1, 1], 100)

    def test_interpolation_order(self):
        """
        Test the difference between nearest-neighbor (order=0) 
        and linear (order=1).
        """
        shape = (5, 5, 5)
        left = self.create_nifti(shape, fill_value=1)

        # Right image is shifted by 0.25 voxels.
        right_affine = np.eye(4)
        right_affine[0, 3] = 0.25
        right = self.create_nifti(shape, fill_value=10, affine=right_affine)
        right.dataobj[2, 2, 2] = 0

        # Order 0: Nearest neighbor will likely snap to the integer or 0
        # depending on rounding, but it stays discrete (10 or 0).
        res_0 = tg.nifti.multiply(left, right, order=0)

        # Order 1: Linear interpolation should handle the 0.5 shift
        # by averaging neighbors.
        # Note: Since the image is constant 10s, linear interpolation of 10 and
        # 10 is still 10, but at the edges it interpolates with 0.
        res_1 = tg.nifti.multiply(left, right, order=1)

        # We just check that they are not identical bit-wise.
        self.assertFalse(np.array_equal(res_0.get_fdata(), res_1.get_fdata()))

    def test_non_commutative_geometry(self):
        """
        Ensure result always takes Left's shape and affine.
        """
        shape_small = (5, 5, 5)
        shape_large = (10, 10, 10)

        img_small = self.create_nifti(shape_small)
        img_large = self.create_nifti(shape_large)

        # Case A: Small * Large -> Result should be Small
        res_small = tg.nifti.multiply(img_small, img_large)
        self.assertEqual(res_small.shape, shape_small)
        np.testing.assert_array_equal(res_small.affine, img_small.affine)

        # Case B: Large * Small -> Result should be Large
        res_large = tg.nifti.multiply(img_large, img_small)
        self.assertEqual(res_large.shape, shape_large)
        np.testing.assert_array_equal(res_large.affine, img_large.affine)


class TestThreshold(unittest.TestCase):

    def test_threshold_logic(self):
        """Test that values are correctly binarized based on the threshold."""

        # Create a 1D gradient wrapped in a 3D volume: [0.0, 0.5, 1.0, 1.5, 2.0]
        data = np.array([0.0, 0.5, 1.0, 1.5, 2.0]).reshape(5, 1, 1)
        affine = np.eye(4)
        img = nib.Nifti1Image(data, affine)

        # Threshold at 1.0.
        # Expected: 0.0, 0.5 -> 0 | 1.0, 1.5, 2.0 -> 1
        result = tg.nifti.threshold(img, value=1.0)
        result_data = result.get_fdata()

        expected = np.array([0, 0, 1, 1, 1]).reshape(5, 1, 1)
        np.testing.assert_array_equal(result_data, expected)

    def test_metadata_and_dtype(self):
        """Ensure affine is preserved and output is strictly uint8."""

        # Use a non-identity affine to ensure it's copied correctly.
        shape = (3, 3, 3)
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        img = nib.Nifti1Image(np.random.rand(*shape), affine)

        result = tg.nifti.threshold(img, value=0.5)

        # Check affine.
        np.testing.assert_array_equal(result.affine, affine)

        # Check header data type.
        self.assertEqual(result.header.get_data_dtype(), np.uint8)

        # Check array data type.
        self.assertEqual(result.dataobj.dtype, np.uint8)

    def test_all_or_nothing(self):
        """Test cases where all voxels are above or below threshold."""

        data = np.ones((5, 5, 5)) * 10.0
        img = nib.Nifti1Image(data, np.eye(4))

        # Case 1: Threshold higher than all data.
        res_zeros = tg.nifti.threshold(img, value=20.0)
        self.assertEqual(res_zeros.get_fdata().sum(), 0)

        # Case 2: Threshold lower than all data.
        res_ones = tg.nifti.threshold(img, value=5.0)
        np.testing.assert_array_equal(res_ones.get_fdata(), np.ones((5, 5, 5)))
