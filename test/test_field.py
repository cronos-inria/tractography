import numpy as np
from nibabel.nifti1 import Nifti1Image

import tractography as tg


def test_field_evaluate():
    """OpenCL and NumPy should agree on non-cubic image indexing."""
    data = np.arange(2 * 3 * 4 * 2, dtype=np.float32).reshape(2, 3, 4, 2)
    field = tg.field.Image(Nifti1Image(data, np.eye(4)))
    points = np.array([[0, 0, 0], [1, 2, 3], [1, 0, 2]], dtype=np.float32)

    indices = points.astype(int)
    expected = data[indices[:, 0], indices[:, 1], indices[:, 2]]
    np.testing.assert_array_equal(field.evaluate(points), expected)
    np.testing.assert_array_equal(tg.field.evaluate(field, points), expected)


def test_field_evaluate_rounds_half_voxels_to_even():
    """Both evaluation paths should use ties-to-even nearest-voxel rounding."""
    data = np.arange(4 * 4 * 4, dtype=np.float32).reshape(4, 4, 4, 1)
    field = tg.field.Image(Nifti1Image(data, np.eye(4)))
    points = np.array(
        [[0.5, 1.5, 2.5], [1.5, 0.5, 0.5], [-0.5, 2.5, 2.5]],
        dtype=np.float32,
    )
    expected = data[[0, 2, 0], [2, 0, 2], [2, 0, 2]]

    np.testing.assert_array_equal(field.evaluate(points), expected)
    np.testing.assert_array_equal(tg.field.evaluate(field, points), expected)


def test_field_evaluate_returns_zero_outside_image():
    """Coordinates rounding beyond any image edge should produce zeros."""
    data = np.ones((2, 3, 4, 2), dtype=np.float32)
    field = tg.field.Image(Nifti1Image(data, np.eye(4)))
    points = np.array([[-0.51, 0, 0], [0, 3, 0], [0, 0, 4]], dtype=np.float32)

    expected = np.zeros((3, 2), dtype=np.float32)
    np.testing.assert_array_equal(field.evaluate(points), expected)
    np.testing.assert_array_equal(tg.field.evaluate(field, points), expected)


def test_field_evaluate_empty_points():
    """Empty input should return a correctly shaped result without OpenCL setup."""
    field = tg.field.Image(Nifti1Image(np.zeros((2, 3, 4, 2)), np.eye(4)))

    result = tg.field.evaluate(field, np.empty((0, 3)))

    assert result.shape == (0, 2)
    assert result.dtype == np.float32
    