from nibabel.nifti1 import Nifti1Image
import numpy as np
import pytest

import tractography as tg


def test_domain_contains():
    """Test the basic domain contains functionality."""

    # Some very basic image domain.
    image_data = np.zeros((3, 3, 3))
    image_data[1, 1, 1] = 1
    affine = np.eye(4)
    nii = Nifti1Image(image_data, affine=affine)

    domain = tg.domain.Image(nii)
    points = np.random.rand(5, 3).astype(np.float32) + 0.5
    assert np.all(tg.domain.contains(domain, points))
    assert not tg.domain.contains(domain, np.array([[0, 0, 0]], dtype=np.float32))

    # Test bad image domain (not 3D).
    pytest.raises(
        ValueError,
        tg.domain.Image,
        Nifti1Image(np.zeros((3, 3, 3, 3)), affine=np.eye(4))
    )


def test_image_domain_contains_world_coordinates():
    """Test with a translated image domain in world coordinates."""

     # Some very basic image domain with a translation in world coordinates.
    image_data = np.zeros((3, 3, 3))
    image_data[1, 1, 1] = 1
    affine = np.eye(4)
    affine[:3, 3] = (10, 20, 30)
    domain = tg.domain.Image(Nifti1Image(image_data, affine=affine))

    points = np.array([[11, 21, 31], [10, 20, 30]], dtype=np.float32)
    expected = np.array([True, False])

    assert np.array_equal(domain.contains(points), expected)
    assert np.array_equal(tg.domain.contains(domain, points), expected)


def test_domain_image_contains_reference_implementation():
    """Test that the OpenCL and reference implementation agree."""

    image_data = np.zeros((3, 3, 3))
    image_data[1, 1, 1] = 1
    affine = np.eye(4)
    nii = Nifti1Image(image_data, affine=affine)

    domain = tg.domain.Image(nii)
    points = np.random.rand(50, 3) * 5 - 1  # Some points outside the domain.

    assert np.array_equal(
        domain.contains(points),
        tg.domain.contains(domain, points)
    )