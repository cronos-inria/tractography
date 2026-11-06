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


def test_image_domain_builds_oriented_world_space_boundary():
    """The marching-cubes boundary is transformed and outward oriented."""
    image = np.ones((1, 1, 1), dtype=np.uint8)
    affine = np.eye(4)
    affine[:3, :3] = np.diag([2.0, 3.0, 4.0])
    affine[:3, 3] = [10.0, 20.0, 30.0]
    domain = tg.domain.Image(Nifti1Image(image, affine))

    assert domain.boundary.is_watertight
    assert domain.n_boundary_triangles > 0
    assert domain.boundary_cumulative_areas[-1] == 1.0
    assert np.all(np.diff(domain.boundary_cumulative_areas) >= 0)
    np.testing.assert_allclose(
        domain.boundary.bounds,
        [[9.0, 18.5, 28.0], [11.0, 21.5, 32.0]],
    )
    face_centers = domain.boundary.triangles_center - np.array([10.0, 20.0, 30.0])
    outward = domain.boundary.face_normals
    assert np.all(np.einsum("ij,ij->i", face_centers, outward) > 0)


def test_empty_image_domain_is_rejected():
    """An empty mask cannot provide a boundary for sampling."""
    with pytest.raises(ValueError, match="at least one nonzero voxel"):
        tg.domain.Image(Nifti1Image(np.zeros((2, 2, 2)), np.eye(4)))


def test_domain_seed_samples_interior_points_with_inward_orientations():
    """OpenCL boundary samples return points and inward unit directions."""
    domain = tg.domain.Image(
        Nifti1Image(np.ones((5, 5, 5), dtype=np.uint8), np.eye(4))
    )

    seeds = tg.domain.seed(domain, 64, rng_seed=123)

    assert len(seeds) == 64
    locations = np.asarray([sample.location for sample in seeds])
    orientations = np.asarray([sample.orientation for sample in seeds])
    assert np.all(domain.contains(locations))
    np.testing.assert_allclose(np.linalg.norm(orientations, axis=1), 1.0, atol=1e-6)
    inward = np.array([2.0, 2.0, 2.0]) - locations
    assert np.all(np.einsum("ij,ij->i", orientations, inward) > 0)


def test_domain_seed_validates_count_and_handles_zero():
    """Zero seeds return immediately and negative counts are rejected."""
    domain = tg.domain.Image(Nifti1Image(np.ones((1, 1, 1)), np.eye(4)))
    assert tg.domain.seed(domain, 0) == []
    with pytest.raises(ValueError, match="non-negative integer"):
        tg.domain.seed(domain, -1)