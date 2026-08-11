import numpy as np
import trimesh
from nibabel.nifti1 import Nifti1Image

from tractography.domain import Domain


def test_domain_contains_and_boundary_contains():
    mesh = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
    domain = Domain(mesh)

    assert domain.contains(np.array([[0.0, 0.0, 0.0]]))[0]
    assert not domain.contains(np.array([[2.0, 0.0, 0.0]]))[0]
    assert domain.boundary_contains(np.array([[1.0, 0.0, 0.0]]))[0]
    assert not domain.boundary_contains(np.array([[0.2, 0.2, 0.2]]))[0]


def test_domain_to_mask_returns_a_binary_nifti_image():
    mesh = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
    domain = Domain(mesh)

    mask = domain.to_mask(shape=(5, 5, 5), affine=np.eye(4))

    assert isinstance(mask, Nifti1Image)
    assert mask.shape == (5, 5, 5)
    assert np.any(mask.get_fdata())


def test_domain_sample_returns_boundary_points():
    mesh = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
    domain = Domain(mesh)

    samples = domain.sample(10)

    assert samples.shape == (10, 3)


def test_domain_from_image_uses_affine():
    data = np.ones((1, 1, 1), dtype=np.uint8)
    affine = np.array(
        [[1.0, 0.0, 0.0, 10.0], [0.0, 1.0, 0.0, 20.0], [0.0, 0.0, 1.0, 30.0], [0.0, 0.0, 0.0, 1.0]]
    )
    image = Nifti1Image(data, affine)

    domain = Domain.from_image(image)

    assert domain.contains(np.array([[10.0, 20.0, 30.0]]))[0]
