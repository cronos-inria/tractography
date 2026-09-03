import numpy as np
import pytest
import trimesh
from nibabel.nifti1 import Nifti1Image

import tractography as tg


def test_domain_boundary_init():
    """Test the simple initialization of a domain boundary"""

    # Normal creation.
    ico = trimesh.creation.icosahedron()
    domain_boundary = tg.domain.DomainBoundary(ico.vertices, ico.faces)
    assert isinstance(domain_boundary, tg.domain.DomainBoundary)

    # Must be watertight.
    vertices = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
    ])
    triangles = np.array([
        [0, 1, 2],
    ])
    with pytest.raises(ValueError):
        tg.domain.DomainBoundary(vertices, triangles)

    # Must be 3D.
    vertices_4d = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
    ])
    with pytest.raises(ValueError):
        tg.domain.DomainBoundary(vertices_4d, triangles)

    # Must be triangular.
    tetrahedra = np.array([
        [0, 1, 2, 3]
    ])
    with pytest.raises(ValueError):
        tg.domain.DomainBoundary(vertices, tetrahedra)


def test_domain_boundary_contains():
    """Test the __contains__ method."""

    ico = trimesh.creation.icosahedron()
    domain_boundary = tg.domain.DomainBoundary(ico.vertices, ico.faces)

    # Shorthand, can also use .contains.
    assert ico.vertices[0:1] in domain_boundary
    weights = [0.1, 0.2, 0.7]
    point = np.dot(ico.vertices[ico.faces[0]].T, weights)
    assert point[None, :] in domain_boundary
    assert not np.r_[0, 0, 2][None, :] in domain_boundary

    # All points must be on the boundary.
    assert not np.array([[0, 0, 2], [0, 0, 0]]) in domain_boundary


def test_domain_from_image():
    """Test the creation of a domain from an image."""

    # A simple box.
    image_data = np.zeros((10, 10, 10))
    image_data[1:10, 1:10, 1:10] = 1
    image = Nifti1Image(image_data, np.eye(4))
    domain = tg.domain.from_image(image)
    assert isinstance(domain, tg.domain.Domain)

    # The bounding box of the mesh should more or less match the initial box.
    high = np.max(domain.vertices, axis=0)
    assert np.all(np.isclose(high, 9.5, 2))
    low = np.min(domain.vertices, axis=0)
    assert np.all(np.isclose(low, -0.5, 2))
