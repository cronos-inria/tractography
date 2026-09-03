from pathlib import Path

import meshio
import nibabel as nib
import numpy as np
import pytest

from tractography import _mesh
from tractography import mesh as mesh_module
from tractography.mesh import Mesh


def _tetrahedron() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        dtype=np.float32,
    )
    tetrahedra = np.array([[0, 1, 2, 3]], dtype=np.uint32)
    return vertices, tetrahedra


def _image(data: np.ndarray, affine: np.ndarray | None = None) -> nib.Nifti1Image:
    return nib.Nifti1Image(data, np.eye(4) if affine is None else affine)


def test_mesh_normalizes_arrays_and_finds_nearest_vertex() -> None:
    vertices, tetrahedra = _tetrahedron()
    mesh = Mesh(vertices.astype(np.float64), tetrahedra.astype(np.int64))

    assert mesh.vertices.dtype == np.float32
    assert mesh.tetrahedra.dtype == np.uint32
    assert mesh.vertices.flags.c_contiguous
    assert mesh.tetrahedra.flags.c_contiguous
    np.testing.assert_array_equal(mesh.nearest_vertex([0.9, 0.1, 0]), [1, 0, 0])
    np.testing.assert_array_equal(
        mesh_module.nearest_vertex(mesh, [0.9, 0.1, 0]), [1, 0, 0]
    )


def test_mesh_accepts_well_shaped_empty_arrays() -> None:
    mesh = Mesh(np.empty((0, 3)), np.empty((0, 4), dtype=np.int64))

    assert mesh.vertices.shape == (0, 3)
    assert mesh.tetrahedra.shape == (0, 4)
    with pytest.raises(ValueError, match="non-empty"):
        mesh.nearest_vertex([0, 0, 0])


@pytest.mark.parametrize(
    ("vertices", "tetrahedra", "exception"),
    [
        (np.empty((3,)), np.empty((0, 4), dtype=int), ValueError),
        (np.empty((0, 3)), np.empty((4,), dtype=int), ValueError),
        (np.array([[np.nan, 0, 0]]), np.empty((0, 4), dtype=int), ValueError),
        (np.zeros((4, 3)), np.array([[0.0, 1, 2, 3]]), TypeError),
        (np.zeros((4, 3)), np.array([[-1, 1, 2, 3]]), ValueError),
        (np.zeros((4, 3)), np.array([[0, 1, 2, 4]]), ValueError),
    ],
)
def test_mesh_rejects_invalid_arrays(vertices, tetrahedra, exception) -> None:
    with pytest.raises(exception):
        Mesh(vertices, tetrahedra)


def test_from_image_collapses_all_nonzero_values(monkeypatch) -> None:
    observed_masks = []
    vertices, tetrahedra = _tetrahedron()

    def create(mask, *args):
        observed_masks.append(mask.copy())
        return vertices, tetrahedra

    monkeypatch.setattr(mesh_module._mesh, "create", create)
    first = np.zeros((3, 3, 3), dtype=np.float32)
    first[1, 1, 1] = -2.5
    first[1, 1, 2] = 8.0
    second = first != 0

    mesh_module.from_image(_image(first))
    mesh_module.from_image(_image(second.astype(np.uint8)))

    assert len(observed_masks) == 2
    np.testing.assert_array_equal(observed_masks[0], observed_masks[1])
    assert observed_masks[0].dtype == np.uint8
    assert observed_masks[0].flags.c_contiguous
    assert set(np.unique(observed_masks[0])) == {0, 1}


def test_from_image_maps_axis_physical_vertices_to_world(monkeypatch) -> None:
    affine = np.array(
        [
            [0, -3, 0, 10],
            [2, 0, 0, 20],
            [0, 0, 4, 30],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )
    native_vertices = np.array(
        [[0, 0, 0], [2, 0, 0], [0, 3, 0], [0, 0, 4]],
        dtype=np.float32,
    )
    _, tetrahedra = _tetrahedron()
    monkeypatch.setattr(
        mesh_module._mesh,
        "create",
        lambda *args: (native_vertices, tetrahedra),
    )

    mesh = mesh_module.from_image(_image(np.ones((2, 2, 2)), affine))

    voxel_vertices = native_vertices / np.array([2, 3, 4])
    expected = nib.affines.apply_affine(affine, voxel_vertices)
    np.testing.assert_allclose(mesh.vertices, expected)


@pytest.mark.parametrize(
    ("data", "kwargs", "message"),
    [
        (np.zeros((2, 2, 2)), {}, "nonzero"),
        (np.full((2, 2, 2), np.nan), {}, "finite"),
        (np.ones((2, 2)), {}, "three-dimensional"),
        (np.ones((2, 2, 2)), {"tetrahedra_size": 0}, "tetrahedra_size"),
        (np.ones((2, 2, 2)), {"distance": np.inf}, "distance"),
    ],
)
def test_from_image_rejects_invalid_inputs(data, kwargs, message) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        mesh_module.from_image(_image(data), **kwargs)


def test_from_image_rejects_singular_affine() -> None:
    affine = np.array(
        [[1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=float,
    )
    with pytest.raises(ValueError, match="invertible"):
        mesh_module.from_image(_image(np.ones((2, 2, 2)), affine))


def test_native_rejects_noncontiguous_or_wrong_dtype_masks() -> None:
    mask = np.ones((2, 2, 2), dtype=np.uint8)
    with pytest.raises(TypeError):
        _mesh.create(mask.astype(np.float32), 1, 1, 1, 1, 1)
    with pytest.raises(TypeError):
        _mesh.create(mask[:, :, ::-1], 1, 1, 1, 1, 1)
    with pytest.raises(ValueError, match="positive"):
        _mesh.create(mask, 0, 1, 1, 1, 1)


def test_native_outputs_and_repeated_calls() -> None:
    mask = np.zeros((3, 3, 3), dtype=np.uint8)
    mask[1, 1, 1] = 1

    outputs = [
        _mesh.create(mask, 1, 1, 1, 0.8, 0.2),
        _mesh.create(mask, 1, 1, 1, 0.8, 0.2),
    ]

    for vertices, tetrahedra in outputs:
        assert vertices.dtype == np.float32
        assert tetrahedra.dtype == np.uint32
        assert vertices.ndim == 2 and vertices.shape[1] == 3
        assert tetrahedra.ndim == 2 and tetrahedra.shape[1] == 4
        assert vertices.flags.c_contiguous
        assert tetrahedra.flags.c_contiguous
        assert tetrahedra.size
        assert np.max(tetrahedra) < len(vertices)
        assert len(np.unique(tetrahedra)) == len(vertices)


def test_meshio_round_trip(tmp_path: Path) -> None:
    vertices, tetrahedra = _tetrahedron()
    mesh = Mesh(vertices, tetrahedra)
    filename = tmp_path / "mesh.vtu"

    mesh_module.save(filename, mesh)
    loaded = mesh_module.load(filename)

    np.testing.assert_allclose(loaded.vertices, mesh.vertices)
    np.testing.assert_array_equal(loaded.tetrahedra, mesh.tetrahedra)


def test_load_combines_tetra_blocks_that_are_not_first(monkeypatch) -> None:
    vertices = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 1],
        ],
        dtype=float,
    )
    mesh_data = meshio.Mesh(
        vertices,
        [
            ("triangle", np.array([[0, 1, 2]])),
            ("tetra", np.array([[0, 1, 2, 3]])),
            ("tetra", np.array([[1, 2, 3, 4]])),
        ],
    )
    monkeypatch.setattr(mesh_module.meshio, "read", lambda filename: mesh_data)

    loaded = mesh_module.load("unused.vtu")

    np.testing.assert_array_equal(
        loaded.tetrahedra,
        [[0, 1, 2, 3], [1, 2, 3, 4]],
    )
