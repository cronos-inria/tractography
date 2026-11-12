from importlib.resources import files

from nibabel.nifti1 import Nifti1Image
import numpy as np
import pyopencl as cl
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
    """Voxel-cell faces are transformed and outward oriented."""
    image = np.ones((1, 1, 1), dtype=np.uint8)
    affine = np.eye(4)
    affine[:3, :3] = np.diag([2.0, 3.0, 4.0])
    affine[:3, 3] = [10.0, 20.0, 30.0]
    domain = tg.domain.Image(Nifti1Image(image, affine))

    assert domain.boundary.is_watertight
    assert domain.n_boundary_triangles > 0
    assert len(domain.boundary.vertices) == 8
    assert domain.n_boundary_triangles == 12
    assert domain.boundary_cumulative_areas[-1] == 1.0
    assert np.all(np.diff(domain.boundary_cumulative_areas) >= 0)
    np.testing.assert_allclose(
        domain.boundary.bounds,
        [[9.0, 18.5, 28.0], [11.0, 21.5, 32.0]],
    )
    face_centers = domain.boundary.triangles_center - np.array([10.0, 20.0, 30.0])
    outward = domain.boundary.face_normals
    assert np.all(np.einsum("ij,ij->i", face_centers, outward) > 0)


def test_image_domain_boundary_omits_faces_between_occupied_voxels():
    domain = tg.domain.Image(
        Nifti1Image(np.ones((2, 1, 1), dtype=np.uint8), np.eye(4))
    )

    assert domain.boundary.is_watertight
    assert len(domain.boundary.vertices) == 12
    assert domain.n_boundary_triangles == 20
    assert domain.boundary.area == pytest.approx(10.0)


def test_image_domain_contains_uses_half_open_voxel_cells():
    domain = tg.domain.Image(
        Nifti1Image(np.ones((1, 1, 1), dtype=np.uint8), np.eye(4))
    )
    points = np.array(
        [[-0.5, 0, 0], [0.49, 0.49, 0.49], [0.5, 0, 0]],
        dtype=np.float32,
    )

    expected = np.array([True, True, False])
    np.testing.assert_array_equal(domain.contains(points), expected)
    np.testing.assert_array_equal(tg.domain.contains(domain, points), expected)


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


def test_domain_boundary_bvh_finds_segment_triangle_hit():
    domain = tg.domain.Image(
        Nifti1Image(np.ones((1, 1, 1), dtype=np.uint8), np.eye(4))
    )
    context = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(context)
    source = domain.get_source() + """
    __kernel void probe(DOMAIN_ARGS, __global int* result)
    {
        DOMAIN_BUILD(domain)
        uint triangle;
        float3 barycentric;
        float hit_t;
        result[0] = bvh_segment_hit(
            domain.boundary_bvh,
            (float3)(0.0f, 0.0f, 0.0f),
            (float3)(-1.0f, 0.0f, 0.0f),
            &triangle,
            &barycentric,
            &hit_t
        );
    }
    """
    result = np.zeros(1, dtype=np.int32)
    buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, size=result.nbytes)
    kernel_dir = files("tractography.resources") / "kernels"
    program = cl.Program(context, source).build(f"-I {kernel_dir}")
    program.probe(queue, (1,), None, *domain.get_args(context), buffer)
    cl.enqueue_copy(queue, result, buffer)
    assert result[0] == 1


def test_domain_boundary_bvh_finds_sampled_inward_ray_exit():
    domain = tg.domain.Image(
        Nifti1Image(np.ones((1, 1, 1), dtype=np.uint8), np.eye(4))
    )
    context = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(context)
    source = domain.get_source() + """
    __kernel void probe(DOMAIN_ARGS, __global int* results)
    {
        const uint gid = get_global_id(0);
        DOMAIN_BUILD(domain)
        uint2 rng_state = rng_init(1234, gid);
        float3 point;
        float3 orientation;
        uint sampled_triangle;
        float3 sampled_barycentric;
        domain_sample_boundary(
            domain.boundary,
            &rng_state,
            &point,
            &orientation,
            &sampled_triangle,
            &sampled_barycentric
        );
        uint hit_triangle;
        float3 hit_barycentric;
        float hit_t;
        results[gid] = bvh_segment_hit(
            domain.boundary_bvh,
            point,
            point + 2.0f * orientation,
            &hit_triangle,
            &hit_barycentric,
            &hit_t
        );
    }
    """
    results = np.zeros(32, dtype=np.int32)
    buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, size=results.nbytes)
    kernel_dir = files("tractography.resources") / "kernels"
    program = cl.Program(context, source).build(f"-I {kernel_dir}")
    program.probe(queue, (len(results),), None, *domain.get_args(context), buffer)
    cl.enqueue_copy(queue, results, buffer)
    assert np.all(results == 1)