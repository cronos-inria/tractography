from abc import ABC, abstractmethod
from importlib.resources import files
from typing import Any

from nibabel.affines import apply_affine
from nibabel.spatialimages import SpatialImage
import numpy as np
import pyopencl as cl
import pyopencl.cltypes as clt
import trimesh
from trimesh import Trimesh

from .seeds import Seed

type Points = np.ndarray[tuple[int, int], Any]
type ContainsResult = np.ndarray[tuple[int], np.dtype[np.bool_]]

READ_ONLY = cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR
WRITE_ONLY = cl.mem_flags.WRITE_ONLY
_OPENCL_DIR = files("tractography.resources") / "kernels"


class BaseDomain(ABC):
    """Abstract interface for domains used to constrain tractography.

    ``contains()`` is the authoritative definition of domain membership.
    ``boundary`` supplies the surface used for boundary sampling and endpoint
    intersections; subclasses need not assume it encloses every point in the
    domain.

    Subclasses provide their domain-specific OpenCL definitions through
    ``get_source()`` and should append ``super().get_source()`` to include the
    shared boundary sampler. Their ``get_args()`` implementations should also
    append ``super().get_args(context)`` to include the boundary buffers.
    """

    def __init__(self, boundary: Trimesh) -> None:
        """Initialize the base domain.

        Boundary triangles must be non-degenerate and wound so their normals
        point out of the domain.
        """
        super().__init__()
        self.boundary = boundary
        self._set_boundary_data(boundary)

    def _set_boundary_data(self, boundary: Trimesh) -> None:
        if len(boundary.faces) == 0:
            raise ValueError("The boundary mesh must contain at least one triangle.")

        vertices = np.asarray(boundary.vertices, dtype=np.float64)
        faces = np.asarray(boundary.faces, dtype=np.uint32)
        triangle_areas = np.asarray(boundary.area_faces, dtype=np.float64)
        valid = np.isfinite(triangle_areas) & (triangle_areas > 0)
        faces = faces[valid]
        triangle_areas = triangle_areas[valid]
        if len(faces) == 0:
            raise ValueError("The boundary mesh must contain a non-degenerate triangle.")

        vertices4 = np.ones((len(vertices), 4), dtype=clt.float)
        vertices4[:, :3] = vertices
        indices4 = np.zeros((len(faces), 4), dtype=clt.uint)
        indices4[:, :3] = faces
        total_area = float(np.sum(triangle_areas, dtype=np.float64))
        if not np.isfinite(total_area) or total_area <= 0:
            raise ValueError("The boundary mesh must have a finite, positive area.")
        cumulative = np.cumsum(triangle_areas / total_area).astype(clt.float)
        cumulative[-1] = 1.0

        self.boundary_vertices = np.ascontiguousarray(vertices4)
        self.boundary_indices = np.ascontiguousarray(indices4)
        self.boundary_cumulative_areas = np.ascontiguousarray(cumulative)
        self.n_boundary_triangles = len(faces)

        # Build a compact binary BVH over the same non-degenerate triangles
        # used by the boundary sampler. Leaves reference slots in
        # boundary_indices, not the original mesh face array.
        triangle_vertices = vertices[faces]
        triangle_min = triangle_vertices.min(axis=1)
        triangle_max = triangle_vertices.max(axis=1)
        centroids = (triangle_min + triangle_max) * 0.5
        node_min: list[np.ndarray | None] = []
        node_max: list[np.ndarray | None] = []
        node_data: list[tuple[int, int, int, int] | None] = []
        ordered_triangles: list[int] = []
        leaf_size = 4

        def build_bvh(triangle_ids: np.ndarray) -> int:
            node_id = len(node_data)
            node_min.append(None)
            node_max.append(None)
            node_data.append(None)
            bounds_min = triangle_min[triangle_ids].min(axis=0)
            bounds_max = triangle_max[triangle_ids].max(axis=0)

            if len(triangle_ids) <= leaf_size:
                start = len(ordered_triangles)
                ordered_triangles.extend(int(i) for i in triangle_ids)
                node_data[node_id] = (-1, -1, start, len(triangle_ids))
            else:
                axis = int(np.argmax(bounds_max - bounds_min))
                sorted_ids = triangle_ids[np.argsort(centroids[triangle_ids, axis])]
                middle = len(sorted_ids) // 2
                left = build_bvh(sorted_ids[:middle])
                right = build_bvh(sorted_ids[middle:])
                node_data[node_id] = (left, right, 0, 0)

            node_min[node_id] = bounds_min
            node_max[node_id] = bounds_max
            return node_id

        build_bvh(np.arange(len(faces), dtype=np.int32))
        self.boundary_bvh_bounds_min = np.zeros((len(node_min), 4), dtype=clt.float)
        self.boundary_bvh_bounds_max = np.zeros((len(node_max), 4), dtype=clt.float)
        self.boundary_bvh_bounds_min[:, :3] = np.asarray(node_min)
        self.boundary_bvh_bounds_max[:, :3] = np.asarray(node_max)
        self.boundary_bvh_nodes = np.ascontiguousarray(node_data, dtype=clt.int)
        self.boundary_bvh_triangles = np.ascontiguousarray(ordered_triangles, dtype=clt.uint)
        self.n_boundary_bvh_nodes = len(node_data)

    def get_source(self) -> str:
        """Return OpenCL definitions for boundary sampling and the BVH type."""
        return r"""
        #include "utils/core.cl"
        #include "utils/bvh.cl"

        typedef struct {
            __global const float4* vertices;
            __global const uint4* indices;
            __global const float* cumulative_areas;
            uint n_triangles;
        } Boundary;

        inline void domain_sample_boundary(
            Boundary boundary,
            uint2* rng_state,
            float3* point,
            float3* orientation,
            uint* sampled_triangle,
            float3* barycentric)
        {
            const float target = randu(rng_state);
            uint lower = 0;
            uint upper = boundary.n_triangles;
            while (lower < upper) {
                const uint middle = lower + (upper - lower) / 2;
                if (boundary.cumulative_areas[middle] <= target) {
                    lower = middle + 1;
                } else {
                    upper = middle;
                }
            }
            const uint triangle_id = min(lower, boundary.n_triangles - 1);
            *sampled_triangle = triangle_id;
            const uint4 indices = boundary.indices[triangle_id];
            const float3 a = boundary.vertices[indices.x].xyz;
            const float3 b = boundary.vertices[indices.y].xyz;
            const float3 c = boundary.vertices[indices.z].xyz;

            const float root_u = sqrt(randu(rng_state));
            const float v = randu(rng_state);
            *barycentric = (float3)(1.0f - root_u, root_u * (1.0f - v), root_u * v);
            const float3 sampled = (1.0f - root_u) * a
                + root_u * (1.0f - v) * b
                + root_u * v * c;
            const float3 outward = normalize(cross(b - a, c - a));
            *orientation = -outward;
            *point = sampled + 1e-4f * (*orientation);
        }

        """

    def get_args(self, context: cl.Context) -> list[Any]:
        """Upload boundary data and return its OpenCL arguments.

        Args:
            context: OpenCL context for creating domain buffers.

        Returns:
            Boundary vertices, triangle indices, normalized cumulative areas,
            triangle count, BVH bounds/nodes/triangle order, and BVH node count
            in ``DOMAIN_ARGS`` order.
        """
        return [
            cl.Buffer(context, READ_ONLY, hostbuf=self.boundary_vertices),
            cl.Buffer(context, READ_ONLY, hostbuf=self.boundary_indices),
            cl.Buffer(context, READ_ONLY, hostbuf=self.boundary_cumulative_areas),
            clt.uint(self.n_boundary_triangles),
            cl.Buffer(context, READ_ONLY, hostbuf=self.boundary_bvh_bounds_min),
            cl.Buffer(context, READ_ONLY, hostbuf=self.boundary_bvh_bounds_max),
            cl.Buffer(context, READ_ONLY, hostbuf=self.boundary_bvh_nodes),
            cl.Buffer(context, READ_ONLY, hostbuf=self.boundary_bvh_triangles),
            clt.uint(self.n_boundary_bvh_nodes),
        ]

    @abstractmethod
    def contains(self, points: Points) -> ContainsResult:
        """Authoritatively determine whether world-space points are in-domain.

        Args:
            points: Array of points with shape ``(n_points, 3)``.

        Returns:
            Boolean array with one result per point.

        Note:
            Boundary geometry is for sampling and intersections. It does not
            replace this membership test, since a domain's supplied boundary
            may be open or may represent only part of its extent.
        """
        pass


def _voxel_boundary_mesh(mask: np.ndarray, affine: np.ndarray) -> Trimesh:
    """Build the exact surface of the union of occupied voxel cells.

    Voxel centers have integer coordinates and each voxel occupies a unit
    half-open cell. A face is emitted only where an occupied voxel neighbors
    an empty voxel or the outside of the array.
    """
    face_definitions = (
        ((-1, 0, 0), ((-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1))),
        ((1, 0, 0), ((1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1))),
        ((0, -1, 0), ((-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1))),
        ((0, 1, 0), ((-1, 1, -1), (-1, 1, 1), (1, 1, 1), (1, 1, -1))),
        ((0, 0, -1), ((-1, -1, -1), (-1, 1, -1), (1, 1, -1), (1, -1, -1))),
        ((0, 0, 1), ((-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1))),
    )

    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    quads: list[np.ndarray] = []
    for direction, corner_offsets in face_definitions:
        neighbor_slices = tuple(
            slice(1 + offset, 1 + offset + size)
            for offset, size in zip(direction, mask.shape)
        )
        exposed_voxels = np.argwhere(mask & ~padded[neighbor_slices])
        if len(exposed_voxels) == 0:
            continue
        offsets = np.asarray(corner_offsets, dtype=np.int64)
        quads.append(2 * exposed_voxels[:, np.newaxis, :] + offsets)

    doubled_corners = np.concatenate(quads, axis=0).reshape(-1, 3)
    unique_corners, inverse_indices = np.unique(
        doubled_corners, axis=0, return_inverse=True
    )
    quad_faces = inverse_indices.reshape(-1, 4)
    triangles = np.concatenate(
        (quad_faces[:, (0, 1, 2)], quad_faces[:, (0, 2, 3)]), axis=0
    )
    voxel_vertices = unique_corners.astype(np.float64) * 0.5
    world_vertices = apply_affine(affine, voxel_vertices)
    return Trimesh(vertices=world_vertices, faces=triangles, process=False)


class Image(BaseDomain):
    """Tracking domain defined by the nonzero voxels of a 3D image.

    Each voxel occupies the half-open cell ``[i - 0.5, i + 0.5)`` on each
    index axis. The boundary mesh is the exposed-face surface of the union of
    those cells. ``contains()`` remains the authoritative membership test.
    """

    def __init__(self, image_data: SpatialImage):
        """Initialize the domain from a 3D spatial image.

        The image is converted to a binary mask. Its inverse affine is retained
        to map world-space points to voxel coordinates.

        Args:
            image_data: 3D image defining the domain.

        Raises:
            ValueError: If the image is not 3D.
        """

        if image_data.ndim != 3:
            raise ValueError("Only 3D images are supported for image domains.")

        self.values = np.ascontiguousarray(
            image_data.get_fdata() != 0, 
            dtype=clt.int
        )
        self.dims = np.ascontiguousarray((*image_data.shape, 0), dtype=clt.uint)
        affine = image_data.affine
        self.inverse_affine = np.ascontiguousarray(
            np.linalg.inv(affine if affine is not None else np.eye(4)), 
            dtype=clt.float,
        )
        if not np.any(self.values):
            raise ValueError("Image domain must contain at least one nonzero voxel.")
        boundary = _voxel_boundary_mesh(
            self.values.astype(bool), affine if affine is not None else np.eye(4)
        )
        boundary.fix_normals(multibody=True)
        super().__init__(boundary)

    def get_source(self) -> str:
        """Return OpenCL source for image containment and boundary sampling."""
        return super().get_source() + """
        #define DOMAIN_ARGS \\
            __global const float4* domain_boundary_vertices, \\
            __global const uint4* domain_boundary_indices, \\
            __global const float* domain_boundary_cumulative_areas, \\
            uint domain_n_boundary_triangles, \\
            __global const float4* domain_boundary_bvh_bounds_min, \\
            __global const float4* domain_boundary_bvh_bounds_max, \\
            __global const int4* domain_boundary_bvh_nodes, \\
            __global const uint* domain_boundary_bvh_triangles, \\
            uint domain_n_boundary_bvh_nodes, \\
            __global const int* domain_mask, \\
            const uint3 domain_dims, \\
            __global const float4* domain_inverse_affine

        #define DOMAIN_BUILD(d) \\
            Boundary boundary = { \\
                domain_boundary_vertices, \\
                domain_boundary_indices, \\
                domain_boundary_cumulative_areas, \\
                domain_n_boundary_triangles \\
            }; \\
            BVH boundary_bvh = { \\
                domain_boundary_vertices, \\
                domain_boundary_indices, \\
                domain_boundary_bvh_bounds_min, \\
                domain_boundary_bvh_bounds_max, \\
                domain_boundary_bvh_nodes, \\
                domain_boundary_bvh_triangles, \\
                domain_n_boundary_bvh_nodes \\
            }; \\
            Domain d = { \\
                boundary, boundary_bvh, domain_mask, domain_dims, domain_inverse_affine \\
            };

        typedef struct {
            Boundary boundary;
            BVH boundary_bvh;
            __global const int* mask;
            uint3 dims;
            __global const float4* domain_inverse_affine;
        } Domain;

        bool domain_contains(Domain d, float3 pos) {

            // Transform the point from world coordinates to voxel coordinates.
            float4 pos_h = (float4)(pos.x, pos.y, pos.z, 1.0f);
            float4 voxel_h = (float4)(
                dot(d.domain_inverse_affine[0], pos_h),
                dot(d.domain_inverse_affine[1], pos_h),
                dot(d.domain_inverse_affine[2], pos_h),
                1.0f);
            // Voxels occupy half-open cells [i - 0.5, i + 0.5), matching
            // the exposed-face boundary generated by Image.
            int3 voxel = convert_int3(floor(voxel_h.xyz + 0.5f));

            // Check if the voxel coordinates are within the image bounds.
            if (voxel.x < 0 || voxel.x >= d.dims.x ||
                voxel.y < 0 || voxel.y >= d.dims.y ||
                voxel.z < 0 || voxel.z >= d.dims.z) {
                return false;
            }

            // Any non-zero value in the mask indicates the voxel is within the
            // domain.
            size_t idx = voxel.x * (d.dims.y * d.dims.z) + voxel.y * d.dims.z + voxel.z;
            return d.mask[idx] != 0;
        }
        """

    def get_args(self, context: cl.Context) -> list[Any]:
        """Create OpenCL buffers and return arguments for the image domain.

        Args:
            context: OpenCL context for creating the buffers.

        Returns:
            The mask, dimensions, and inverse-affine kernel arguments.
        """
        values_buffer = cl.Buffer(context, READ_ONLY, hostbuf=self.values)
        affine_buffer = cl.Buffer(context, READ_ONLY, hostbuf=self.inverse_affine)
        return super().get_args(context) + [values_buffer, self.dims, affine_buffer]

    def contains(self, points: Points) -> ContainsResult:
        """Check whether world-space points map to nonzero image voxels.

        Points are mapped to the nearest voxel using the image's inverse affine.
        Points outside the image bounds are considered outside the domain.
        This method is an implementation reference, consider using
        tg.domain.contains for better performance with OpenCL.

        Args:
            points: Array of points with shape ``(n_points, 3)``.

        Returns:
            Boolean array with one result per point.
        """

        # Convert all points to homogeneous coordinates.
        voxel = np.floor(apply_affine(self.inverse_affine, points) + 0.5).astype(int)

        # Check if the voxel coordinates are within the image bounds.
        x, y, z = voxel[:, :3].T
        width, height, depth = self.values.shape
        in_bounds = (
            (0 <= x) & (x < width)
            & (0 <= y) & (y < height)
            & (0 <= z) & (z < depth)
        )

        # Query the image values for the in-bounds voxels.
        result = np.zeros(x.shape, dtype=np.bool_)
        result[in_bounds] = self.values[*voxel[in_bounds].T] != 0
        return result


def contains(domain: BaseDomain, points: Points) -> ContainsResult:
    """Check domain membership for points using an OpenCL kernel.

    Args:
        domain: Domain implementation that supplies OpenCL source and arguments.
        points: Array of world-space points with shape ``(n_points, 3)``.

    Returns:
        Boolean array indicating whether each point is inside the domain.
    """

    # Stop early on empty points array.
    n_points = points.shape[0]
    if n_points == 0:
        return np.array([], dtype=np.bool_)

    # Create the global OpenCL context.
    _context = cl.create_some_context(interactive=False)
    _queue = cl.CommandQueue(_context)

    # Create the OpenCL program from the domain source.
    source = domain.get_source() + """
    __kernel void contains(
        DOMAIN_ARGS,
        __global const float4* points,
        uint n_points,
        __global bool* results)
    {
        uint gid = get_global_id(0);
	    if (gid >= n_points) return;

        DOMAIN_BUILD(d)
        results[gid] = domain_contains(d, points[gid].xyz);
    }
    """
    
    # Points are stored in a (n_points, 4) array with the last component set
    # to 0 to match the OpenCL float4 format.
    points_data = np.zeros((n_points, 4), dtype=clt.float)
    points_data[:, :3] = points
    points_buffer = cl.Buffer(_context, READ_ONLY, hostbuf=points_data)

    # Prepare the results buffer.
    result = np.empty(n_points, dtype=np.bool_)
    results_buffer = cl.Buffer(_context, WRITE_ONLY, size=result.nbytes)
    args = domain.get_args(_context) + [
        points_buffer,
        clt.uint(n_points),
        results_buffer,
    ]

    # Run and transfer the results back to the host.
    program = cl.Program(_context, source).build(f"-I {_OPENCL_DIR}")
    program.contains(_queue, (n_points,), None, *args)
    cl.enqueue_copy(_queue, result, results_buffer)

    return result


def seed(
    domain: BaseDomain,
    n_seeds: int,
    rng_seed: int | None = None,
) -> list[Seed]:
    """Generate seeds by sampling the domain boundary with OpenCL.

    Each seed is sampled uniformly over the boundary surface area. Its
    orientation points inward, and its location is offset slightly from the
    boundary by the domain sampler.

    Args:
        domain: Domain providing boundary-sampling OpenCL code and arguments.
        n_seeds: Number of seeds to generate.
        rng_seed: Optional seed for reproducible sampling.

    Returns:
        A list of seeds with world-space locations and inward orientations.

    Raises:
        ValueError: If ``n_seeds`` is not a non-negative integer.
    """
    if not isinstance(n_seeds, (int, np.integer)) or n_seeds < 0:
        raise ValueError("n_seeds must be a non-negative integer.")
    if n_seeds == 0:
        return []

    if rng_seed is None:
        rng_seed = int(np.random.default_rng().integers(0, 2**32, dtype=np.uint32))
    rng_seed = int(rng_seed) % 2**32

    source = domain.get_source() + """
    __kernel void sample_domain_boundary(
        DOMAIN_ARGS,
        uint rng_seed,
        uint n_seeds,
        __global float4* locations,
        __global float4* orientations)
    {
        const uint gid = get_global_id(0);
        if (gid >= n_seeds) return;

        DOMAIN_BUILD(domain)
        uint2 rng_state = rng_init(rng_seed, gid);
        float3 point;
        float3 orientation;
            uint triangle_id;
        float3 barycentric;
        domain_sample_boundary(
            domain.boundary,
            &rng_state,
            &point,
            &orientation,
            &triangle_id,
            &barycentric
        );
        locations[gid] = (float4)(point, 1.0f);
        orientations[gid] = (float4)(orientation, 0.0f);
    }
    """

    context = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(context)
    locations = np.empty((n_seeds, 4), dtype=clt.float)
    orientations = np.empty((n_seeds, 4), dtype=clt.float)
    locations_buffer = cl.Buffer(context, WRITE_ONLY, size=locations.nbytes)
    orientations_buffer = cl.Buffer(context, WRITE_ONLY, size=orientations.nbytes)
    args = domain.get_args(context) + [
        clt.uint(rng_seed),
        clt.uint(n_seeds),
        locations_buffer,
        orientations_buffer,
    ]

    program = cl.Program(context, source).build(f"-I {_OPENCL_DIR}")
    program.sample_domain_boundary(queue, (n_seeds,), None, *args)
    cl.enqueue_copy(queue, locations, locations_buffer)
    cl.enqueue_copy(queue, orientations, orientations_buffer)

    return [
        Seed(location[:3], orientation[:3])
        for location, orientation in zip(locations, orientations)
    ]