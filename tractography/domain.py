from abc import ABC, abstractmethod
from typing import Any

from nibabel.affines import apply_affine
from nibabel.spatialimages import SpatialImage
import numpy as np
import pyopencl as cl
import pyopencl.cltypes as clt

type Points = np.ndarray[tuple[int, int], Any]
type ContainsResult = np.ndarray[tuple[int], np.dtype[np.bool_]]

READ_ONLY = cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR
WRITE_ONLY = cl.mem_flags.WRITE_ONLY


class BaseDomain(ABC):
    """Abstract interface for domains used to constrain tractography."""

    @abstractmethod
    def get_source(self) -> str:
        """Return OpenCL source defining the domain.
        
        The returned OpenCL source must implement:
            - the DOMAIN_ARGS macro for defining the kernel arguments.
            - the DOMAIN_BUILD macro for constructing the domain struct.
            - the Domain struct defining the domain data.
            - the domain_contains function for checking if a point is within
                the domain.
        
        """
        pass

    @abstractmethod
    def get_args(self, context: cl.Context) -> list[Any]:
        """Upload domain data and return kernel arguments in the required order.

        Args:
            context: OpenCL context for creating domain buffers.

        Returns:
            Arguments to pass to the domain's OpenCL kernel.
        """
        pass

    @abstractmethod
    def contains(self, points: Points) -> ContainsResult:
        """Return whether each world-space point belongs to the domain.

        Args:
            points: Array of points with shape ``(n_points, 3)``.

        Returns:
            Boolean array with one result per point.
        """
        pass


class Image(BaseDomain):
    """Tracking domain defined by the nonzero voxels of a 3D image."""

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

    def get_source(self) -> str:
        """Return OpenCL source for image-domain containment checks.
        
        See BaseDomain.get_source for the required macros and function
        signatures.
        
        """
        return """
        #define DOMAIN_ARGS __global const int* domain_mask, const uint3 domain_dims, __global const float4* domain_inverse_affine
        #define DOMAIN_BUILD(d) Domain d = {domain_mask, domain_dims, domain_inverse_affine};

        typedef struct {
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
            int3 voxel = convert_int3(rint(voxel_h.xyz));

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
        return [values_buffer, self.dims, affine_buffer]

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
        voxel = np.round(apply_affine(self.inverse_affine, points)).astype(int)

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
    program = cl.Program(_context, source).build()
    program.contains(_queue, (n_points,), None, *args)
    cl.enqueue_copy(_queue, result, results_buffer)

    return result