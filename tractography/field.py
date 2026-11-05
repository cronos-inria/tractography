from abc import ABC, abstractmethod
from string import Template
from typing import Any

from nibabel.affines import apply_affine
from nibabel.spatialimages import SpatialImage
import numpy as np
import pyopencl as cl
import pyopencl.cltypes as clt

from .domain import Points

type FieldValues = np.ndarray[tuple[int, int], np.dtype[clt.float]]

READ_ONLY = cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR
WRITE_ONLY = cl.mem_flags.WRITE_ONLY


class BaseField(ABC):
    """Abstract interface for a field of local model parameters."""

    def __init__(self, n_parameters: int):
        self.n_parameters = n_parameters

    @abstractmethod
    def get_source(self) -> str:
        """Return OpenCL source defining the field.

        The returned OpenCL source must implement:
            - the FIELD_ARGS macro for defining the kernel arguments.
            - the FIELD_BUILD macro for constructing the field struct.
            - the Field struct defining the field data.
            - the field_evaluate function for evaluating the field at a point.
        
        """
        pass

    @abstractmethod
    def get_args(self, context: cl.Context) -> list[Any]:
        """Return the OpenCL kernel arguments for the field.
        
        Args:
            context: OpenCL context for creating domain buffers.

        Returns:
            Arguments to pass to the field's OpenCL kernel.
        """
        pass

    @abstractmethod
    def evaluate(self, points: Points) -> Any:
        """Evaluate the field at the given points."""
        pass


class Image(BaseField):

    def __init__(self, image_data: SpatialImage):
        
        if image_data.ndim != 4:
            raise ValueError("Only 4D images are supported for image fields.")
        super().__init__(image_data.shape[3])

        self.values = np.ascontiguousarray(
            image_data.get_fdata(), 
            dtype=clt.float,
        )
        self.dims = np.ascontiguousarray(image_data.shape, dtype=clt.uint)
        affine = image_data.affine
        self.inverse_affine = np.ascontiguousarray(
            np.linalg.inv(affine if affine is not None else np.eye(4)), 
            dtype=clt.float,
        )

    def get_source(self) -> str:
        """Return OpenCL source for image field evaluation.
                
        See BaseField.get_source for the required macros and function
        signatures.
        
        """
        return """
        #define FIELD_ARGS __global const float *field_data, const uint4 field_dims, __global const float4* field_inverse_affine
        #define FIELD_BUILD(f) Field f = {field_data, field_dims, field_inverse_affine};

        typedef struct {
            __global const float *data;
            const uint4 dims;
            __global const float4* inverse_affine;
        } Field;
        
        void field_evaluate(Field f, float3 point, float* result) {
            
            // Transform the point from world coordinates to voxel coordinates.
            float4 point_h = (float4)(point.x, point.y, point.z, 1.0f);
            float4 voxel_h = (float4)(
                dot(f.inverse_affine[0], point_h),
                dot(f.inverse_affine[1], point_h),
                dot(f.inverse_affine[2], point_h),
                1.0f);
            int3 voxel = convert_int3(rint(voxel_h.xyz));
            
            // Sample the image at the voxel coordinates (nearest neighbor).
            if (voxel.x >= 0 && voxel.x < f.dims.x &&
                voxel.y >= 0 && voxel.y < f.dims.y &&
                voxel.z >= 0 && voxel.z < f.dims.z) {
                size_t index = ((voxel.x * f.dims.y + voxel.y) * f.dims.z + voxel.z) * f.dims.w;
                for (int i = 0; i < f.dims.w; ++i) {
                    result[i] = f.data[index + i];
                }
            } else {
                for (int i = 0; i < f.dims.w; ++i) {
                    result[i] = 0.0f; // Outside the image bounds
                }
            }
        }
        """

    def get_args(self, context: cl.Context) -> list[Any]:
        return [cl.Buffer(context, READ_ONLY, hostbuf=self.values),
                self.dims,
                cl.Buffer(context, READ_ONLY, hostbuf=self.inverse_affine)]

    def evaluate(self, points: Points) -> FieldValues:

        # Convert all points to homogeneous coordinates.
        voxel = np.round(apply_affine(self.inverse_affine, points)).astype(int)

        # Check if the voxel coordinates are within the image bounds.
        x, y, z = voxel[:, :3].T
        width, height, depth, n_parameters = self.values.shape
        in_bounds = (
            (0 <= x) & (x < width)
            & (0 <= y) & (y < height)
            & (0 <= z) & (z < depth)
        )

        # Initialize the result array with zeros.
        result = np.zeros((points.shape[0], n_parameters), dtype=clt.float)
        result[in_bounds, :] = self.values[*voxel[in_bounds].T]
        return result


def evaluate(field: BaseField, points: Points) -> FieldValues:

    # Stop early on empty points array.
    n_points = points.shape[0]
    if n_points == 0:
        return np.empty((0, field.n_parameters), dtype=clt.float)

    # Create the global OpenCL context.
    _context = cl.create_some_context(interactive=False)
    _queue = cl.CommandQueue(_context)

    # Create the OpenCL program from the domain source.
    source = field.get_source() + Template("""
    __kernel void evaluate(
        FIELD_ARGS,
        __global const float4* points,
        uint n_points,
        __global float* results)
    {
        uint gid = get_global_id(0);
        if (gid >= n_points) return;

        FIELD_BUILD(f)
        float tmp[$N_PARAMETERS];
        field_evaluate(f, points[gid].xyz, tmp);
        for (int i = 0; i < $N_PARAMETERS; ++i) {
            results[gid * $N_PARAMETERS + i] = tmp[i];
        }
    }
    """).safe_substitute({"N_PARAMETERS": field.n_parameters})

    # Points are stored in a (n_points, 4) array with the last component set
    # to 0 to match the OpenCL float4 format.
    points_data = np.zeros((n_points, 4), dtype=clt.float)
    points_data[:, :3] = points
    points_buffer = cl.Buffer(_context, READ_ONLY, hostbuf=points_data)

    # Prepare the results buffer.
    result = np.empty((n_points, field.n_parameters), dtype=clt.float)
    results_buffer = cl.Buffer(_context, WRITE_ONLY, size=result.nbytes)
    args = field.get_args(_context) + [
        points_buffer,
        clt.uint(n_points),
        results_buffer,
    ]

    # Run and transfer the results back to the host.
    program = cl.Program(_context, source).build()
    program.evaluate(_queue, (n_points,), None, *args)
    cl.enqueue_copy(_queue, result, results_buffer)

    return result