from importlib.resources import files
from string import Template

from nibabel.nifti1 import Nifti1Image
from nibabel.spatialimages import SpatialImage
from nibabel.streamlines.tractogram import Tractogram
import numpy as np
import pyopencl.cltypes as clt
from scipy.sparse import coo_matrix, csr_matrix

from .algorithm import BaseAlgorithm
from .domain import BaseDomain
from .field import BaseField
from .opencl import BufferAccess, OpenCLRuntime
from .utils import normalize_odf


_OPENCL_DIR = files("tractography.resources") / "kernels"


def tractogram(
        algorithm: BaseAlgorithm,
        domain: BaseDomain,
        field: BaseField,
        seeds: np.ndarray,
        save_at: float = 1.0,
        maximum_streamline_length: float = 300.0,
        runtime: OpenCLRuntime | None = None,
    ) -> Tractogram:
    """Track streamlines from position-and-orientation seeds.

    Each seed is advanced by the OpenCL implementation of ``algorithm`` while
    it remains inside ``domain`` and ``field`` can be evaluated. The returned
    streamline points are recorded at intervals of ``save_at``; the initial
    seed position is always included. The storage limit is determined by
    ``maximum_streamline_length``.

    Args:
        algorithm: Orientation-update algorithm, including the local model
            used to interpret the field parameters.
        domain: Region in which tracking is allowed to continue.
        field: Spatial field of local-model parameters. Its parameter count
            must match ``algorithm.local_model.n_parameters``.
        seeds: Array with shape ``(n_seeds, 6)``. Each row contains the seed
            position ``(x, y, z)`` followed by its initial direction
            ``(dx, dy, dz)``. Positions are interpreted in the coordinate
            system used by the domain and field.
        save_at: Distance interval between saved streamline points, in the
            coordinate units used by the inputs. Defaults to ``1.0``.
        maximum_streamline_length: Maximum length reserved for each
            streamline, in the same units. Defaults to ``300.0``.
        runtime: Optional runtime to reuse. One is created when omitted.

    Returns:
        A NiBabel ``Tractogram`` containing one streamline per seed, with
        points in the input coordinate system and an identity
        ``affine_to_rasmm``.

    Raises:
        ValueError: If the local model and field have different parameter
            counts.

    Note:
        This function executes an OpenCL kernel and requires a working OpenCL
        runtime. An empty seed array returns an empty tractogram without
        creating an OpenCL context.
    """

    # Verify the local model and the field are compatible.
    if algorithm.local_model.n_parameters != field.n_parameters:
        raise ValueError(
            "Incompatible local model and field, they do not have the "
            "same number of parameters: "
            f"{algorithm.local_model.n_parameters} != {field.n_parameters}"
        )

    # Stop early on empty seeds.
    n_seeds = seeds.shape[0]
    if n_seeds == 0:
        return Tractogram([], affine_to_rasmm=np.eye(4))
    if runtime is None:
        runtime = OpenCLRuntime()

    # Compute the maximum number of steps based on the maximum streamline
    # length and the save interval.
    n_steps = int(np.floor(maximum_streamline_length / save_at))

    # Create the OpenCL program from the domain source.
    source = domain.get_source() + field.get_source() + algorithm.get_source()
    source +=  Template("""
    __kernel void tractogram(
        DOMAIN_ARGS,
        FIELD_ARGS,
        ALGORITHM_ARGS,
        uint rng_seed,
        __global const float4* seeds,
        uint n_seeds,
        __global float4* streamlines,
        __global uint* lengths,
        float save_at)
    {
        uint gid = get_global_id(0);
        if (gid >= n_seeds) return;

        DOMAIN_BUILD(domain);
        FIELD_BUILD(field);
        ALGORITHM_BUILD(algorithm);

        // Initialize the first streamline point and orientation with the seed.
        float4 point = seeds[gid * 2];
        float4 orientation = seeds[gid * 2 + 1];
        uint2 rng_state = rng_init(rng_seed, gid);

        streamlines[gid * $N_STEPS] = point;
        size_t n = 1;
        float time = 0;
        float tmp[$N_PARAMETERS];
        while (n < $N_STEPS) {

            // Check if we are still in the domain.
            if (!domain_contains(domain, point.xyz)) {
                break;
            }

            // Evaluate the field at the current point. If it fails, stop the
            // streamline.
            if (!field_evaluate(field, point.xyz, tmp)) {
                break;
            }

            // Update the orientation using the current field parameters.
            orientation = algorithm_update_orientation(algorithm, tmp, orientation, &rng_state);

            // Move the point forward.
            point += algorithm.step_size * orientation;
            
            // Move time forward and record point if necessary.
            time += algorithm.step_size;
            if (time >= save_at) {
                time -= save_at;
                streamlines[gid * $N_STEPS + n] = point - time * orientation;
                n++;
            }
        }
        lengths[gid] = n;
    }
    """).safe_substitute({"N_STEPS": n_steps, "N_PARAMETERS": field.n_parameters})

    # Orientations are stored in a (n_points, 4) array with the last component set
    # to 0 to match the OpenCL float4 format.
    seeds_data = np.zeros((n_seeds, 8), dtype=clt.float)
    seeds_data[:, :3] = seeds[:, :3]
    seeds_data[:, 4:7] = seeds[:, 3:]
    seeds_buffer = runtime.buffer_from_array(seeds_data, BufferAccess.READ_ONLY)

    # Prepare the streamlines buffer.
    streamlines = np.zeros((n_seeds, n_steps, 4), dtype=clt.float)
    streamlines_buffer = runtime.allocate_buffer(streamlines.nbytes)
    lengths = np.zeros((n_seeds,), dtype=np.uint32)
    lengths_buffer = runtime.allocate_buffer(lengths.nbytes)

    args = domain.get_args(runtime) + field.get_args(runtime) + algorithm.get_args()
    args += [
        clt.uint(algorithm.get_rng_seed()),
        seeds_buffer,
        clt.uint(n_seeds),
        streamlines_buffer,
        lengths_buffer,
        clt.float(save_at)
    ]

    # Run and transfer the results back to the host.
    program = runtime.build_program(source, options=("-I", str(_OPENCL_DIR)))
    program.tractogram(runtime.queue, (n_seeds,), None, *args)
    runtime.copy_from_buffer(streamlines_buffer, streamlines, is_blocking=True)
    runtime.copy_from_buffer(lengths_buffer, lengths, is_blocking=True)

    return Tractogram([streamlines[i, :n, :3] for i, n in enumerate(lengths)], affine_to_rasmm=np.eye(4))


def tod(
    algorithm: BaseAlgorithm,
    domain: BaseDomain,
    field: BaseField,
    image: SpatialImage,
    n_seeds: int,
    maximum_streamline_length: float = 300.0,
    runtime: OpenCLRuntime | None = None,
) -> Nifti1Image:
    """Estimate a track orientation distribution (TOD) as SH coefficients.

    Seeds are sampled uniformly over the domain boundary and directed inward.
    Each track contributes the real, symmetric SH representation of its
    orientation at every integration step. The output grid and affine are
    taken from ``image``. Each nonempty voxel is normalized to unit integral.

    Args:
        algorithm: Orientation-update algorithm and local model.
        domain: Tracking domain and boundary used for seed generation.
        field: Spatial field of local-model parameters.
        image: Three-dimensional image describing the output grid and affine.
        n_seeds: Number of boundary-seeded tracks to generate.
        maximum_streamline_length: Maximum propagation distance, in the
            coordinate units used by the domain and field.
        runtime: Optional runtime to reuse. One is created when omitted.

    Returns:
        A NIfTI image with shape ``image.shape + (45,)`` containing normalized
        real, symmetric SH coefficients for the track orientation distribution.

    Raises:
        ValueError: If the model and field are incompatible, seed count or
            maximum length is invalid, or the output image is not 3D.

    Note:
        Requires a working OpenCL runtime unless the seed count or maximum
        streamline length is zero.
    """
    if algorithm.local_model.n_parameters != field.n_parameters:
        raise ValueError(
            "Incompatible local model and field, they do not have the "
            "same number of parameters: "
            f"{algorithm.local_model.n_parameters} != {field.n_parameters}"
        )
    if not isinstance(n_seeds, (int, np.integer)) or n_seeds < 0:
        raise ValueError("n_seeds must be a non-negative integer.")
    if not np.isfinite(maximum_streamline_length) or maximum_streamline_length < 0:
        raise ValueError("maximum_streamline_length must be finite and non-negative.")
    if len(image.shape) != 3:
        raise ValueError("The output image description must be 3D.")

    output_shape = tuple(int(size) for size in image.shape)
    affine = image.affine if image.affine is not None else np.eye(4)
    inverse_affine = np.linalg.inv(affine).astype(clt.float)
    raw_tod = np.zeros(output_shape + (45,), dtype=np.float32)

    step_size = float(algorithm.step_size)
    if not np.isfinite(step_size) or step_size <= 0:
        raise ValueError("algorithm.step_size must be finite and positive.")
    n_steps = int(np.floor(maximum_streamline_length / step_size))
    if n_seeds == 0 or n_steps == 0:
        header = image.header.copy()
        header.set_data_dtype(np.float32)
        return Nifti1Image(normalize_odf(raw_tod), affine, header=header)

    if runtime is None:
        runtime = OpenCLRuntime()
    source = domain.get_source() + field.get_source() + algorithm.get_source()
    source += '#include "utils/spharm.cl"\n'
    source += Template(r"""
    __kernel void tod(
        DOMAIN_ARGS,
        FIELD_ARGS,
        ALGORITHM_ARGS,
        uint rng_seed,
        uint n_seeds,
        const uint4 output_dims,
        __global const float4* output_inverse_affine,
        __global float* coefficients)
    {
        const uint gid = get_global_id(0);
        if (gid >= n_seeds) return;

        DOMAIN_BUILD(domain);
        FIELD_BUILD(field);
        ALGORITHM_BUILD(algorithm);

        uint2 rng_state = rng_init(rng_seed, gid);
        float3 point;
        float3 initial_orientation;
        uint start_triangle;
        float3 start_barycentric;
        domain_sample_boundary(
            domain.boundary,
            &rng_state,
            &point,
            &initial_orientation,
            &start_triangle,
            &start_barycentric
        );
        float4 orientation = (float4)(initial_orientation, 0.0f);
        float local_model[$N_PARAMETERS];
        float ylm[45];
        float ylm_dp[45];
        float ylm_dt[45];
        float voxel_coefficients[45] = {0.0f};
        int3 previous_index = (int3)(-1, -1, -1);

        for (uint step = 0; step < $N_STEPS; ++step) {
            if (!domain_contains(domain, point)) break;
            if (!field_evaluate(field, point, local_model)) break;

            // Deposit the current direction in the output voxel, if the
            // track point lies on the requested image grid.
            const float4 point_h = (float4)(point, 1.0f);
            const float3 voxel = (float3)(
                dot(output_inverse_affine[0], point_h),
                dot(output_inverse_affine[1], point_h),
                dot(output_inverse_affine[2], point_h));
            const int3 index = convert_int3(floor(voxel + 0.5f));
            if (index.x >= 0 && index.x < output_dims.x &&
                index.y >= 0 && index.y < output_dims.y &&
                index.z >= 0 && index.z < output_dims.z) {
                const float2 angles = cart2sph((float4)(orientation.xyz, 0.0f));
                evaluate_sh(angles.x, angles.y, ylm, ylm_dp, ylm_dt);
                if (index.x == previous_index.x &&
                    index.y == previous_index.y &&
                    index.z == previous_index.z) {
                    for (uint i = 0; i < 45; ++i) {
                        voxel_coefficients[i] += ylm[i];
                    }
                } else {
                    if (previous_index.x >= 0) {
                        const size_t previous_base =
                            (((size_t)previous_index.x * output_dims.y + previous_index.y)
                                * output_dims.z + previous_index.z) * 45;
                        for (uint i = 0; i < 45; ++i) {
                            atomic_add_global_float(
                                coefficients + previous_base + i,
                                voxel_coefficients[i]
                            );
                        }
                    }
                    const size_t base =
                        (((size_t)index.x * output_dims.y + index.y) * output_dims.z + index.z) * 45;
                    for (uint i = 0; i < 45; ++i) {
                        voxel_coefficients[i] = ylm[i];
                    }
                    previous_index = index;
                }
            } else if (previous_index.x >= 0) {
                const size_t previous_base =
                    (((size_t)previous_index.x * output_dims.y + previous_index.y)
                        * output_dims.z + previous_index.z) * 45;
                for (uint i = 0; i < 45; ++i) {
                    atomic_add_global_float(
                        coefficients + previous_base + i,
                        voxel_coefficients[i]
                    );
                }
                previous_index = (int3)(-1, -1, -1);
            }

            orientation = algorithm_update_orientation(
                algorithm, local_model, orientation, &rng_state);
            if (!all(isfinite(orientation)) || length(orientation) < 0.5f) break;
            point += algorithm.step_size * orientation.xyz;
        }
        if (previous_index.x >= 0) {
            const size_t base =
                (((size_t)previous_index.x * output_dims.y + previous_index.y)
                    * output_dims.z + previous_index.z) * 45;
            for (uint i = 0; i < 45; ++i) {
                atomic_add_global_float(coefficients + base + i, voxel_coefficients[i]);
            }
        }
    }
    """).safe_substitute({
        "N_PARAMETERS": field.n_parameters,
        "N_STEPS": n_steps,
    })

    output_dims = np.asarray((*output_shape, 0), dtype=np.uint32)
    inverse_affine_buffer = runtime.buffer_from_array(
        np.ascontiguousarray(inverse_affine), BufferAccess.READ_ONLY
    )
    coefficients_buffer = runtime.buffer_from_array(
        raw_tod, BufferAccess.READ_WRITE
    )
    args = domain.get_args(runtime) + field.get_args(runtime) + algorithm.get_args()
    args += [
        clt.uint(algorithm.get_rng_seed()),
        clt.uint(n_seeds),
        output_dims,
        inverse_affine_buffer,
        coefficients_buffer,
    ]

    program = runtime.build_program(source, options=("-I", str(_OPENCL_DIR)))
    program.tod(runtime.queue, (n_seeds,), None, *args)
    runtime.copy_from_buffer(coefficients_buffer, raw_tod, is_blocking=True)

    header = image.header.copy()
    header.set_data_dtype(np.float32)
    return Nifti1Image(normalize_odf(raw_tod), affine, header=header)


def connectome(
    algorithm: BaseAlgorithm,
    domain: BaseDomain,
    field: BaseField,
    n_seeds: int,
    maximum_streamline_length: float = 300.0,
    minimum_streamline_length: float = 0.0,
    runtime: OpenCLRuntime | None = None,
) -> csr_matrix:
    """Estimate an asymmetric, piecewise-linear boundary connectome.

    Each kernel work-item samples a seed on a boundary triangle and checks
    ``domain_contains`` at the start of each tracking iteration. When tracking
    stops for any reason, its final segment is tested against the boundary BVH.
    ``domain_contains`` is authoritative for domain membership, but a boundary
    hit is accepted independently of whether that final point is in-domain.
    The kernel deposits the outer product of the start and boundary-hit
    barycentric weights onto the triangle vertices. Tracks with no boundary
    hit are omitted. Rows and columns correspond to
    ``domain.boundary.vertices`` in their existing order.

    Args:
        algorithm: Orientation-update algorithm for the local field model.
        domain: Tracking domain and boundary surface used for seed generation.
        field: Spatial local-model field.
        n_seeds: Number of boundary-seeded tracks to generate.
        maximum_streamline_length: Maximum propagation distance, in the
            coordinate units used by the domain and field.
        minimum_streamline_length: Minimum distance actually propagated for a
            track to contribute. Defaults to zero.
        runtime: Optional runtime to reuse. One is created when omitted.

    Returns:
        A directed floating-point ``(n_vertices, n_vertices)`` SciPy CSR
        matrix. Each accepted track contributes total weight one before any
        duplicate entries are summed.

    Raises:
        ValueError: If seed count, maximum length, or algorithm/field
            compatibility is invalid.

    Note:
        The kernel writes at most nine weighted contributions per seed, using
        O(n_seeds) device memory; it does not allocate a dense
        vertex-by-vertex matrix.
    """
    if algorithm.local_model.n_parameters != field.n_parameters:
        raise ValueError(
            "Incompatible local model and field, they do not have the "
            "same number of parameters: "
            f"{algorithm.local_model.n_parameters} != {field.n_parameters}"
        )
    if not isinstance(n_seeds, (int, np.integer)) or n_seeds < 0:
        raise ValueError("n_seeds must be a non-negative integer.")
    if not np.isfinite(maximum_streamline_length) or maximum_streamline_length < 0:
        raise ValueError("maximum_streamline_length must be finite and non-negative.")
    if not np.isfinite(minimum_streamline_length) or minimum_streamline_length < 0:
        raise ValueError("minimum_streamline_length must be finite and non-negative.")
    if minimum_streamline_length > maximum_streamline_length:
        raise ValueError(
            "minimum_streamline_length must not exceed maximum_streamline_length."
        )

    n_vertices = len(domain.boundary.vertices)
    if n_seeds == 0 or n_vertices == 0:
        return csr_matrix((n_vertices, n_vertices), dtype=np.float32)

    step_size = float(algorithm.step_size)
    if not np.isfinite(step_size) or step_size <= 0:
        raise ValueError("algorithm.step_size must be finite and positive.")
    n_steps = int(np.floor(maximum_streamline_length / step_size))
    if n_steps == 0:
        return csr_matrix((n_vertices, n_vertices), dtype=np.float32)

    if runtime is None:
        runtime = OpenCLRuntime()
    source = domain.get_source() + field.get_source() + algorithm.get_source()
    source += Template(r"""
    __kernel void connectome(
        DOMAIN_ARGS,
        FIELD_ARGS,
        ALGORITHM_ARGS,
        uint rng_seed,
        uint n_seeds,
        float minimum_streamline_length,
        __global int* contribution_rows,
        __global int* contribution_columns,
        __global float* contribution_weights)
    {
        const uint gid = get_global_id(0);
        if (gid >= n_seeds) return;

        DOMAIN_BUILD(domain);
        FIELD_BUILD(field);
        ALGORITHM_BUILD(algorithm);

        uint2 rng_state = rng_init(rng_seed, gid);
        const uint output_base = gid * 9;
        for (uint output_index = 0; output_index < 9; ++output_index) {
            contribution_rows[output_base + output_index] = -1;
        }
        float3 point;
        float3 orientation;
        uint start_triangle;
        float3 start_barycentric;
        domain_sample_boundary(
            domain.boundary,
            &rng_state,
            &point,
            &orientation,
            &start_triangle,
            &start_barycentric
        );

        float local_model[$N_PARAMETERS];
        uint hit_triangle = 0;
        float3 hit_barycentric = (float3)(0.0f);
        float3 previous_point = point;
        uint tracked_steps = 0;
        for (uint step = 0; step < $N_STEPS; ++step) {
            // Check before evaluating or updating. If this is outside, the
            // previous/current pair is the segment that crossed the domain.
            if (!domain_contains(domain, point)) break;
            if (!field_evaluate(field, point, local_model)) break;

            orientation = algorithm_update_orientation(
                algorithm,
                local_model,
                (float4)(orientation, 0.0f),
                &rng_state
            ).xyz;
            if (!all(isfinite(orientation)) || length(orientation) < 0.5f) break;

            previous_point = point;
            point += algorithm.step_size * orientation;
            tracked_steps++;
        }

        if (tracked_steps * algorithm.step_size < minimum_streamline_length) return;

        // Check the final segment regardless of whether tracking stopped at
        // the domain edge, the field, an invalid orientation, or the step cap.
        float hit_t;
        const bool reached_boundary = bvh_segment_hit(
            domain.boundary_bvh,
            previous_point,
            point,
            &hit_triangle,
            &hit_barycentric,
            &hit_t
        );
        if (!reached_boundary) return;

        const uint4 start_indices = domain_boundary_indices[start_triangle];
        const uint4 end_indices = domain_boundary_indices[hit_triangle];
        const uint start_vertices[3] = {start_indices.x, start_indices.y, start_indices.z};
        const uint end_vertices[3] = {end_indices.x, end_indices.y, end_indices.z};
        const float start_weights[3] = {
            start_barycentric.x, start_barycentric.y, start_barycentric.z
        };
        const float end_weights[3] = {
            hit_barycentric.x, hit_barycentric.y, hit_barycentric.z
        };
        uint contribution = 0;
        for (uint i = 0; i < 3; ++i) {
            for (uint j = 0; j < 3; ++j) {
                const uint output_index = output_base + contribution++;
                contribution_rows[output_index] = (int)start_vertices[i];
                contribution_columns[output_index] = (int)end_vertices[j];
                contribution_weights[output_index] = start_weights[i] * end_weights[j];
            }
        }
    }
    """).safe_substitute({
        "N_PARAMETERS": field.n_parameters,
        "N_STEPS": n_steps,
    })

    n_contributions = n_seeds * 9
    contribution_rows = np.full(n_contributions, -1, dtype=np.int32)
    contribution_columns = np.full(n_contributions, -1, dtype=np.int32)
    contribution_weights = np.zeros(n_contributions, dtype=np.float32)
    rows_buffer = runtime.allocate_buffer(contribution_rows.nbytes)
    columns_buffer = runtime.allocate_buffer(contribution_columns.nbytes)
    weights_buffer = runtime.allocate_buffer(contribution_weights.nbytes)
    args = domain.get_args(runtime) + field.get_args(runtime) + algorithm.get_args()
    args += [
        clt.uint(algorithm.get_rng_seed()),
        clt.uint(n_seeds),
        clt.float(minimum_streamline_length),
        rows_buffer,
        columns_buffer,
        weights_buffer,
    ]

    program = runtime.build_program(source, options=("-I", str(_OPENCL_DIR)))
    program.connectome(runtime.queue, (n_seeds,), None, *args)
    runtime.copy_from_buffer(rows_buffer, contribution_rows, is_blocking=True)
    runtime.copy_from_buffer(columns_buffer, contribution_columns, is_blocking=True)
    runtime.copy_from_buffer(weights_buffer, contribution_weights, is_blocking=True)

    valid = contribution_rows >= 0
    if not np.any(valid):
        return csr_matrix((n_vertices, n_vertices), dtype=np.float32)

    matrix = coo_matrix(
        (
            contribution_weights[valid],
            (contribution_rows[valid], contribution_columns[valid]),
        ),
        shape=(n_vertices, n_vertices),
        dtype=np.float32,
    ).tocsr()
    matrix.sum_duplicates()
    return matrix