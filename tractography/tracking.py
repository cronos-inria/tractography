from importlib.resources import files
from string import Template

from nibabel.streamlines.tractogram import Tractogram
import numpy as np
import pyopencl as cl
import pyopencl.cltypes as clt
from scipy.sparse import coo_matrix, csr_matrix

from .algorithm import BaseAlgorithm
from .domain import BaseDomain
from .field import BaseField


_OPENCL_DIR = files("tractography.resources") / "kernels"
READ_ONLY = cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR
WRITE_ONLY = cl.mem_flags.WRITE_ONLY


def tractogram(
        algorithm: BaseAlgorithm,
        domain: BaseDomain,
        field: BaseField,
        seeds: np.ndarray,
        save_at: float = 1.0,
        maximum_streamline_length: float = 300.0,
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

    # Compute the maximum number of steps based on the maximum streamline
    # length and the save interval.
    n_steps = int(np.floor(maximum_streamline_length / save_at))

    # Create the global OpenCL context.
    _context = cl.create_some_context(interactive=False)
    _queue = cl.CommandQueue(_context)

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
    seeds_buffer = cl.Buffer(_context, READ_ONLY, hostbuf=seeds_data)

    # Prepare the streamlines buffer.
    streamlines = np.zeros((n_seeds, n_steps, 4), dtype=clt.float)
    streamlines_buffer = cl.Buffer(_context, WRITE_ONLY, size=streamlines.nbytes)
    lengths = np.zeros((n_seeds,), dtype=np.uint32)
    lengths_buffer = cl.Buffer(_context, WRITE_ONLY, size=lengths.nbytes)

    args = domain.get_args(_context) + field.get_args(_context) + algorithm.get_args()
    args += [
        clt.uint(algorithm.get_rng_seed()),
        seeds_buffer,
        clt.uint(n_seeds),
        streamlines_buffer,
        lengths_buffer,
        clt.float(save_at)
    ]

    # Run and transfer the results back to the host.
    program = cl.Program(_context, source).build(f"-I {_OPENCL_DIR}")
    program.tractogram(_queue, (n_seeds,), None, *args)
    cl.enqueue_copy(_queue, streamlines, streamlines_buffer)
    cl.enqueue_copy(_queue, lengths, lengths_buffer)

    return Tractogram([streamlines[i, :n, :3] for i, n in enumerate(lengths)], affine_to_rasmm=np.eye(4))


def connectome(
    algorithm: BaseAlgorithm,
    domain: BaseDomain,
    field: BaseField,
    n_seeds: int,
    maximum_streamline_length: float = 300.0,
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

    n_vertices = len(domain.boundary.vertices)
    if n_seeds == 0 or n_vertices == 0:
        return csr_matrix((n_vertices, n_vertices), dtype=np.float32)

    step_size = float(algorithm.step_size)
    if not np.isfinite(step_size) or step_size <= 0:
        raise ValueError("algorithm.step_size must be finite and positive.")
    n_steps = int(np.floor(maximum_streamline_length / step_size))
    if n_steps == 0:
        return csr_matrix((n_vertices, n_vertices), dtype=np.float32)

    context = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(context)
    source = domain.get_source() + field.get_source() + algorithm.get_source()
    source += Template(r"""
    __kernel void connectome(
        DOMAIN_ARGS,
        FIELD_ARGS,
        ALGORITHM_ARGS,
        uint rng_seed,
        uint n_seeds,
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
        }

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
    rows_buffer = cl.Buffer(context, WRITE_ONLY, size=contribution_rows.nbytes)
    columns_buffer = cl.Buffer(context, WRITE_ONLY, size=contribution_columns.nbytes)
    weights_buffer = cl.Buffer(context, WRITE_ONLY, size=contribution_weights.nbytes)
    args = domain.get_args(context) + field.get_args(context) + algorithm.get_args()
    args += [
        clt.uint(algorithm.get_rng_seed()),
        clt.uint(n_seeds),
        rows_buffer,
        columns_buffer,
        weights_buffer,
    ]

    program = cl.Program(context, source).build(f"-I {_OPENCL_DIR}")
    program.connectome(queue, (n_seeds,), None, *args)
    cl.enqueue_copy(queue, contribution_rows, rows_buffer)
    cl.enqueue_copy(queue, contribution_columns, columns_buffer)
    cl.enqueue_copy(queue, contribution_weights, weights_buffer)

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