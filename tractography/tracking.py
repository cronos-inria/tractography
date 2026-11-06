from importlib.resources import files
from string import Template
from unittest import result

from nibabel.streamlines.tractogram import Tractogram
import numpy as np
import pyopencl as cl
import pyopencl.cltypes as clt

from .algorithm import BaseAlgorithm
from .domain import BaseDomain
from .field import BaseField
from .local import BaseLocalModel


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