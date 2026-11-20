import numpy as np
import pyopencl as cl
import pytest

from tractography.opencl import BufferAccess, OpenCLRuntime


def test_runtime_builds_packaged_kernels_and_strictly_requires_templates():
    runtime = OpenCLRuntime()

    with pytest.raises(KeyError):
        runtime.build_kernel_program("utils/seeds.cl")

    program = runtime.build_kernel_program(
        "utils/seeds.cl",
        {
            "model": "MODEL_DTI",
            "n_coefficients": 6,
            "n_seeds": 1,
            "nnz": 1,
        },
    )
    assert program.context.int_ptr == runtime.context.int_ptr


def test_runtime_rejects_queue_from_another_context():
    runtime = OpenCLRuntime()
    other_context = cl.Context(devices=[runtime.device])

    with pytest.raises(ValueError, match="supplied context"):
        OpenCLRuntime(context=other_context, queue=runtime.queue)


def test_runtime_buffer_transfers_round_trip_non_contiguous_data():
    runtime = OpenCLRuntime()
    data = np.arange(24, dtype=np.float32).reshape(4, 6)[:, ::2]
    buffer = runtime.buffer_from_array(data, BufferAccess.READ_ONLY)
    result = np.empty_like(data)

    runtime.copy_from_buffer(buffer, result, is_blocking=True)

    np.testing.assert_array_equal(result, data)


def test_runtime_rejects_non_positive_buffer_sizes():
    runtime = OpenCLRuntime()

    with pytest.raises(ValueError, match="positive"):
        runtime.allocate_buffer(0)
