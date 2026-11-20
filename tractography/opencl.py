"""Runtime and utilities for executing packaged OpenCL kernels.

Importing this module does not initialize an OpenCL context. Construct an
:class:`OpenCLRuntime` explicitly when OpenCL execution is needed, then reuse
it for operations whose buffers and programs must share a context.
"""

from enum import Enum
from importlib.resources import as_file, files
from pathlib import Path
from string import Template
from typing import Mapping, Sequence

import numpy as np
import numpy.typing as npt
import pyopencl as cl


_KERNEL_DIR = files("tractography.resources") / "kernels"


class BufferAccess(Enum):
    """Kernel-side access mode for an OpenCL buffer."""

    READ_ONLY = cl.mem_flags.READ_ONLY
    WRITE_ONLY = cl.mem_flags.WRITE_ONLY
    READ_WRITE = cl.mem_flags.READ_WRITE


class OpenCLRuntime:
    """Own an OpenCL context and command queue for reusable kernel execution."""

    def __init__(
        self,
        device: cl.Device | None = None,
        *,
        context: cl.Context | None = None,
        queue: cl.CommandQueue | None = None,
    ) -> None:
        """Create a runtime, optionally using an existing device or context.

        If neither a context nor a device is supplied, PyOpenCL selects a
        device non-interactively. A supplied queue must belong to the selected
        context.
        """
        if context is None and queue is not None:
            context = queue.context
        if context is None:
            if device is None:
                context = cl.create_some_context(interactive=False)
            else:
                context = cl.Context(devices=[device])
        elif device is not None and all(
            device.int_ptr != context_device.int_ptr
            for context_device in context.devices
        ):
            raise ValueError("device must belong to the supplied context.")

        assert context is not None
        if queue is not None and queue.context.int_ptr != context.int_ptr:
            raise ValueError("queue must belong to the supplied context.")

        self.context = context
        self.device = (
            device if device is not None
            else queue.device if queue is not None
            else context.devices[0]
        )
        if queue is not None and queue.device.int_ptr != self.device.int_ptr:
            raise ValueError("queue must use the selected device.")
        self.queue = (
            queue
            if queue is not None
            else cl.CommandQueue(context, device=self.device)
        )

    def build_program(
        self,
        source: str,
        *,
        options: Sequence[str] = (),
    ) -> cl.Program:
        """Build an OpenCL program from source using this runtime's context."""
        return cl.Program(self.context, source).build(options=list(options))

    def build_kernel_program(
        self,
        names: str | Sequence[str],
        values: Mapping[str, object] | None = None,
        *,
        options: Sequence[str] = (),
    ) -> cl.Program:
        """Load, template, and build one or more packaged kernel files.

        Kernel sources are concatenated in the supplied order. Template
        substitution is strict, so a missing value raises ``KeyError`` rather
        than silently leaving a placeholder in the OpenCL source.
        """
        if isinstance(names, str):
            names = [names]

        with as_file(_KERNEL_DIR) as kernel_dir:
            source = "".join(
                (Path(kernel_dir) / name).read_text(encoding="utf-8")
                for name in names
            )
            source = Template(source).substitute(values or {})
            build_options = [*options, "-I", str(kernel_dir)]
            return self.build_program(source, options=build_options)

    def buffer_from_array(
        self,
        data: npt.ArrayLike,
        access: BufferAccess = BufferAccess.READ_ONLY,
    ) -> cl.Buffer:
        """Create a device buffer initialized from a contiguous host array."""
        array = np.ascontiguousarray(data)
        flags = access.value | cl.mem_flags.COPY_HOST_PTR
        return cl.Buffer(self.context, flags, hostbuf=array)

    def allocate_buffer(
        self,
        nbytes: int,
        access: BufferAccess = BufferAccess.WRITE_ONLY,
    ) -> cl.Buffer:
        """Allocate an uninitialized device buffer of ``nbytes`` bytes."""
        if nbytes <= 0:
            raise ValueError("nbytes must be positive.")
        return cl.Buffer(self.context, access.value, size=nbytes)

    def copy_to_buffer(
        self,
        buffer: cl.Buffer,
        data: npt.ArrayLike,
        *,
        is_blocking: bool = False,
    ) -> cl.Event:
        """Enqueue a host-to-device copy and return its completion event."""
        return cl.enqueue_copy(
            self.queue,
            buffer,
            np.ascontiguousarray(data),
            is_blocking=is_blocking,
        )

    def copy_from_buffer(
        self,
        buffer: cl.Buffer,
        data: np.ndarray,
        *,
        is_blocking: bool = False,
    ) -> cl.Event:
        """Enqueue a device-to-host copy and return its completion event."""
        return cl.enqueue_copy(
            self.queue,
            data,
            buffer,
            is_blocking=is_blocking,
        )

    def finish(self) -> None:
        """Wait for all commands previously submitted to this runtime."""
        self.queue.finish()
