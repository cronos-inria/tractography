"""Helpers for compiling OpenCL kernels used only by tests."""

from importlib.resources import as_file, files
from pathlib import Path

from tractography.opencl import OpenCLRuntime


_TEST_KERNEL_DIR = Path(__file__).parent / "resources" / "kernels"
_RUNTIME_KERNEL_DIR = files("tractography.resources") / "kernels"


def build_test_kernel_program(runtime: OpenCLRuntime, name: str):
    """Build a test kernel with includes resolved from packaged kernels."""
    source = (_TEST_KERNEL_DIR / name).read_text(encoding="utf-8")
    with as_file(_RUNTIME_KERNEL_DIR) as kernel_directory:
        return runtime.build_program(source, options=("-I", str(kernel_directory)))
