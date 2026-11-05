from abc import ABC, abstractmethod
from string import Template
from typing import Any
from importlib.resources import files

import numpy as np
import pyopencl as cl
import pyopencl.cltypes as clt

from .local import BaseLocalModel, Orientations
from .core import cart2sph

_OPENCL_DIR = files("tractography.resources") / "kernels"
READ_ONLY = cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR
WRITE_ONLY = cl.mem_flags.WRITE_ONLY


class BaseAlgorithm(ABC):
    """Abstract interface for algorithms that update orientations."""

    def __init__(self, local_model: BaseLocalModel):
        self.local_model = local_model

    @abstractmethod
    def get_source(self) -> str:
        """Return the OpenCL source implementing the algorithm."""

    @abstractmethod
    def get_args(self) -> list[Any]:
        """Return the scalar OpenCL arguments for this algorithm."""

    @abstractmethod
    def update_orientation(
        self, local_models: np.ndarray, orientations: Orientations
    ) -> Orientations:
        """Update orientations using the algorithm's NumPy implementation."""


class Transport(BaseAlgorithm):
    """Configure deterministic orientation updates from a local model.

    Args:
        local_model: Model that evaluates a value and its orientation
            derivatives.
        step_size: Step duration used to scale the drift.
        gamma: Drift scaling factor.
    """

    def __init__(self, local_model: BaseLocalModel, step_size: float = 0.01, gamma: float = 1.0):
        super().__init__(local_model)
        self.step_size = step_size
        self.gamma = gamma

    def get_source(self) -> str:
        """Return the OpenCL source for the transport orientation update."""
        return self.local_model.get_source() + """
#include "utils/core.cl"

#define ALGORITHM_ARGS float dt, float gamma
#define ALGORITHM_BUILD(a) Algorithm a = {dt, gamma};

typedef struct {
    float dt;
    float gamma;
} Algorithm;

float4 algorithm_update_orientation(
    Algorithm algorithm,
    float* local_model,
    float4 orientation)
{
    ModelValue evaluated_model = local_model_evaluate(local_model, orientation);

    // Get spherical angles (colatitude 'theta' and azimuth 'phi')
    const float2 angles = cart2sph(orientation);
    float st, ct;
    float sp, cp;
    sp = sincos(angles.x, &cp);
    st = sincos(angles.y, &ct);

    // Tangent frame vectors (et = colatitude tangent, ep = azimuth tangent).
    // ep is kept as a unit vector; the 1/sin(theta) factor for the gradient
    // is applied explicitly in the drift, guarded to avoid division by zero.
    const float4 et = (float4)(ct * cp, ct * sp, -st, 0.0f);
    const float4 ep = (float4)(-sp, cp, 0.0f, 0.0f);

    // Calculate SDE tangent vector.
    const float4 drift = (evaluated_model.dtheta * et + (evaluated_model.dphi) * ep) / max(evaluated_model.value, 1e-6f);
    const float scale_drift = algorithm.gamma * algorithm.dt;
    const float4 tangent = scale_drift * drift;

    // The final result is the new orientation after moving along
    // the tangent vector.
    return exps2(orientation, tangent, 1.0f);
}
        """

    def get_args(self) -> list[Any]:
        """Return the scalar OpenCL arguments for this algorithm."""
        return [clt.float(self.step_size), clt.float(self.gamma)]

    def update_orientation(self, local_models: np.ndarray, orientations: Orientations):
        """Update orientations using the local model's NumPy implementation.

        Args:
            local_models: Model parameters, with one row per orientation.
            orientations: Input directions with shape ``(n_orientations, 3)``.

        Returns:
            Updated directions with shape ``(n_orientations, 3)``.
        """

        n_orientations = orientations.shape[0]
        model_values = self.local_model.evaluate(local_models, orientations)

        # Get spherical angles (colatitude 'theta' and azimuth 'phi')
        phi, theta, _ = cart2sph(*orientations.T)
        sp = np.sin(phi)
        cp = np.cos(phi)
        st = np.sin(theta)
        ct = np.cos(theta)

        # Tangent frame vectors (et = colatitude tangent, ep = azimuth tangent).
        # ep is kept as a unit vector; the 1/sin(theta) factor for the gradient
        # is applied explicitly in the drift, guarded to avoid division by zero.
        et = np.stack([ct * cp, ct * sp, -st], axis=-1)
        ep = np.stack([-sp, cp, np.zeros(n_orientations)], axis=-1)

        dtheta = model_values[:, 1, np.newaxis]
        dphi = model_values[:, 2, np.newaxis]
        value = model_values[:, 0, np.newaxis]
        drift = (dtheta * et + dphi * ep) / np.maximum(value, 1e-6)

        # Calculate SDE tangent vector.
        tangent = self.step_size * self.gamma * drift

        # The final result is the new orientation after moving along
        # the tangent vector.
        return exps2(orientations, tangent, 1.0)


def exps2(p: np.ndarray, x: np.ndarray, t: float) -> np.ndarray:
    """Apply the spherical exponential map to a batch of points.

    Assumes ``p`` contains unit vectors and ``x`` contains tangent vectors
    at those points.

    Args:
        p: Points on the sphere with shape ``(n_points, 3)``.
        x: Tangent vectors with shape ``(n_points, 3)``.
        t: Step multiplier.

    Returns:
        Updated points with shape ``(n_points, 3)``.
    """
    n = np.linalg.norm(x, axis=1)
    c = np.cos(t * n)
    s = np.sin(t * n)
    scaled_sine = np.divide(s, n, out=np.zeros_like(n), where=n != 0)
    return c[:, np.newaxis] * p + x * scaled_sine[:, np.newaxis]


def update_orientation(algorithm: BaseAlgorithm, local_models: np.ndarray, orientations: Orientations) -> Orientations:
    """Update orientations using the OpenCL implementation.

    Args:
        algorithm: Transport configuration and local model.
        local_models: Model parameters, with one row per orientation.
        orientations: Input directions with shape ``(n_orientations, 3)``.

    Returns:
        Updated directions with shape ``(n_orientations, 3)``.

    Raises:
        ValueError: If ``local_models`` does not have one correctly sized
            row per orientation.
    """

    # Stop early on empty orientations array.
    n_orientations = orientations.shape[0]
    if n_orientations == 0:
        return np.empty((0, 3), dtype=np.float32)

    # Verify the local model shape.
    if local_models.ndim != 2 or local_models.shape[1] != algorithm.local_model.n_parameters or local_models.shape[0] != n_orientations:
        raise ValueError(
            f"Expected local_models to have shape ({n_orientations}, {algorithm.local_model.n_parameters}), "
            f"but got {local_models.shape}"
        )

    # Create the global OpenCL context.
    _context = cl.create_some_context(interactive=False)
    _queue = cl.CommandQueue(_context)

    # Create the OpenCL program from the domain source.
    source = algorithm.get_source() + Template("""
    __kernel void update_orientation(
        ALGORITHM_ARGS,
        __global const float* local_models,
        __global const float4* orientations,
        uint n_orientations,
        __global float4* results)
    {
        uint gid = get_global_id(0);
        if (gid >= n_orientations) return;

        ALGORITHM_BUILD(algorithm);

        float tmp[$N_PARAMETERS];
        for (int i = 0; i < $N_PARAMETERS; ++i) {
            tmp[i] = local_models[gid * $N_PARAMETERS + i];
        }
        results[gid] = algorithm_update_orientation(algorithm, tmp, orientations[gid]);
    }
    """).substitute(N_PARAMETERS=algorithm.local_model.n_parameters)

    local_models_data = np.ascontiguousarray(local_models, dtype=clt.float)
    local_models_buffer = cl.Buffer(_context, READ_ONLY, hostbuf=local_models_data)
    
    # Orientations are stored in a (n_points, 4) array with the last component set
    # to 0 to match the OpenCL float4 format.
    orientations_data = np.zeros((n_orientations, 4), dtype=clt.float)
    orientations_data[:, :3] = orientations
    orientations_buffer = cl.Buffer(_context, READ_ONLY, hostbuf=orientations_data)

    # Prepare the results buffer.
    result = np.empty((n_orientations, 4), dtype=clt.float)
    results_buffer = cl.Buffer(_context, WRITE_ONLY, size=result.nbytes)
    args = algorithm.get_args() + [
        local_models_buffer,
        orientations_buffer,
        clt.uint(n_orientations),
        results_buffer,
    ]

    # Run and transfer the results back to the host.
    program = cl.Program(_context, source).build(f"-I {_OPENCL_DIR}")
    program.update_orientation(_queue, (n_orientations,), None, *args)
    cl.enqueue_copy(_queue, result, results_buffer)

    return result[:, :3].copy()