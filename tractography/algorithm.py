from abc import ABC, abstractmethod
from string import Template
from typing import Any
from importlib.resources import files

import numpy as np
import pyopencl as cl
import pyopencl.cltypes as clt
from scipy.optimize import minimize

from .local import BaseLocalModel, Orientations
from .core import cart2sph

_OPENCL_DIR = files("tractography.resources") / "kernels"
READ_ONLY = cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR
WRITE_ONLY = cl.mem_flags.WRITE_ONLY


class BaseAlgorithm(ABC):
    """Abstract interface for algorithms that update orientations."""

    def __init__(self, local_model: BaseLocalModel, step_size: float = 1.0):
        self.local_model = local_model
        self.step_size = step_size

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
    """Configure transport orientation updates from a local model.

    Args:
        local_model: Model that evaluates a value and its orientation
            derivatives.
        step_size: Step duration used to scale the drift.
        gamma: Drift scaling factor.
    """

    def __init__(self, local_model: BaseLocalModel, step_size: float = 0.001, gamma: float = 1.0):
        super().__init__(local_model, step_size=step_size)
        self.gamma = gamma

    def get_source(self) -> str:
        """Return the OpenCL source for the transport orientation update."""
        return self.local_model.get_source() + """
#include "utils/core.cl"

#define ALGORITHM_ARGS float algorithm_step_size, float algorithm_gamma
#define ALGORITHM_BUILD(a) Algorithm a = {algorithm_step_size, algorithm_gamma};

typedef struct {
    float step_size;
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
    const float scale_drift = algorithm.gamma * algorithm.step_size;
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


class Deterministic(BaseAlgorithm):
    """Maximize the local model near each input direction on the unit sphere.

    The NumPy implementation is a reference optimizer: it minimizes the
    negative model value with SciPy's SLSQP method, enforces unit length as an
    equality constraint, and supplies the model's analytic gradient. It
    normalizes trial directions when evaluating the objective and normalizes
    the returned direction. ``tolerance`` and ``max_iterations`` configure
    SLSQP. ``max_step_size`` is not used by this implementation.

    The OpenCL implementation performs Newton ascent in a two-dimensional
    tangent frame. The local model provides the gradient, while the Hessian is
    estimated from nearby gradients. If the estimated Hessian is negative
    definite, the update uses the Newton direction. Otherwise, it takes a
    fixed small gradient-ascent step. It caps the update angle at
    ``max_step_size``, maps the candidate back to the sphere with the
    exponential map, and backtracks until the model value increases or the
    search is exhausted. The inherited ``step_size`` attribute belongs to
    tracking and does not control either orientation update.

    The model's ``dphi`` is already divided by ``sin(theta)``. It therefore
    directly multiplies the unit azimuth tangent vector and must not be scaled
    by ``sin(theta)`` again. Both implementations seek a nearby local maximum,
    not necessarily the global maximum.

    Args:
        local_model: Model whose value is maximized for each input orientation.
        max_iterations: Maximum number of optimizer iterations per orientation.
        tolerance: SLSQP objective tolerance and OpenCL gradient-norm threshold.
        max_step_size: Maximum OpenCL orientation-update angle, in radians.
    """

    def __init__(
        self,
        local_model: BaseLocalModel,
        max_iterations: int = 10,
        tolerance: float = 1e-5,
        max_step_size: float = 0.5,
        step_size: float = 0.1,
    ):
        super().__init__(local_model, step_size=step_size)
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.max_step_size = max_step_size

    def get_source(self) -> str:
        """Return OpenCL source for the deterministic Newton update."""
        return self.local_model.get_source() + """
#include "utils/core.cl"

#define ALGORITHM_ARGS float algorithm_step_size,uint max_iterations, float tolerance, float max_step_size
#define ALGORITHM_BUILD(a) Algorithm a = {algorithm_step_size, max_iterations, tolerance, max_step_size};

typedef struct {
    float step_size;
    uint max_iterations;
    float tolerance;
    float max_step_size;
} Algorithm;

float4 deterministic_gradient(float* local_model, float4 orientation)
{
    const ModelValue value = local_model_evaluate(local_model, orientation);
    const float2 angles = cart2sph(orientation);
    float st, ct, sp, cp;
    sp = sincos(angles.x, &cp);
    st = sincos(angles.y, &ct);
    const float4 e1 = (float4)(ct * cp, ct * sp, -st, 0.0f);
    const float4 e2 = (float4)(-sp, cp, 0.0f, 0.0f);
    // The local model supplies the spherical gradient components. In
    // particular, dphi already includes the 1/sin(theta) factor, so no
    // additional coordinate scaling is needed here.
    return value.dtheta * e1 + value.dphi * e2;
}

float4 deterministic_line_search(
    float* local_model,
    float4 orientation,
    float4 tangent,
    float current_value,
    bool* accepted)
{
    // Try successively shorter geodesic steps and keep the first one that
    // strictly increases the local-model value.
    float scale = 1.0f;
    *accepted = false;
    for (uint iteration = 0; iteration < 8; ++iteration) {
        const float4 candidate = exps2(orientation, scale * tangent, 1.0f);
        const float candidate_value = local_model_evaluate(local_model, candidate).value;
        if (candidate_value > current_value + 1e-7f) {
            *accepted = true;
            return candidate;
        }
        scale *= 0.5f;
    }
    return orientation;
}

float4 algorithm_update_orientation(
    Algorithm algorithm,
    float* local_model,
    float4 orientation)
{
    // Work with a unit direction so the tangent frame and spherical
    // exponential map have their expected geometry.
    const float h = 1e-2f;
    orientation = normalize(orientation);

    for (uint iteration = 0; iteration < algorithm.max_iterations; ++iteration) {
        // Build an orthonormal tangent basis at the current orientation.
        const float2 angles = cart2sph(orientation);
        float st, ct, sp, cp;
        sp = sincos(angles.x, &cp);
        st = sincos(angles.y, &ct);
        const float4 e1 = (float4)(ct * cp, ct * sp, -st, 0.0f);
        const float4 e2 = (float4)(-sp, cp, 0.0f, 0.0f);

        // Use the model-provided gradient at the current orientation.
        const ModelValue current = local_model_evaluate(local_model, orientation);
        const float f0 = current.value;
        const float g1 = current.dtheta;
        const float g2 = current.dphi;
        if (sqrt(g1 * g1 + g2 * g2) <= algorithm.tolerance) {
            break;
        }

        const float4 step1 = h * e1;
        const float4 step2 = h * e2;
        const float4 gp1 = deterministic_gradient(local_model, exps2(orientation, step1, 1.0f));
        const float4 gm1 = deterministic_gradient(local_model, exps2(orientation, -step1, 1.0f));
        const float4 gp2 = deterministic_gradient(local_model, exps2(orientation, step2, 1.0f));
        const float4 gm2 = deterministic_gradient(local_model, exps2(orientation, -step2, 1.0f));

        // Estimate the covariant Hessian by differencing nearby gradients
        // along geodesics and projecting their changes onto this tangent basis.
        const float h11 = dot(gp1 - gm1, e1) / (2.0f * h);
        const float h22 = dot(gp2 - gm2, e2) / (2.0f * h);
        const float h12a = dot(gp2 - gm2, e1) / (2.0f * h);
        const float h12b = dot(gp1 - gm1, e2) / (2.0f * h);
        const float h12 = 0.5f * (h12a + h12b);
        const float determinant = h11 * h22 - h12 * h12;

        float d1, d2;
        if (h11 < -1e-7f && h22 < -1e-7f && determinant > 1e-12f) {
            // Solve the 2-D Newton-ascent system: delta = -H^-1 * gradient.
            d1 = (-h22 * g1 + h12 * g2) / determinant;
            d2 = (h12 * g1 - h11 * g2) / determinant;
        } else {
            // Away from a negative-definite maximum, take a small ascent step
            // instead of applying an unstable or non-ascent Newton direction.
            const float gradient_norm = sqrt(g1 * g1 + g2 * g2);
            d1 = 0.1f * g1 / gradient_norm;
            d2 = 0.1f * g2 / gradient_norm;
        }

        const float step_norm = sqrt(d1 * d1 + d2 * d2);
        if (step_norm > algorithm.max_step_size) {
            // Limit the angular displacement for stability.
            const float scale = algorithm.max_step_size / step_norm;
            d1 *= scale;
            d2 *= scale;
        }
        if (sqrt(d1 * d1 + d2 * d2) <= 1e-7f) {
            break;
        }

        bool accepted = false;
        const float4 tangent = d1 * e1 + d2 * e2;
        orientation = deterministic_line_search(
            local_model, orientation, tangent, f0, &accepted
        );
        if (!accepted) {
            break;
        }
    }
    return orientation;
}
"""

    def get_args(self) -> list[Any]:
        """Return the scalar OpenCL arguments for this algorithm."""
        return [
            clt.float(self.step_size),
            clt.uint(self.max_iterations),
            clt.float(self.tolerance),
            clt.float(self.max_step_size),
        ]

    def update_orientation(self, local_models: np.ndarray, orientations: Orientations):
        """Maximize each local model with constrained SciPy optimization."""
        result = np.array(orientations, dtype=np.result_type(orientations, np.float32), copy=True)

        for i, initial_orientation in enumerate(result):
            initial_orientation = np.asarray(initial_orientation, dtype=np.float64)
            initial_orientation /= np.linalg.norm(initial_orientation)
            parameters = local_models[i : i + 1]

            def objective(orientation):
                norm = np.linalg.norm(orientation)
                direction = orientation / norm
                model_value = self.local_model.evaluate(
                    parameters, direction[None, :]
                )[0]
                phi, theta, _ = cart2sph(*direction)
                st, ct = np.sin(theta), np.cos(theta)
                sp, cp = np.sin(phi), np.cos(phi)
                e1 = np.array([ct * cp, ct * sp, -st])
                e2 = np.array([-sp, cp, 0.0])

                # dphi already contains the 1/sin(theta) gradient factor.
                gradient = model_value[1] * e1 + model_value[2] * e2

                # Account for the normalization used to map solver vectors
                # onto the sphere when returning the Euclidean Jacobian.
                gradient = (gradient - direction * np.dot(direction, gradient)) / norm
                return -model_value[0], -gradient

            if self.max_iterations <= 0:
                result[i] = initial_orientation
                continue

            optimization = minimize(
                objective,
                initial_orientation,
                method="SLSQP",
                jac=True,
                constraints={
                    "type": "eq",
                    "fun": lambda orientation: np.dot(orientation, orientation) - 1.0,
                    "jac": lambda orientation: 2.0 * orientation,
                },
                options={
                    "ftol": self.tolerance,
                    "maxiter": self.max_iterations,
                },
            )
            result[i] = optimization.x / np.linalg.norm(optimization.x)
        return result


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
        algorithm: Orientation-update algorithm and its local model.
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