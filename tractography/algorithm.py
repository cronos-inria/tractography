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

    def get_rng_seed(self) -> int:
        """Return a seed for per-work-item OpenCL RNG initialization."""
        return 0

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
    float4 orientation,
    uint2* rng_state)
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


class Diffusion(BaseAlgorithm):
    """Update orientations with drift and tangent-plane Gaussian noise.

    Args:
        local_model: Model that evaluates a value and its orientation
            derivatives.
        step_size: Integration time step.
        gamma: Drift and noise scaling factor.
        noise_variance: Variance of the stochastic tangent-plane term.
        seed: Optional seed controlling the NumPy and OpenCL random streams.
    """

    def __init__(
        self,
        local_model: BaseLocalModel,
        step_size: float = 0.001,
        gamma: float = 1.0,
        noise_variance: float = 0.1,
        seed: int | None = None,
    ):
        super().__init__(local_model, step_size=step_size)
        self.gamma = gamma
        self.noise_variance = noise_variance
        self._rng = np.random.default_rng(seed)

    def get_source(self) -> str:
        """Return the OpenCL source for the diffusion orientation update."""
        return self.local_model.get_source() + """
#include "utils/core.cl"

#define ALGORITHM_ARGS float algorithm_step_size, float algorithm_gamma, float algorithm_noise_variance
#define ALGORITHM_BUILD(a) Algorithm a = {algorithm_step_size, algorithm_gamma, algorithm_noise_variance};

typedef struct {
    float step_size;
    float gamma;
    float noise_variance;
} Algorithm;

float4 algorithm_update_orientation(
    Algorithm algorithm,
    float* local_model,
    float4 orientation,
    uint2* rng_state)
{
    const ModelValue evaluated_model = local_model_evaluate(local_model, orientation);
    const float2 angles = cart2sph(orientation);
    float st, ct;
    float sp, cp;
    sp = sincos(angles.x, &cp);
    st = sincos(angles.y, &ct);

    const float4 et = (float4)(ct * cp, ct * sp, -st, 0.0f);
    const float4 ep = (float4)(-sp, cp, 0.0f, 0.0f);
    const float4 drift =
        (evaluated_model.dtheta * et + evaluated_model.dphi * ep)
        / max(evaluated_model.value, 1e-6f);

    float4 noise = (float4)(0.0f);
    if (algorithm.noise_variance > 0.0f) {
        noise = randn(rng_state) * et + randn(rng_state) * ep;
    }

    const float scale_drift = algorithm.gamma * algorithm.step_size;
    const float scale_noise = native_sqrt(
        algorithm.noise_variance * algorithm.gamma * algorithm.step_size
    );
    const float4 tangent = scale_drift * drift + scale_noise * noise;
    return exps2(orientation, tangent, 1.0f);
}
"""

    def get_args(self) -> list[Any]:
        """Return the scalar OpenCL arguments for this algorithm."""
        return [
            clt.float(self.step_size),
            clt.float(self.gamma),
            clt.float(self.noise_variance),
        ]

    def get_rng_seed(self) -> int:
        """Draw one seed; device code derives one RNG state per work item."""
        return int(self._rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))

    def update_orientation(self, local_models: np.ndarray, orientations: Orientations):
        """Update orientations using the NumPy diffusion reference."""
        n_orientations = orientations.shape[0]
        model_values = self.local_model.evaluate(local_models, orientations)

        phi, theta, _ = cart2sph(*orientations.T)
        sp, cp = np.sin(phi), np.cos(phi)
        st, ct = np.sin(theta), np.cos(theta)
        et = np.stack([ct * cp, ct * sp, -st], axis=-1)
        ep = np.stack([-sp, cp, np.zeros(n_orientations)], axis=-1)

        dtheta = model_values[:, 1, np.newaxis]
        dphi = model_values[:, 2, np.newaxis]
        value = model_values[:, 0, np.newaxis]
        drift = (dtheta * et + dphi * ep) / np.maximum(value, 1e-6)

        noise = np.zeros_like(drift)
        if self.noise_variance > 0.0:
            samples = self._rng.standard_normal((n_orientations, 2))
            noise = samples[:, :1] * et + samples[:, 1:] * ep

        scale_drift = self.gamma * self.step_size
        scale_noise = np.sqrt(
            self.noise_variance * self.gamma * self.step_size
        )
        tangent = scale_drift * drift + scale_noise * noise
        return exps2(orientations, tangent, 1.0)


class Probabilistic(BaseAlgorithm):
    """Sample a new orientation from the local model near the current one.

    Candidate directions are drawn uniformly from a spherical cap centered on
    each input orientation. One candidate is then selected with probability
    proportional to its nonnegative local-model value. This is a finite-sample
    approximation to sampling the local orientation density.

    Args:
        local_model: Model whose nonnegative value defines the sampling density.
        n_samples: Number of candidate directions drawn per input orientation.
        max_angle: Radius of the spherical sampling cap, in radians.
        seed: Optional seed controlling the NumPy and OpenCL random streams.
        step_size: Inherited tracking step size; it does not affect sampling.
    """

    def __init__(
        self,
        local_model: BaseLocalModel,
        n_samples: int = 64,
        max_angle: float = np.pi / 12,
        seed: int | None = None,
        step_size: float = 0.25,
    ):
        super().__init__(local_model, step_size=step_size)
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if not np.isfinite(max_angle) or not 0.0 <= max_angle <= np.pi:
            raise ValueError("max_angle must be finite and between 0 and pi")
        self.n_samples = int(n_samples)
        self.max_angle = float(max_angle)
        self._rng = np.random.default_rng(seed)

    def get_source(self) -> str:
        """Return the OpenCL source for probabilistic orientation sampling."""
        return self.local_model.get_source() + Template("""
#include "utils/core.cl"

#define ALGORITHM_ARGS float algorithm_step_size, float algorithm_max_angle
#define ALGORITHM_BUILD(a) Algorithm a = {algorithm_step_size, algorithm_max_angle};
#define N_SAMPLES $N_SAMPLES

typedef struct {
    float step_size;
    float max_angle;
} Algorithm;

float4 algorithm_update_orientation(
    Algorithm algorithm,
    float* local_model,
    float4 orientation,
    uint2* rng_state)
{
    const float orientation_norm = length(orientation.xyz);
    if (orientation_norm <= 1e-8f) {
        orientation = (float4)(0.0f, 0.0f, 1.0f, 0.0f);
    } else {
        orientation = (float4)(orientation.xyz / orientation_norm, 0.0f);
    }

    // Construct an orthonormal frame around the current orientation.
    const float4 helper = fabs(orientation.z) < 0.9f
        ? (float4)(0.0f, 0.0f, 1.0f, 0.0f)
        : (float4)(0.0f, 1.0f, 0.0f, 0.0f);
    const float3 tangent1_xyz = normalize(cross(helper.xyz, orientation.xyz));
    const float4 tangent1 = (float4)(tangent1_xyz, 0.0f);
    const float4 tangent2 = (float4)(cross(orientation.xyz, tangent1.xyz), 0.0f);
    const float cos_max_angle = cos(algorithm.max_angle);

    // First pass computes the total density. The RNG state is replayed in the
    // second pass, avoiding per-work-item storage for the candidate directions.
    const uint2 initial_rng_state = *rng_state;
    float total_weight = 0.0f;
    for (uint i = 0; i < N_SAMPLES; ++i) {
        const float cos_angle = 1.0f - randu(rng_state) * (1.0f - cos_max_angle);
        const float sin_angle = sqrt(max(0.0f, 1.0f - cos_angle * cos_angle));
        const float azimuth = 2.0f * PI * randu(rng_state);
        const float4 candidate = cos_angle * orientation + sin_angle *
            (cos(azimuth) * tangent1 + sin(azimuth) * tangent2);
        const float value = local_model_evaluate(local_model, candidate).value;
        total_weight += isfinite(value) ? fmax(value, 0.0f) : 0.0f;
    }

    if (total_weight <= 0.0f) {
        return orientation;
    }

    const float target = randu(rng_state) * total_weight;
    *rng_state = initial_rng_state;
    float cumulative_weight = 0.0f;
    for (uint i = 0; i < N_SAMPLES; ++i) {
        const float cos_angle = 1.0f - randu(rng_state) * (1.0f - cos_max_angle);
        const float sin_angle = sqrt(max(0.0f, 1.0f - cos_angle * cos_angle));
        const float azimuth = 2.0f * PI * randu(rng_state);
        const float4 candidate = cos_angle * orientation + sin_angle *
            (cos(azimuth) * tangent1 + sin(azimuth) * tangent2);
        const float value = local_model_evaluate(local_model, candidate).value;
        cumulative_weight += isfinite(value) ? fmax(value, 0.0f) : 0.0f;
        if (cumulative_weight > target) {
            return candidate;
        }
    }
    return orientation;
}
""").substitute(N_SAMPLES=self.n_samples)

    def get_args(self) -> list[Any]:
        """Return the scalar OpenCL arguments for this algorithm."""
        return [clt.float(self.step_size), clt.float(self.max_angle)]

    def get_rng_seed(self) -> int:
        """Draw one seed; device code derives one RNG state per work item."""
        return int(self._rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))

    def update_orientation(self, local_models: np.ndarray, orientations: Orientations):
        """Sample orientations from local-model values in spherical caps."""
        n_orientations = orientations.shape[0]
        result_dtype = np.result_type(orientations, np.float32)
        if n_orientations == 0:
            return np.empty((0, 3), dtype=result_dtype)

        unit_orientations = np.asarray(orientations, dtype=result_dtype).copy()
        norms = np.linalg.norm(unit_orientations, axis=1)
        invalid = norms <= 1e-8
        unit_orientations[~invalid] /= norms[~invalid, np.newaxis]
        unit_orientations[invalid] = (0.0, 0.0, 1.0)

        # Build a stable orthonormal tangent frame for each input direction.
        helpers = np.zeros_like(unit_orientations)
        use_z_axis = np.abs(unit_orientations[:, 2]) < 0.9
        helpers[use_z_axis, 2] = 1.0
        helpers[~use_z_axis, 1] = 1.0
        tangent1 = np.cross(helpers, unit_orientations)
        tangent1 /= np.linalg.norm(tangent1, axis=1, keepdims=True)
        tangent2 = np.cross(unit_orientations, tangent1)

        random_values = self._rng.random((n_orientations, self.n_samples, 2))
        cos_max_angle = np.cos(self.max_angle)
        cos_angles = 1.0 - random_values[:, :, 0] * (1.0 - cos_max_angle)
        sin_angles = np.sqrt(np.maximum(0.0, 1.0 - cos_angles**2))
        azimuths = 2.0 * np.pi * random_values[:, :, 1]
        candidates = (
            cos_angles[:, :, np.newaxis] * unit_orientations[:, np.newaxis, :]
            + sin_angles[:, :, np.newaxis]
            * (
                np.cos(azimuths)[:, :, np.newaxis] * tangent1[:, np.newaxis, :]
                + np.sin(azimuths)[:, :, np.newaxis] * tangent2[:, np.newaxis, :]
            )
        )

        repeated_models = np.repeat(local_models, self.n_samples, axis=0)
        values = self.local_model.evaluate(
            repeated_models, candidates.reshape(-1, 3)
        )[:, 0].reshape(n_orientations, self.n_samples)
        weights = np.where(np.isfinite(values), np.maximum(values, 0.0), 0.0)
        totals = np.sum(weights, axis=1)
        result = unit_orientations.copy()
        valid = totals > 0.0
        if np.any(valid):
            targets = self._rng.random(np.count_nonzero(valid)) * totals[valid]
            cumulative = np.cumsum(weights[valid], axis=1)
            selected = np.sum(cumulative <= targets[:, np.newaxis], axis=1)
            selected = np.minimum(selected, self.n_samples - 1)
            result[valid] = candidates[valid, selected]
        return result.astype(result_dtype, copy=False)


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
    float4 orientation,
    uint2* rng_state)
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
        uint rng_seed,
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
        uint2 rng_state = rng_init(rng_seed, gid);
        results[gid] = algorithm_update_orientation(algorithm, tmp, orientations[gid], &rng_state);
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
        clt.uint(algorithm.get_rng_seed()),
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