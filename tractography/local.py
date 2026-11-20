from abc import ABC, abstractmethod
from collections.abc import Sequence
from importlib.resources import files
from string import Template
from typing import Any

import numpy as np
import pyopencl.cltypes as clt

from .core import ishtmtx, cart2sph
from .opencl import BufferAccess, OpenCLRuntime

type Orientations = np.ndarray[tuple[int, int], Any]
type LocalModelValues = np.ndarray[tuple[int, int], np.dtype[clt.float]]

_OPENCL_DIR = files("tractography.resources") / "kernels"
SOFTMAX_SCALE = 100.0


class BaseLocalModel(ABC):
    """Abstract interface for a local model evaluated at orientations."""

    opencl_model: str

    def __init__(self, n_parameters: int):
        """Initialize the model with its required coefficient count.

        Args:
            n_parameters: Number of parameters used to evaluate the model.
        """
        self.n_parameters = n_parameters

    def get_source(self) -> str:
        """Return OpenCL source defining the local model implementation.

        The source must implement ``local_model_evaluate`` and return the
        model value and its derivatives with respect to theta and phi.
        
        IMPORTANT: The derivative with respect to phi must be pre-divided
        by ``sin(theta)``.
        """

        return """
        typedef struct {
            float value;
            float dtheta;
            float dphi;
        } ModelValue;
        """

    @abstractmethod
    def evaluate(
            self, 
            parameters: np.ndarray,
            orientations: Orientations,
        ) -> LocalModelValues:
        """Evaluate the local model at the given orientations.

        Args:
            parameters: Real symmetric spherical-harmonic coefficients with
                one row per orientation and shape
                (n_orientations, n_parameters).
            orientations: Array of shape (n_orientations, 3) containning the
                xyz coordinates of the orientations.
            
    
        Returns:
            Model values and orientation derivatives, in a format defined by
            the concrete model.
        """
        pass


class RealSymmetricSphericalHarmonics(BaseLocalModel):
    """Real symmetric spherical-harmonic model with a softmax output.
    
    This is the local model used by MRtrix and Dipy for fiber orientation
    distribution estimation. See https://userdocs.mrtrix.org/en/latest/concepts/spherical_harmonics.html
    
    """

    opencl_model = "MODEL_SYMMETRIC_REAL_SPHERICAL_HARMONICS"

    def get_source(self) -> str:
        """Return OpenCL source for evaluating this spherical-harmonic model."""
        return super().get_source() + Template("""
        #include "utils/core.cl"
        #include "utils/spharm.cl"
        #define SOFTMAX_SCALE $SOFTMAX_SCALE

        ModelValue local_model_evaluate(float* model, float4 orientation) {
            
            // The helper fills all 45 supported harmonics, even when this
            // model uses only the first 15 coefficients.
            float ylm[45];
            float ylm_dt[45];
            float ylm_dp[45];

            // Pre-compute the spherical harmonics at the point of interest.
            float2 angles = cart2sph(orientation);
            evaluate_sh_dphi_over_sin_theta(angles.x, angles.y, ylm, ylm_dp, ylm_dt);

            ModelValue result = (ModelValue) {0.0f, 0.0f, 0.0f};
            
            // Evaluate the FOD and its derivatives.
            for (uint i = 0; i < $N_PARAMETERS; i++) {
                const float c = model[i];
                result.value += c * ylm[i];
                result.dtheta += c * ylm_dt[i];
                result.dphi += c * ylm_dp[i];
            }

            // Softmax to ensure FOD > 0.
            const float d = dsoftmax(result.value, SOFTMAX_SCALE);
            result.value = softmax(result.value, SOFTMAX_SCALE);
            result.dtheta = result.dtheta * d;
            result.dphi = result.dphi * d;

            return result;
        }
        """).safe_substitute({"N_PARAMETERS": self.n_parameters, "SOFTMAX_SCALE": SOFTMAX_SCALE})

    def evaluate(
            self, 
            parameters: np.ndarray,
            orientations: Orientations,
        ) -> LocalModelValues:
        """Evaluate the model and orientation derivatives using NumPy.

        This implementation serves as a reference. Use the module-level
        ``evaluate`` function for OpenCL execution.

        Args:
            parameters: Real symmetric spherical-harmonic coefficients with
                one row per orientation and shape
                (n_orientations, n_parameters).
            orientations: Array of shape (n_orientations, 3) containing the
            xyz coordinates of the orientations.

        Returns:
            An array with columns for the softmax-transformed model value,
            theta derivative, and phi derivative divided by ``sin(theta)``.
        """

        # Stop early on empty orientations.
        if orientations.shape[0] == 0:
            return np.empty((0, 3), dtype=clt.float)
        
        def softmax(x, k):
            return np.where(x * k < 30.0, np.log(1 + np.exp(k * x)) / k, x)

        def dsoftmax(x, k):
            return 1.0 / (1 + np.exp(-k * x))

        phi, theta, _ = cart2sph(*orientations.T)
        mtx, mtx_derivatives = ishtmtx(phi, theta, self.n_parameters)
        value = np.einsum("ij,ij->i", mtx, parameters)
        d = dsoftmax(value, SOFTMAX_SCALE)

        value = softmax(value, SOFTMAX_SCALE)

        # The derivative with respect to phi needs to be divided by sin(theta).
        dphi_numerator = np.einsum("ij,ij->i", mtx_derivatives[1], parameters)
        sin_theta = np.sin(theta)
        at_pole = np.isclose(sin_theta, 0.0, atol=1e-7, rtol=0.0)
        dphi = np.divide(
            dphi_numerator,
            sin_theta,
            out=np.zeros_like(dphi_numerator),
            where=~at_pole,
        )
        dtheta = np.einsum("ij,ij->i", mtx_derivatives[0], parameters)
        if np.any(at_pole):
            
            # Evaluate the finite pole limits from inside the valid polar
            # range, avoiding the coordinate singularity and endpoint
            # derivative ambiguity.
            pole_theta = np.where(np.cos(theta[at_pole]) >= 0, 1e-6, np.pi - 1e-6)
            _, pole_derivatives = ishtmtx(
                phi[at_pole], pole_theta, self.n_parameters
            )
            pole_parameters = parameters[at_pole]
            dtheta[at_pole] = np.einsum(
                "ij,ij->i", pole_derivatives[0], pole_parameters
            )
            dphi[at_pole] = np.einsum(
                "ij,ij->i", pole_derivatives[1], pole_parameters
            ) / np.sin(pole_theta)
        return np.stack([value, dtheta * d, dphi * d], axis=-1)


class DiffusionTensorImaging(BaseLocalModel):
    """Evaluate a symmetric diffusion tensor along each orientation.

    Each parameter row stores the six independent tensor coefficients in
    ``(Dxx, Dyy, Dzz, Dxy, Dxz, Dyz)`` order. The model value is
    ``u.T @ D @ u``. Its output columns contain the value, the derivative
    with respect to the polar angle, and the azimuthal derivative divided
    by ``sin(theta)``.
    """

    opencl_model = "MODEL_DTI"

    def __init__(self):
        super().__init__(6)

    def get_source(self) -> str:
        """Return OpenCL source for evaluating the tensor encoding."""
        return super().get_source() + """
#include "utils/core.cl"

ModelValue local_model_evaluate(float* model, float4 orientation) {
    const float4 tensor_dot_u = (float4)(
        model[0] * orientation.x + model[3] * orientation.y + model[4] * orientation.z,
        model[3] * orientation.x + model[1] * orientation.y + model[5] * orientation.z,
        model[4] * orientation.x + model[5] * orientation.y + model[2] * orientation.z,
        0.0f);
    const float2 angles = cart2sph(orientation);
    float st, ct, sp, cp;
    sp = sincos(angles.x, &cp);
    st = sincos(angles.y, &ct);

    const float4 e_theta = (float4)(ct * cp, ct * sp, -st, 0.0f);
    const float4 e_phi = (float4)(-sp, cp, 0.0f, 0.0f);
    ModelValue result;
    result.value = dot(orientation, tensor_dot_u);
    result.dtheta = 2.0f * dot(tensor_dot_u, e_theta);
    result.dphi = 2.0f * dot(tensor_dot_u, e_phi);
    return result;
}
"""

    def evaluate(
        self,
        parameters: np.ndarray,
        orientations: Orientations,
    ) -> LocalModelValues:
        """Evaluate tensor values and angular derivatives with NumPy."""
        n_orientations = orientations.shape[0]
        if n_orientations == 0:
            return np.empty((0, 3), dtype=clt.float)

        tensor = np.zeros((n_orientations, 3, 3), dtype=np.result_type(parameters, orientations))
        tensor[:, 0, 0] = parameters[:, 0]
        tensor[:, 1, 1] = parameters[:, 1]
        tensor[:, 2, 2] = parameters[:, 2]
        tensor[:, 0, 1] = tensor[:, 1, 0] = parameters[:, 3]
        tensor[:, 0, 2] = tensor[:, 2, 0] = parameters[:, 4]
        tensor[:, 1, 2] = tensor[:, 2, 1] = parameters[:, 5]

        tensor_dot_u = np.einsum("nij,nj->ni", tensor, orientations)
        value = np.einsum("ni,ni->n", orientations, tensor_dot_u)
        phi, theta, _ = cart2sph(*orientations.T)
        sin_theta, cos_theta = np.sin(theta), np.cos(theta)
        sin_phi, cos_phi = np.sin(phi), np.cos(phi)
        e_theta = np.stack(
            [cos_theta * cos_phi, cos_theta * sin_phi, -sin_theta], axis=-1
        )
        e_phi = np.stack([-sin_phi, cos_phi, np.zeros(n_orientations)], axis=-1)
        dtheta = 2.0 * np.einsum("ni,ni->n", tensor_dot_u, e_theta)
        dphi = 2.0 * np.einsum("ni,ni->n", tensor_dot_u, e_phi)
        return np.stack([value, dtheta, dphi], axis=-1).astype(clt.float)


def from_shape(shape: Sequence[int]) -> BaseLocalModel:
    """Create the built-in local model matching a field's coefficient count.

    Six coefficients select diffusion tensor imaging; 45 select real
    symmetric spherical harmonics. The returned model also provides the
    OpenCL model identifier used by kernels that need compile-time dispatch.
    """
    if len(shape) == 0:
        raise ValueError("A local-model shape must include a coefficient dimension.")

    n_parameters = shape[-1]
    if n_parameters == 6:
        return DiffusionTensorImaging()
    if n_parameters == 45:
        return RealSymmetricSphericalHarmonics(n_parameters)
    raise ValueError(
        "For now, only DTI and fODF with 45 coefficients are supported (lmax=8)."
    )


def evaluate(
        model: BaseLocalModel,
        parameters: np.ndarray,
        orientations: Orientations,
        runtime: OpenCLRuntime | None = None,
    ) -> LocalModelValues:
    """Evaluate a local model at multiple orientations using OpenCL.

    Args:
        model: Local model providing the OpenCL implementation.
        parameters: Model coefficients with one row per orientation and shape
                (n_orientations, n_parameters).
        orientations: Array of shape (n_orientations, 3) containing the
                      xyz coordinates of the orientations.
        runtime: Optional runtime to reuse. A runtime is created when omitted.

    Returns:
        For non-empty input, an array with one row per orientation and
        columns for the model value, theta derivative, and phi derivative
        divided by ``sin(theta)``. For empty input, this function
        returns an empty array of shape ``(0, 3)``.
    """

    # Stop early on empty orientations.
    n_orientations = orientations.shape[0]
    if n_orientations == 0:
        return np.empty((0, 3), dtype=clt.float)
    if runtime is None:
        runtime = OpenCLRuntime()

    # Create the OpenCL program from the domain source.
    source = model.get_source() + Template("""
    __kernel void evaluate(
        __global const float* model,
        __global const float4* orientations,
        uint n_orientations,
        __global float* results)
    {
        uint gid = get_global_id(0);
        if (gid >= n_orientations) return;

        float tmp[$N_PARAMETERS];
        for (int i = 0; i < $N_PARAMETERS; ++i) {
            tmp[i] = model[gid * $N_PARAMETERS + i];
        }
        ModelValue mv = local_model_evaluate(tmp, orientations[gid]);
        results[gid * 3 + 0] = mv.value;
        results[gid * 3 + 1] = mv.dtheta;
        results[gid * 3 + 2] = mv.dphi;
    }
    """).safe_substitute({"N_PARAMETERS": model.n_parameters})

    # Create the OpenCL buffer for the model parameters.
    parameters_data = np.ascontiguousarray(parameters, dtype=clt.float)
    model_buffer = runtime.buffer_from_array(parameters_data, BufferAccess.READ_ONLY)
    
    # Orientations are stored as a 2D array of shape (n_orientations, 3).
    orientations_data = np.zeros((n_orientations, 4), dtype=clt.float)
    orientations_data[:, :3] = orientations
    orientations_buffer = runtime.buffer_from_array(
        orientations_data, BufferAccess.READ_ONLY
    )

    # Prepare the results buffer.
    result = np.empty((n_orientations, 3), dtype=clt.float)
    results_buffer = runtime.allocate_buffer(result.nbytes)
    args = [
        model_buffer,
        orientations_buffer,
        clt.uint(n_orientations),
        results_buffer,
    ]

    # Run and transfer the results back to the host.
    program = runtime.build_program(source, options=("-I", str(_OPENCL_DIR)))
    program.evaluate(runtime.queue, (n_orientations,), None, *args)
    runtime.copy_from_buffer(results_buffer, result, is_blocking=True)

    return result