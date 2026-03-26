#ifndef __DIFFUSION_TENSOR_IMAGING__
#define __DIFFUSION_TENSOR_IMAGING__

#include "utils/core.cl"

/**
 * EVALUATE_MODEL
 * Evaluates the diffusion tensor signal model and its angular derivatives at a
 * voxel for the given orientation. The model is
 *     f(u) = u^t Du
 *
 * model: Flattened tensor coefficients array with dimensions given by dims.
 * The six coefficients are stored as {Dxx, Dyy, Dzz, Dxy, Dxz, Dyz}.
 * dims: Model dimensions as {nx, ny, nz, n_coefficients}.
 * voxel: Voxel-space position where the tensor is sampled.
 * orientation: Unit direction used to evaluate the tensor signal.
 *
 * Returns a model_value_t containing the signal value and its derivatives with
 * respect to theta and phi.
 */
model_value_t evaluate_model(__global const float *model, uint4 dims, float3 voxel, float4 orientation) {

    // Nearest-neighbor voxel.
	uint3 index = to_index(voxel);

    // Local copies of dimensions.
    const uint ny = dims.y;
    const uint nz = dims.z;
    const uint n_coefficients = dims.w;

    // Calculate the base index for the current voxel (x, y, z).
    const size_t base_index = (size_t) index.x * (ny * nz * n_coefficients) +
                              (size_t) index.y * (nz * n_coefficients) +
                              (size_t) index.z * n_coefficients;

    // Reconstruct the diffusion tensor.
    float4 tensor[4] = {0.0f, 0.0f, 0.0f, 0.0f}; 
    tensor[0][0] = model[base_index];
    tensor[1][1] = model[base_index + 1];
    tensor[2][2] = model[base_index + 2];
    tensor[0][1] = model[base_index + 3];
    tensor[1][0] = tensor[0][1];
    tensor[0][2] = model[base_index + 4];
    tensor[2][0] = tensor[0][2];
    tensor[1][2] = model[base_index + 5];
    tensor[2][1] = tensor[1][2];

    // Evaluate the DTI model.
    float4 tensor_dot_u = {0.0f, 0.0f, 0.0f, 0.0f};
    tensor_dot_u.x = dot(tensor[0], orientation);
    tensor_dot_u.y = dot(tensor[1], orientation);
    tensor_dot_u.z = dot(tensor[2], orientation);

    float u_dot_tensor_dot_u = dot(orientation, tensor_dot_u);

    float2 angles = cart2sph(orientation);
	float phi = angles.x;
	float theta = angles.y;

	float st, ct, sp, cp;
	st = sincos(theta, &ct);
	sp = sincos(phi, &cp);

    float4 u_phi = {-st * sp, st * cp, 0.0f, 0.0f};
    float4 u_theta = {ct * cp, ct * sp, -st, 0.0f};

    model_value_t evaluated_model = (model_value_t) {0.0f, 0.0f, 0.0f};
    evaluated_model.value = u_dot_tensor_dot_u;
    evaluated_model.dphi = 2.0f * dot(tensor_dot_u, u_phi);
    evaluated_model.dtheta = 2.0f * dot(tensor_dot_u, u_theta);

	return evaluated_model;
}

#endif