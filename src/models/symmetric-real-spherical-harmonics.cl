#ifndef __SYMMETRIC_REAL_SPHERICAL_HARMONICS__
#define __SYMMETRIC_REAL_SPHERICAL_HARMONICS__

#include "utils/core.cl"
#include "utils/spharm.cl"


/**
 * EVALUATE_MODEL
 * Evaluates the spherical-harmonic model and angular derivatives at a voxel
 * for the given orientation.
 *
 * model: Flattened model coefficients array with dimensions given by dims.
 * dims: Model dimensions as {nx, ny, nz, n_coefficients}.
 * voxel: Voxel-space position where the model is sampled.
 * orientation: Unit direction used to evaluate spherical harmonics.
 *
 * Returns a model_value_t containing value, dtheta, and dphi.
 */
model_value_t evaluate_model(__global const float *model, uint4 dims, float3 voxel, float4 orientation) {

	float ylm[45];
	float ylm_dt[45];
	float ylm_dp[45]; 

	// Pre-compute the spherical harmonics at the point of interest.
	float2 angles = cart2sph(orientation);
	ishtmtx(angles.x, angles.y, ylm, ylm_dp, ylm_dt);

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

	model_value_t evaluated_model = (model_value_t){0.0f, 0.0f, 0.0f};
    
	// Evaluate the FOD and its derivatives.
    for (uint i = 0; i < n_coefficients; i++) {
        const float c = model[base_index + i];
        evaluated_model.value += c * ylm[i];
        evaluated_model.dtheta += c * ylm_dt[i];
        evaluated_model.dphi += c * ylm_dp[i];
    }

	// Softmax to ensure FOD > 0.
    const float d = dsoftmax(evaluated_model.value, SOFTMAX_SCALE);
	evaluated_model.value = softmax(evaluated_model.value, SOFTMAX_SCALE);
    evaluated_model.dtheta = evaluated_model.dtheta * d;
    evaluated_model.dphi = evaluated_model.dphi * d;

	return evaluated_model;
}

#endif