#ifndef __SYMMETRIC_REAL_SPHERICAL_HARMONICS__
#define __SYMMETRIC_REAL_SPHERICAL_HARMONICS__

#include "utils/core.cl"
#include "utils/spharm.cl"

#define SOFTMAX_SCALE 100.0f

/**
 * NEAREST NEIGHBOR SYMMETRIC REAL SPHERICAL HARMONICS
 * This local model represents a symmetric fiber orientation distribution as linear combination of
 * real spherical harmonics. It is the model used by MRtrix3, see [1]. The interpolation is
 * piecewise constant, that is nearest-neighbor.
 *
 * [1] https://mrtrix.readthedocs.io/en/latest/concepts/spherical_harmonics.html.
 *
**/

/**
 * INTERPOLATE_MODEL
 * Nearest-neighbor interpolation of spherical harmonics coefficients.
**/
void interpolate_model(__global const float *model, uint4 model_shape, float3 voxel, float *interpolated_model) {

    // Nearest-neighbor voxel.
	uint3 index = to_index(voxel);

    // Local copies of dimensions.
    const uint ny = model_shape.y;
    const uint nz = model_shape.z;
    const uint n_coefficients = model_shape.w;

    // Calculate the base index for the current voxel (x, y, z).
    const size_t base_index = (size_t) index.x * (ny * nz * n_coefficients) +
                              (size_t) index.y * (nz * n_coefficients) +
                              (size_t) index.z * n_coefficients;

    // Copy the local model.
    for (uint i = 0; i < model_shape[3]; i++) {
        interpolated_model[i] = model[base_index + i];
    }
}

/**
 * EVALUATE_INTERPOLATED_MODEL
 * Evaluates an interpolated spherical harmonics model that no longer depends on
 * location.
**/
model_value_t evaluate_interpolated_model(float *interpolated_model, uint model_shape, float4 orientation) {
    
    float ylm[45];
	float ylm_dt[45];
	float ylm_dp[45]; 

	// Pre-compute the spherical harmonics at the point of interest.
	float2 angles = cart2sph(orientation);
	evaluate_sh_dphi_over_sin_theta(angles.x, angles.y, ylm, ylm_dp, ylm_dt);

	model_value_t evaluated_model = (model_value_t) {0.0f, 0.0f, 0.0f};
    
	// Evaluate the FOD and its derivatives.
    for (uint i = 0; i < model_shape; i++) {
        const float c = interpolated_model[i];
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


/**
 * EVALUATE_MODEL
 * Evaluates the spherical-harmonic model and angular derivatives at a voxel
 * for the given orientation.
 *
 * model: Flattened model coefficients array with dimensions given by dims.
 * model_shape: Model dimensions as {nx, ny, nz, n_coefficients}.
 * voxel: Voxel-space position where the model is sampled.
 * orientation: Unit direction used to evaluate spherical harmonics.
 *
 * Returns a model_value_t containing value, dtheta, and dphi.
 */
model_value_t evaluate_model(__global const float *model, uint4 model_shape, float3 voxel, float4 orientation) {

	float interpolated_model[45];
    interpolate_model(model, model_shape, voxel, interpolated_model);
	return evaluate_interpolated_model(interpolated_model, model_shape[3], orientation);
}

#endif