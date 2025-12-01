#ifndef __DIFFUSION_CORE__
#define __DIFFUSION_CORE__

/**
 * UPDATE_ORIENTATION	
 * Computes the next fiber orientation using a Stochastic Differential Equation (SDE)
 * based on the FOD's spherical harmonic representation.
 *
 * fod: Pointer to the flat FOD array with a shape specified by dims.
 * ylm: Spherical harmonic basis for current orientation.
 * ylm_dp: Derivative of ylm w.r.t azimuth.
 * ylm_dt: Derivative of ylm w.r.t colatitude.
 * index: The 3D coordinate (x, y, z) of the voxel currently being processed.
 * dims: The dimensions {nx, ny, nz, n_orientations} of the flattened FOD array.
 * orientation: The current orientation vector.
 * state: State for the pseudo-random number generator.
 * dt: Time step size.
 * gamma: Scaling factor (inverse curvature).
 * noise_variance: The variance of the noise component.
 *
 * Returns the new orientation vector.
 */
float4 update_orientation(
    __global const float *restrict fod,
    const float *restrict ylm,
    const float *restrict ylm_dp,
    const float *restrict ylm_dt,
    uint3 index,
    uint4 dims,
    float4 orientation,
    uint2 *state,
    float dt,
    float gamma,
    float noise_variance)
{
    // Local copies of dimensions.
    const uint ny = dims.y;
    const uint nz = dims.z;
    const uint n_coefficients = dims.w;

    // Calculate the base index for the current voxel (x, y, z).
    const size_t base_index = (size_t) index.x * (ny * nz * n_coefficients) +
                              (size_t) index.y * (nz * n_coefficients) +
                              (size_t) index.z * n_coefficients;

    float fod_value = 0.0f;
    float fod_colatitude_value = 0.0f;
    float fod_azimuth_value = 0.0f;
    
	// Evaluate the FOD and its derivatives.
    for (uint i = 0; i < n_coefficients; i++) {
        const float c = fod[base_index + i];
        fod_value            += c * ylm[i];
        fod_colatitude_value += c * ylm_dt[i];
        fod_azimuth_value    += c * ylm_dp[i];
    }
    
	// Softmax to ensure FOD > 0.
    const float scale_factor = 100.0f;
    const float d = dsoftmax(fod_value, scale_factor);
    const float s = softmax(fod_value, scale_factor);
    const float drift_theta = fod_colatitude_value * d;
    const float drift_phi   = fod_azimuth_value * d;
    
    // Check for division by zero / very small denominator.
    if (s <= 1e-6f) {
        return (float4)(0.0f);
    }

    // Get spherical angles (colatitude 'theta' and azimuth 'phi')
    const float2 angles = cart2sph(orientation);
    float st, ct;
    float sp, cp;
    sp = sincos(angles.x, &cp);
    st = sincos(angles.y, &ct);

    // Tangent frame vectors (et = colatitude tangent, ep = azimuth tangent)
    const float4 et = (float4)(ct * cp, ct * sp, -st, 0.0f);
    const float4 ep = (float4)(-sp, cp, 0.0f, 0.0f);

    const float4 drift = (drift_theta * et + drift_phi * ep) / s; 
    const float4 noise = randn(state) * et + randn(state) * ep;

    // Calculate SDE tangent vector.
    const float scale_drift = gamma * dt;
    const float scale_noise = native_sqrt(noise_variance * gamma * dt);

    const float4 tangent = (scale_drift) * drift + (scale_noise) * noise;
    
    // The final result is the new orientation after moving along
	// the tangent vector.
    return exps2(orientation, tangent, 1.0f);
}

#endif
