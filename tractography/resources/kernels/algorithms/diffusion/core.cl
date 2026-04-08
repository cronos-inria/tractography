#ifndef __DIFFUSION_CORE__
#define __DIFFUSION_CORE__

/**
 * UPDATE_ORIENTATION
 * Computes the next fibre orientation using a Stochastic Differential Equation
 * (SDE) from the model value and angular derivatives.
 *
 * evaluated_model: Model value and derivatives at current position/orientation.
 * orientation: Current orientation vector.
 * state: Pseudo-random number generator state.
 * dt: Integration time step.
 * gamma: Inverse curvature scaling.
 * noise_variance: Variance of the stochastic term.
 *
 * Returns the updated orientation vector.
 */
float4 update_orientation(
    model_value_t evaluated_model,
    float4 orientation,
    uint2 *state,
    float dt,
    float gamma,
    float noise_variance)
{
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

    const float4 drift = (evaluated_model.dtheta * et + (evaluated_model.dphi / max(st, 1e-6f)) * ep) / max(evaluated_model.value, 1e-6f);
    float4 noise = {0.0f, 0.0f, 0.0f, 0.0f};
    if (noise_variance > 0.0f) {
        noise = randn(state) * et + randn(state) * ep;
    }

    // Calculate SDE tangent vector.
    const float scale_drift = gamma * dt;
    const float scale_noise = native_sqrt(noise_variance * gamma * dt);
    const float4 tangent = scale_drift * drift + scale_noise * noise;

    // The final result is the new orientation after moving along
    // the tangent vector.
    return exps2(orientation, tangent, 1.0f);
}

#endif
