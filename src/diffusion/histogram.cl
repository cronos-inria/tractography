#include "utils/spharm.cl"
#include "seeds.cl"

inline void atomic_add_global_float(__global float *global_val, float local_val)
{
    union {
        unsigned int int_val;
        float float_val;
    } old, newval;

    do {
        // Read old value
        old.float_val = *global_val;
        // Compute new value
        newval.float_val = old.float_val + local_val;
        // Try to atomically replace old value with new value
    } while (atomic_cmpxchg((__global unsigned int *)global_val,
                            old.int_val, newval.int_val) != old.int_val);
}

__kernel void histogram(
        __global const float fod[$nx][$ny][$nz][$n_coefficients],
        __global const float4 fod_inverse_affine[4],
        __global const float seed_fod[$nnz][$n_coefficients],
        __global const float4 seed_fod_voxels[$nnz],
        __global const float4 seed_fod_affine[4],
        __global uint2 randoms[$n_seeds],
        float dt,
		float save_at,
		float gamma,
		float noise_variance,
		uint seeds_per_thread,
        __global float hist[$nx][$ny][$nz][$n_coefficients])
{
    uint gid = get_global_id(0);
	uint2 state = randoms[gid];
	float4 local_fod_inverse_affine[4] = {fod_inverse_affine[0], fod_inverse_affine[1], fod_inverse_affine[2], fod_inverse_affine[3]};
	float4 local_seed_fod_affine[4] = {seed_fod_affine[0], seed_fod_affine[1], seed_fod_affine[2], seed_fod_affine[3]};

	for (size_t j = 0; j < seeds_per_thread; j++) {
		// Generate the seed.
		float4 location;
		float4 orientation;
		seed_from_fod(seed_fod, seed_fod_voxels, local_seed_fod_affine, &state, &location, &orientation);

		float ylm[$n_coefficients];
		float ylm_dt[$n_coefficients];
		float ylm_dp[$n_coefficients]; 
		size_t n = 1;
		float time = 0;
		uint3 previous_index = {0, 0, 0};
		uint3 index = {0, 0, 0};
		float coefficients[$n_coefficients] = {0};
		while (n < $n_steps) {

			// Go back to voxel space.
			float3 voxel = to_voxel(local_fod_inverse_affine, location);
			previous_index = index;
			index = to_index(voxel);

			// Compute the coefficients of the Dirac associated with the current
			// orientation an add it to the histogram.
			float2 angles = cart2sph(orientation);
			ishtmtx(angles.x, angles.y, ylm, ylm_dp, ylm_dt);

			// Add the current orientation to the histogram.
			if (previous_index.x == index.x && previous_index.y == index.y && previous_index.z == index.z) {
				for (size_t i = 0; i < $n_coefficients; i++) {
					coefficients[i] += ylm[i];
				}
			}
			else {
				for (size_t i = 0; i < $n_coefficients; i++) {
					atomic_add_global_float(hist[previous_index.x][previous_index.y][previous_index.z] + i, coefficients[i]);
					coefficients[i] = ylm[i];
				}
			}

			// Check if we are still in the image and have an fODF.
			if (!in_image(voxel, $nx, $ny, $nz)) {
				break;
			}
			if (fod[index.x][index.y][index.z][0] <= 0.0f) {
				break;
			}

			// Evaluate the value of the fODF and its derivatives.	
			float fod_value = 0.0f;
			float fod_colatitude_value = 0.0f;
			float fod_azimuth_value = 0.0f;
			for (size_t i = 0; i < $n_coefficients; i++) {
				fod_value += fod[index.x][index.y][index.z][i] * ylm[i];
				fod_colatitude_value += fod[index.x][index.y][index.z][i] * ylm_dt[i];
				fod_azimuth_value += fod[index.x][index.y][index.z][i] * ylm_dp[i];
			}
			float d = dsoftmax(fod_value, 100.0f);
			fod_value = softmax(fod_value, 100.0f);
			fod_colatitude_value *= d;
			fod_azimuth_value *= d;

			float st, ct, sp, cp;
			sp = sincos(angles.x, &cp);
			st = sincos(angles.y, &ct);

			// Define the tangent plane. There is no sin(theta) in ep
			// because it cancels with the 1/sin(theta) of the derivative.
			float4 et = {ct * cp, ct * sp, -st, 0.0f};
			float4 ep = {-sp, cp, 0.0f, 0.0f};
			
			// No 1/sin(theta) factor, see comment above.
			float4 drift = (fod_colatitude_value * et + fod_azimuth_value * ep) / fod_value;
			float4 noise = randn(&state) * et + randn(&state) * ep;

			float4 tangent = (gamma * dt) * drift + sqrt(noise_variance * gamma * dt) * noise;
			orientation = exps2(orientation, tangent, 1.0f);

			// Move the point forwared and add it to the streamline.
			location += dt * orientation;

			// Move time forward and record point if necessary.
			time += dt;
			if (time >= save_at) {
				time -= save_at;
				n++;
			}
		}
		if (previous_index.x == index.x && previous_index.y == index.y && previous_index.z == index.z) {
			for (size_t i = 0; i < $n_coefficients; i++) {
				atomic_add_global_float(hist[previous_index.x][previous_index.y][previous_index.z] + i, coefficients[i]);
			}
		}
	}
	randoms[gid] = state;
}
