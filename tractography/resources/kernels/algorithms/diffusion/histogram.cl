#ifndef __DIFFUSION_HISTOGRAM__
#define __DIFFUSION_HISTOGRAM__

#include "utils/core.cl"
#include "utils/spharm.cl"
#define $model
#include "models/select.cl"
#include "algorithms/diffusion/core.cl"

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
        __global float hist[$nx][$ny][$nz][HISTOGRAM_N_COEFFICIENTS])
{
    uint gid = get_global_id(0);
	if (gid >= $n_seeds) return;

	uint4 dims = {$nx, $ny, $nz, $n_coefficients};

	uint2 state = randoms[gid];
	float4 local_fod_inverse_affine[4] = {fod_inverse_affine[0], fod_inverse_affine[1], fod_inverse_affine[2], fod_inverse_affine[3]};
	float4 local_seed_fod_affine[4] = {seed_fod_affine[0], seed_fod_affine[1], seed_fod_affine[2], seed_fod_affine[3]};

	for (size_t j = 0; j < seeds_per_thread; j++) {

		// Generate the seed.
		float4 location;
		float4 orientation;
		seed_from_fod(seed_fod, seed_fod_voxels, local_seed_fod_affine, &state, &location, &orientation);

		float ylm[HISTOGRAM_N_COEFFICIENTS];
		float ylm_dt[HISTOGRAM_N_COEFFICIENTS];
		float ylm_dp[HISTOGRAM_N_COEFFICIENTS]; 
		size_t n = 1;
		float time = 0;
		uint3 previous_index = {0, 0, 0};
		uint3 index = {0, 0, 0};
		float coefficients[HISTOGRAM_N_COEFFICIENTS] = {0};
		while (n < $n_steps) {

			// Go back to voxel space.
			float3 voxel = to_voxel(local_fod_inverse_affine, location);
			previous_index = index;
			index = to_index(voxel);

			// Compute the coefficients of the Dirac associated with the current
			// orientation an add it to the histogram.
			float2 angles = cart2sph(orientation);
			evaluate_sh(angles.x, angles.y, ylm, ylm_dp, ylm_dt);

			// Add the current orientation to the histogram.
			if (previous_index.x == index.x && previous_index.y == index.y && previous_index.z == index.z) {
				for (size_t i = 0; i < HISTOGRAM_N_COEFFICIENTS; i++) {
					coefficients[i] += ylm[i];
				}
			}
			else {
				for (size_t i = 0; i < HISTOGRAM_N_COEFFICIENTS; i++) {
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

			// Update the orientation.
		    model_value_t evaluated_model = evaluate_model(fod, dims, voxel, orientation);
		    orientation = update_orientation(evaluated_model, orientation, &state, dt, gamma, noise_variance);

			// Move the point forward and add it to the streamline.
			location += dt * orientation;

			// Move time forward and record point if necessary.
			time += dt;
			if (time >= save_at) {
				time -= save_at;
				n++;
			}
		}
		if (previous_index.x == index.x && previous_index.y == index.y && previous_index.z == index.z) {
			for (size_t i = 0; i < HISTOGRAM_N_COEFFICIENTS; i++) {
				atomic_add_global_float(hist[previous_index.x][previous_index.y][previous_index.z] + i, coefficients[i]);
			}
		}
	}
	randoms[gid] = state;
}

#endif