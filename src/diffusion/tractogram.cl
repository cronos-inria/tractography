#ifndef __DIFFUSION_TRACTOGRAM__
#define __DIFFUSION_TRACTOGRAM__

#define $model
#include "models/select.cl"
#include "diffusion/core.cl"

__kernel void tractogram(
        __global const float fod[$nx][$ny][$nz][$n_coefficients],
		__global const uint4 *fod_dims,
        __global const float4 affine[4],
        __global const float4 seeds[$n_streamlines][2],
        __global uint2 randoms[$n_streamlines],
        float dt,
		float save_at,
		float gamma,
		float noise_variance,
        __global float4 streamlines[$n_streamlines][$n_steps],
        __global uint lengths[$n_streamlines])
{
	// Get the global ID, with each ID corresponding to a
	// streamline to generate. On some architectures, more threads
	// can be started than requested because of padding. We make
	// sure those threads do nothing.
    uint gid = get_global_id(0);
	if (gid >= $n_streamlines) return;

	// Copy the random state and the model dimensions to local memory.
	uint2 state = randoms[gid];
	uint4 dims = *fod_dims;

	float4 iaffine[4] = {affine[0], affine[1], affine[2], affine[3]};

	// Initialize the first streamline point and orientation with the seed.
	float4 point = seeds[gid][0];
	float4 orientation = seeds[gid][1];

	float ylm[$n_coefficients];
	float ylm_dt[$n_coefficients];
	float ylm_dp[$n_coefficients]; 
	streamlines[gid][0] = point;
	size_t n = 1;
	float time = 0;
	while (n < $n_steps) {

		// Go back to voxel space.
		float3 voxel = to_voxel(iaffine, point);
		if (!in_image(voxel, $nx, $ny, $nz)) {
			break;
		}
		uint3 index = to_index(voxel);

		// Check if we still have an fODF.
        if (fod[index.x][index.y][index.z][0] <= 0.0f) {
			break;
        }

		// Evaluate the value of the fODF and its derivatives.	
		model_value_t evaluated_model = evaluate_model(fod, dims, voxel, orientation);
		orientation = update_orientation(evaluated_model, orientation, &state, dt, gamma, noise_variance);

		// Move the point forward and add it to the streamline.
		point += dt * orientation;
		
		// Move time forward and record point if necessary.
		time += dt;
		if (time >= save_at) {
			time -= save_at;
			streamlines[gid][n] = point - time * orientation;
			n++;
		}

	}
	lengths[gid] = n;
	randoms[gid] = state;
}

#endif