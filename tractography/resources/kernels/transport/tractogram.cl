#ifndef __TRANSPORT_TRACTOGRAM__
#define __TRANSPORT_TRACTOGRAM__

#include "utils/core.cl"
#define $model
#include "models/select.cl"
#include "diffusion/core.cl"

__kernel void tractogram(
        __global const float fod[$nx][$ny][$nz][$n_coefficients],
        __global const float4 affine[4],
        __global const float4 seeds[$n_streamlines][2],
        float dt,
		float save_at,
		float gamma,
        __global float4 streamlines[$n_streamlines][$n_steps],
        __global uint lengths[$n_streamlines])
{
    uint gid = get_global_id(0);
	if (gid >= $n_streamlines) return;

	uint4 dims = {$nx, $ny, $nz, $n_coefficients};

	float4 iaffine[4] = {affine[0], affine[1], affine[2], affine[3]};

	// Initialize the first streamline point and orientation with the seed.
	float4 point = seeds[gid][0];
	float4 orientation = seeds[gid][1];

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

		// Update the orientation.
		model_value_t evaluated_model = evaluate_model(fod, dims, voxel, orientation);
		orientation = update_orientation(evaluated_model, orientation, 0, dt, gamma, 0.0f);

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
}

#endif