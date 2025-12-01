#include "utils/spharm.cl"
#include "diffusion/core.cl"

__kernel void tractography(
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

		// Update the orientation.
		float2 angles = cart2sph(orientation);
		ishtmtx(angles.x, angles.y, ylm, ylm_dp, ylm_dt);
		orientation = update_orientation(fod, ylm, ylm_dp, ylm_dt, index,
			dims, orientation, NULL, dt, gamma, 0.0f);

		// Move the point forwared and add it to the streamline.
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

