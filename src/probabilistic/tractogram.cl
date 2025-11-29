#include "utils/core.cl"
#include "probabilistic/core.cl"

__kernel void tractogram(
        __global const float fod[$nx][$ny][$nz][$n_directions],
        __global const float4 affine[4],
        __global const float4 vertices[$n_directions],
        __global const float4 seeds[$n_streamlines][2],
        __global uint2 randoms[$n_streamlines],
        float dt,
        float max_angle,
        __global float4 streamlines[$n_streamlines][$n_steps],
        __global uint lengths[$n_streamlines])
{
    uint gid = get_global_id(0);
	if (gid >= $n_streamlines) return;

	uint4 dims = {$nx, $ny, $nz, $n_directions};
	uint2 state = randoms[gid];

	float4 iaffine[4] = {affine[0], affine[1], affine[2], affine[3]};

	// Initialize the first streamline point and orientation with the seed.
	float4 point = seeds[gid][0];
	float4 orientation = seeds[gid][1];

	streamlines[gid][0] = point;
	size_t n;
	for (n = 1; n < $n_steps; n++ ) {

		// Go back to voxel space.
		float3 voxel = to_voxel(iaffine, point);
		if (!in_image(voxel, $nx, $ny, $nz)) {
			break;
		}
		uint3 index = to_index(voxel);

		// Pick the next direction.	
		float rand = randu(&state);
		orientation = pick_orientation(fod, vertices, orientation, dims, index, rand, max_angle);

		// If the orientation is 0, there is nowhere to go.
		if (length(orientation) < 0.5f) {
			break;
		}

		// Move the point forwared and add it to the streamline.
		point += dt * orientation;
		streamlines[gid][n] = point;
	}
	lengths[gid] = n;
	randoms[gid] = state;
}
