#include "utils/core.cl"

float4 pick_orientation(
		__global const float fod[$nx][$ny][$nz][$n_directions],
		__global const float4 vertices[$n_directions],
		float4 orientation,
		float3 voxel,
		float max_angle)
{

	uint3 index = to_index(voxel);

	// Pick the valid direction with max value.
	float4 current_orientation = 0;
	float current_max = 0.0;
	for (size_t i = 0; i < $n_directions; i++) {
		if (dot(vertices[i], orientation) < max_angle) {
			continue;
		}

	    float cs = fod[index.x][index.y][index.z][i];
	    if (cs > current_max) {
			current_max = cs;
			current_orientation = vertices[i];
	    }
	}

	return current_orientation;
}

__kernel void tractography(
        __global const float fod[$nx][$ny][$nz][$n_directions],
        __global const float4 affine[4],
        __global const float4 vertices[$n_directions],
        __global const float4 seeds[$n_streamlines][2],
        float dt,
        float max_angle,
        __global float4 streamlines[$n_streamlines][$n_steps],
        __global uint lengths[$n_streamlines])
{
    uint gid = get_global_id(0);

	// Copy the affine to local memory.
	float4 iaffine[4] = {affine[0], affine[1], affine[2], affine[3]};

    // Initialize the first streamline point and orientation with the seed.
	float4 point = seeds[gid][0];
    float4 orientation = seeds[gid][1];

    streamlines[gid][0] = point;
	size_t n;
    for (n = 1; n < $n_steps; n++ ) {

        // Go back to voxel space.
        float3 voxel = to_voxel(iaffine, point);

		// Check if we are in the image, stop if not.
		if (!in_image(voxel, $nx, $ny, $nz)) {
			break;
		}

		// Pick the next direction.	
		orientation = pick_orientation(fod, vertices, orientation, voxel, max_angle);

        // If the orientation is 0, there is nowhere to go.
        if (length(orientation) < 0.5) {
            break;
        }

        // Move the point forwared and add it to the streamline.
		point += dt * orientation;
        streamlines[gid][n] = point;
    }
	lengths[gid] = n;
}
