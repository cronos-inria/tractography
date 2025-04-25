#include "core.cl"

float4 pick_orientation(
		__global const float fod[$nx][$ny][$nz][$n_directions],
		__global const float4 vertices[$n_directions],
		float4 orientation,
		float3 voxel,
		float rand,
		float max_angle)
{

	uint3 index = to_index(voxel);

	// Find the valid orientations.
	uint valid[$n_directions];
	for (size_t i = 0; i < $n_directions; i++) {
		if (dot(vertices[i], orientation) > max_angle) {
			valid[i] = 1;
		}
		else {
			valid[i] = 0;
		}
	}

	// To pick, we need the sum as a normalization factor.
	float sum = 0;
	for (size_t i = 0; i < $n_directions; i++) {
		if (valid[i] == 1) { 
			sum += fod[index.x][index.y][index.z][i];
		}
	}

	// Pick a random direction according to the shape of the FOD.
	float cs = 0;
	for (size_t i = 0; i < $n_directions; i++) {
	    if (valid[i] == 1) {
	        cs += fod[index.x][index.y][index.z][i];
	        if (cs > rand * sum) {
				return vertices[i];
	        }
	    }
	}

	return 0;
}

__kernel void tractography(
        __global const float fod[$nx][$ny][$nz][$n_directions],
        __global const float4 affine[4],
        __global const float4 vertices[$n_directions],
        __global const float4 seeds[$n_streamlines][2],
        float dt,
        float max_angle,
        __global const float randoms[$n_streamlines][$n_steps],
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
		float rand = randoms[gid][n - 1];
		orientation = pick_orientation(fod, vertices, orientation, voxel, rand, max_angle);

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
