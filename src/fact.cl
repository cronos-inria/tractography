#include "core.cl"

float4 pick_orientation(__global const float4 peaks[$nx][$ny][$nz][$n_peaks], float4 orientation, float3 voxel, float cos_angle) {

	// Pick the new orientation which is most colinear with the previous orientation.
	float4 best_orientation = 0;
	float current_max = -1;
	uint3 index = to_index(voxel);
	for (size_t i = 0; i < $n_peaks; i++) {
		float4 peak = peaks[index.x][index.y][index.z][i];
		float cs = dot(peak, orientation);
	    if (fabs(cs) > current_max && fabs(cs) > cos_angle) {
			current_max = fabs(cs);
			if (cs > 0) {
				best_orientation = peak;
			} else {
				best_orientation = -peak;
			}
	    }
	}

	return best_orientation;
}

void duplicate_points(
		__global float4 streamlines[$n_streamlines][$n_steps],
		size_t n,
		uint gid) {

	float4 point = streamlines[gid][n-1];
	for (size_t i = n; i < $n_steps; i++) {
		streamlines[gid][i] = point;
	}
}

__kernel void tractography(
        __global const float4 peaks[$nx][$ny][$nz][$n_peaks],
        __global const float4 affine[4],
        __global const float4 seeds[$n_streamlines][2],
        float dt,
        float cos_angle,
        __global float4 streamlines[$n_streamlines][$n_steps])
{
    uint gid = get_global_id(0);

	// Copy the affine to local memory.
	float4 iaffine[4] = {affine[0], affine[1], affine[2], affine[3]};

    // Initialize the first streamline point and orientation with the seed.
	float4 point = seeds[gid][0];
    float4 orientation = seeds[gid][1];

    streamlines[gid][0] = point;
    for (size_t n = 1; n < $n_steps; n++ ) {

        // Go back to voxel space.
        float3 voxel = to_voxel(iaffine, point);

		// Check if we are in the image, stop if not.
		if (!in_image(voxel, $nx, $ny, $nz)) {
			duplicate_points(streamlines, n, gid);
            break;
		}

		// Pick the next orientation.	
		orientation = pick_orientation(peaks, orientation, voxel, cos_angle);

        // If the orientation is 0, there is nowhere to go.
        if (length(orientation) < 0.5) {
			duplicate_points(streamlines, n, gid);
            break;
        }

        // Move the point forwared and add it to the streamline.
		point += dt * orientation;
        streamlines[gid][n] = point;
    }
}
