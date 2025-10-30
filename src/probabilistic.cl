#include "core.cl"

float4 pick_orientation(
		__global const float fod[$nx][$ny][$nz][$n_directions],
		__global const float4 vertices[$n_directions],
		float4 orientation,
		uint3 index,
		float rand,
		float max_angle)
{

	// Find the valid orientations.
	float sum = 0;
	for (size_t i = 0; i < $n_directions; i++) {
		sum += fod[index.x][index.y][index.z][i] * (dot(vertices[i], orientation) > max_angle);
	}

	// Pick a random direction according to the shape of the FOD.
	float cs = 0;
	for (size_t i = 0; i < $n_directions; i++) {
		cs += fod[index.x][index.y][index.z][i] * (dot(vertices[i], orientation) > max_angle);
		if (cs > rand * sum) {
			return vertices[i];
		}
	}

	return (float4) 0;
}

size_t closest_direction_index(
		__global const float4 directions[162],
		float4 orientation)
{

	// Pick the valid direction with max value.
	size_t best_index = 0;
	float current_max = -2.0f;
	for (size_t i = 0; i < 162; i++) {
		float value = dot(directions[i], orientation);
		if (value > current_max) {
			current_max = value;
			best_index = i;
		}
	}

	return best_index;
}

__kernel void histogram(
        __global const float fod[$nx][$ny][$nz][$n_directions],
        __global const float4 affine[4],
        __global const float4 directions[$n_directions],
        __global const float4 seeds[$n_streamlines][2],
        __global uint2 randoms[$n_streamlines],
        float dt,
        float max_angle,
        __global unsigned int hist[$nx][$ny][$nz][162])
{
    uint gid = get_global_id(0);
	uint2 state = randoms[gid];

	float4 iaffine[4] = {affine[0], affine[1], affine[2], affine[3]};

	// Initialize the first streamline point and orientation with the seed.
	float4 point = seeds[gid][0];
	float4 orientation = seeds[gid][1];

	size_t n;
	for (n = 1; n < $n_steps; n++ ) {

		// Go back to voxel space.
		float3 voxel = to_voxel(iaffine, point);
		if (!in_image(voxel, $nx, $ny, $nz)) {
			break;
		}
		uint3 index = to_index(voxel);

		// Add the current orientation to the histogram.
		atomic_inc(hist[index.x][index.y][index.z] + closest_direction_index(directions, orientation));

		// Pick the next direction.	
		float rand = randu(&state);
		orientation = pick_orientation(fod, directions, orientation, index, rand, max_angle);

		// If the orientation is 0, there is nowhere to go.
		if (length(orientation) < 0.5) {
			break;
		}

		// Move the point forwared and add it to the streamline.
		point += dt * orientation;
	}
	randoms[gid] = state;
}

__kernel void tractography(
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
		orientation = pick_orientation(fod, vertices, orientation, index, rand, max_angle);

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
