#include "core.cl"

size_t pick_orientation(
		__global const float4 vertices[$n_directions],
		float4 orientation)
{
	float best_angle = -2;
	size_t best_index = -1;
	for (size_t i = 0; i < $n_directions; i++) {
		float angle = dot(vertices[i], orientation);
		if (angle > best_angle) {
			best_angle = angle;
			best_index = i;
		}
	}
	return best_index;
}


void sample_fod(
        __global const float fod[$nx][$ny][$nz][$n_coefficients],
		float3 voxel,
		float coefficients[$n_coefficients])
{
	if (!in_image(voxel, $nx, $ny, $nz)) {
		for (size_t i = 0; i < $n_coefficients; i++) {
			coefficients[i] = 0;
		}
	}
	else {
		uint3 index = to_index(voxel);
		for (size_t i = 0; i < $n_coefficients; i++) {
			coefficients[i] = fod[index.x][index.y][index.z][i];
		}
	}

}

__kernel void tractography(
        __global const float fod[$nx][$ny][$nz][$n_coefficients],
        __global const float4 global_iaffine[4],
        __global const float4 vertices[$n_directions],
		__global const float matrix[$n_directions][$n_coefficients],
		__global const float dmatrix[2][$n_directions][$n_coefficients],
        __global const float4 seeds[$n_streamlines][2],
        __global float4 streamlines[$n_streamlines][$n_steps],
		__global uint lengths[$n_streamlines],
		float step_size,
		float acceleration_factor)
{
    uint gid = get_global_id(0);

	// Copy the affine to local memory.
	float4 iaffine[4] = {global_iaffine[0], global_iaffine[1], global_iaffine[2], global_iaffine[3]};

    // Initialize the first streamline point with the seed.
	float4 location = seeds[gid][0];
    float4 orientation = seeds[gid][1];
	float colatitude = acos(orientation.z);
	float azimuth = atan2(orientation.y, orientation.x);
	if (azimuth < 0) {
		azimuth = 2 * PI + azimuth;
	}

	float coefficients[$n_coefficients];
    streamlines[gid][0] = location;
	size_t n;
    for (n = 1; n < $n_steps; n++) {

        // Go back to voxel space.
        float3 voxel = to_voxel(iaffine, location);

		// Check if we still have an FOD.
		sample_fod(fod, voxel, coefficients);
        if (coefficients[0] <= 0.0f) {
			break;
        }

		// Update the orientation displacement.
		size_t index = pick_orientation(vertices, orientation);
		float fod_value = 0.0f;
		float fod_colatitude_value = 0.0f;
		float fod_azimuth_value = 0.0f;
		for (size_t i = 0; i < $n_coefficients; i++) {
			fod_value += coefficients[i] * matrix[index][i];
			fod_colatitude_value += coefficients[i] * dmatrix[0][index][i];
			fod_azimuth_value += coefficients[i] * dmatrix[1][index][i];
		}
		fod_value = fmax(fod_value,  0.001f);

		// Displace angles and fix wrapping of the angles.
		float sc = fmax(sin(colatitude), 0.001f);
		azimuth += fod_azimuth_value / fod_value / sc * step_size * acceleration_factor;
		colatitude += fod_colatitude_value / fod_value * step_size * acceleration_factor;
		wrap(&azimuth, &colatitude);

		// Move foward.
		orientation = sph2cart(azimuth, colatitude);
		location += orientation * step_size;
		streamlines[gid][n] = location;
	}
	lengths[gid] = n;
}
