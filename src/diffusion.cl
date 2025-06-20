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

float4 exps2(float4 p, float4 x, float t) {
    float n = length(x);
    if (n == 0)
        return p;

	float c;
    float s = sincos(t * n, &c);
    return c * p + x * (s / n);
}

__kernel void tractography(
        __global const float fod[$nx][$ny][$nz][$n_directions],
        __global const float4 affine[4],
        __global const float4 vertices[$n_directions],
		__global const float matrix[$n_directions][$n_coefficients],
		__global const float dmatrix[2][$n_directions][$n_coefficients],
        __global const float4 seeds[$n_streamlines][2],
        __global uint2 randoms[$n_streamlines],
        float dt,
		float gamma,
        __global float4 streamlines[$n_streamlines][$n_steps],
        __global uint lengths[$n_streamlines])
{
    uint gid = get_global_id(0);
	uint lid = get_local_id(0);
	uint2* state = randoms + gid;

	float4 iaffine[4] = {affine[0], affine[1], affine[2], affine[3]};

	local float4 nvertices[$n_directions];
	if (lid < $n_directions)
		nvertices[lid] = vertices[lid];
	barrier(CLK_LOCAL_MEM_FENCE);

	// Initialize the first streamline point and orientation with the seed.
	float4 point = seeds[gid][0];
	float4 orientation = seeds[gid][1];

	float coefficients[$n_coefficients];
	streamlines[gid][0] = point;
	size_t n;
	for (n = 1; n < $n_steps; n++ ) {

		// Go back to voxel space.
		float3 voxel = to_voxel(iaffine, point);

		// Check if we are in the image, stop if not.
		if (!in_image(voxel, $nx, $ny, $nz)) {
			break;
		}

		// Check if we still have an FOD.
		sample_fod(fod, voxel, coefficients);
        if (coefficients[0] <= 0.0f) {
			break;
        }

		// Pick the next direction.	
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

		float2 angles = cart2sph(orientation);
		float st, ct, sp, cp;
		st = sincos(angles.y, &ct);
		sp = sincos(angles.x, &cp);

		float4 et = {ct * cp, ct * sp, -st, 0.0f};
		float4 ep = {-sp, cp, 0.0f, 0.0f};
		
		float4 drift = (fod_colatitude_value * et + fod_azimuth_value * ep) / fod_value;
        float4 noise = randn(state) * et + randn(state) * ep;

		float4 tangent = (gamma * dt) * drift + sqrt(gamma * dt) * noise;
		float scaling = 1.0f;
		if (length(tangent) > 0.017f) {
			scaling = 0.017f / length(tangent);
		}
		orientation = exps2(orientation, tangent, scaling);

		// Move the point forwared and add it to the streamline.
		point += dt * scaling * orientation;
		streamlines[gid][n] = point;
	}
	lengths[gid] = n;
}

