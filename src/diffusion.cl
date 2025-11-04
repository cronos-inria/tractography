#include "utils/spharm.cl"

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
        __global const float fod[$nx][$ny][$nz][$n_coefficients],
        __global const float4 affine[4],
        __global const float4 seeds[$n_streamlines][2],
        __global uint2 randoms[$n_streamlines],
		__global const float4 directions[162],
        float dt,
		float save_at,
		float gamma,
        __global unsigned int hist[$nx][$ny][$nz][162])
{
    uint gid = get_global_id(0);
	uint2 state = randoms[gid];

	float4 iaffine[4] = {affine[0], affine[1], affine[2], affine[3]};

	// Initialize the first streamline point and orientation with the seed.
	float4 point = seeds[gid][0];
	float4 orientation = seeds[gid][1];

	float ylm[$n_coefficients];
	float ylm_dt[$n_coefficients];
	float ylm_dp[$n_coefficients]; 
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

		// Add the current orientation to the histogram.
		atomic_inc(hist[index.x][index.y][index.z] + closest_direction_index(directions, orientation));

		// Evaluate the value of the fODF and its derivatives.	
		float2 angles = cart2sph(orientation);
		ishtmtx(angles.x, angles.y, ylm, ylm_dp, ylm_dt);
		float fod_value = 0.0f;
		float fod_colatitude_value = 0.0f;
		float fod_azimuth_value = 0.0f;
		for (size_t i = 0; i < $n_coefficients; i++) {
			fod_value += fod[index.x][index.y][index.z][i] * ylm[i];
			fod_colatitude_value += fod[index.x][index.y][index.z][i] * ylm_dt[i];
			fod_azimuth_value += fod[index.x][index.y][index.z][i] * ylm_dp[i];
		}
		float d = dsoftmax(fod_value, 100.0f);
		fod_value = softmax(fod_value, 100.0f);
		fod_colatitude_value *= d;
		fod_azimuth_value *= d;

		float st, ct, sp, cp;
		sp = sincos(angles.x, &cp);
		st = sincos(angles.y, &ct);

		float4 et = {ct * cp, ct * sp, -st, 0.0f};
		float4 ep = {-sp, cp, 0.0f, 0.0f};
		
		float4 drift = (fod_colatitude_value * et + fod_azimuth_value * ep) / fod_value;
        float4 noise = randn(&state) * et + randn(&state) * ep;

		float4 tangent = (gamma * dt) * drift + sqrt(2.0f * gamma * dt) * noise;
		orientation = exps2(orientation, tangent, 1.0f);

		// Move the point forwared and add it to the streamline.
		point += dt * orientation;

		// Move time forward and record point if necessary.
		time += dt;
		if (time >= save_at) {
			time -= save_at;
			n++;
		}
	}
	randoms[gid] = state;
}


__kernel void tractography(
        __global const float fod[$nx][$ny][$nz][$n_coefficients],
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
    uint gid = get_global_id(0);
	uint2 state = randoms[gid];

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
		float2 angles = cart2sph(orientation);
		ishtmtx(angles.x, angles.y, ylm, ylm_dp, ylm_dt);
		float fod_value = 0.0f;
		float fod_colatitude_value = 0.0f;
		float fod_azimuth_value = 0.0f;
		for (size_t i = 0; i < $n_coefficients; i++) {
			fod_value += fod[index.x][index.y][index.z][i] * ylm[i];
			fod_colatitude_value += fod[index.x][index.y][index.z][i] * ylm_dt[i];
			fod_azimuth_value += fod[index.x][index.y][index.z][i] * ylm_dp[i];
		}
		float d = dsoftmax(fod_value, 100.0f);
		fod_value = softmax(fod_value, 100.0f);
		fod_colatitude_value *= d;
		fod_azimuth_value *= d;

		float st, ct, sp, cp;
		sp = sincos(angles.x, &cp);
		st = sincos(angles.y, &ct);

		float4 et = {ct * cp, ct * sp, -st, 0.0f};
		float4 ep = {-sp, cp, 0.0f, 0.0f};
		
		float4 drift = (fod_colatitude_value * et + fod_azimuth_value * ep) / fod_value;
        float4 noise = randn(&state) * et + randn(&state) * ep;

		float4 tangent = (gamma * dt) * drift + sqrt(noise_variance * gamma * dt) * noise;
		orientation = exps2(orientation, tangent, 1.0f);

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
	randoms[gid] = state;
}

