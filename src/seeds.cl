#ifndef __SEEDS__
#define __SEEDS__

#include "utils/spharm.cl"

#define N_DIRECTIONS 600

// Generates points approximately uniformly distributed on the sphere
// uing the Fibonacci lattice.
void fibonacci_sphere(float4 points[N_DIRECTIONS])
{
    const float phi = (sqrt(5.0f) + 1.0f) * 0.5f;
    const float golden_angle = 2.0f * PI / phi;

    for (int i = 0; i < N_DIRECTIONS; i++) {

        float theta = i * golden_angle;
        float z = 1.0f - (2.0f * i / (N_DIRECTIONS - 1));
        float radius = sqrt(1.0f - z * z);
		float x = radius * cos(theta);
		float y = radius * sin(theta);

        points[i] = (float4) {x, y, z, 0.0f};
    }
}

// Returns an orientation sampled from a single FOD.
float4 sample_fod(
		__global const float fod[$n_coefficients],
		uint2* state)
{
	float4 directions[N_DIRECTIONS];
	fibonacci_sphere(directions);

	// Evaluate the FOD over the provided directions.
	float values[N_DIRECTIONS];
	for (size_t n = 0; n < N_DIRECTIONS; n++) {
		values[n] = shval(fod, directions[n]);
	}

	// Find the valid orientations.
	float sum = 0;
	for (size_t i = 0; i < N_DIRECTIONS; i++) {
		sum += values[i];
	}

	// Pick a random direction according to the shape of the FOD.
	float rand = randu(state);
	float cs = 0;
	for (size_t i = 0; i < N_DIRECTIONS; i++) {
		cs += values[i];
		if (cs > rand * sum) {
			return directions[i];
		}
	}

	return (float4) 0;
}

// Returns a seed (location, orientation) sampled from the provided FOD.
// The FOD are in sparse format, meaning an array of (N, NC) where N is the
// number of FOD and NC the number of coefficients. The location in voxel space
// of the FOD is provided by the voxels parameters.
void seed_from_fod(
        __global const float fod[$nnz][$n_coefficients],
        __global const float4 voxels[$nnz],
        const float4 affine[4],
		uint2* state,
		float4* location,
		float4* orientation)
{
	// Choose the index voxel uniformly from non-zero voxels.
	uint voxel_index = randi(state, $nnz);
	float4 voxel = voxels[voxel_index];

	// Generate a random point uniformly in the voxel and importance sample
	// the local FOD.
	voxel += (float4) {randu(state) - 0.5f, randu(state) - 0.5f, randu(state) - 0.5f, 0.0f}; 	
	*orientation = sample_fod(fod[voxel_index], state);

	// Convert to world coordinates.
	*location = apply_affine(affine, voxel);
}

// Returns seeds (location, orientation) sampled from the provided FOD.
// The FOD are in sparse format, meaning an array of (N, NC) where N is the
// number of FOD and NC the number of coefficients. The location in voxel space
// of the FOD is provided by the voxels parameters.
__kernel void seeds_from_fod(
        __global const float fod[$nnz][$n_coefficients],
        __global const float4 voxels[$nnz],
        __global const float4 affine[4],
		__global uint2 states[$n_seeds],
		__global float4 seeds[$nnz][2])
{
	// Copy to local memory.
    uint gid = get_global_id(0);
	uint2 state = states[gid];
	const float4 iaffine[4] = {affine[0], affine[1], affine[2], affine[3]};
	
	// Generate the seed.
	float4 location, orientation;
	seed_from_fod(fod, voxels, iaffine, &state, &location, &orientation);
	
	seeds[gid][0] = location;
	seeds[gid][1] = orientation;
	states[gid] = state;
}

#endif
