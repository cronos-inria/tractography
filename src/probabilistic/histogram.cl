inline void atomic_add_global_float(__global float *global_val, float local_val)
{
    union {
        unsigned int int_val;
        float float_val;
    } old, newval;

    do {
        // Read old value
        old.float_val = *global_val;
        // Compute new value
        newval.float_val = old.float_val + local_val;
        // Try to atomically replace old value with new value
    } while (atomic_cmpxchg((__global unsigned int *)global_val,
                            old.int_val, newval.int_val) != old.int_val);
}

float4 pick_orientation(
		__global const float fod_values[$nx][$ny][$nz][$n_directions],
		__global const float4 directions[$n_directions],
		float4 orientation,
		uint3 index,
		float rand,
		float max_angle)
{

	// Find the valid orientations.
	float sum = 0.0f;
	for (size_t i = 0; i < $n_directions; i++) {
		sum += fod_values[index.x][index.y][index.z][i] * (dot(directions[i], orientation) > max_angle);
	}
	if (sum <= 0.0f) {
		return (float4) 0; // Nowhere to go.
	}

	// Pick a random direction according to the shape of the FOD.
	float cs = 0;
	for (size_t i = 0; i < $n_directions; i++) {
		cs += fod_values[index.x][index.y][index.z][i] * (dot(directions[i], orientation) > max_angle);
		if (cs > rand * sum) {
			return directions[i];
		}
	}

	return (float4) 0; // Should never happen.
}

__kernel void histogram(
        __global const float fod_values[$nx][$ny][$nz][$n_directions],
        __global const float4 fod_inverse_affine[4],
        __global const float4 directions[$n_directions],
        __global const float seed_fod[$nnz][$n_coefficients],
        __global const float4 seed_fod_voxels[$nnz],
        __global const float4 seed_fod_affine[4],
        __global uint2 randoms[$n_seeds],
        float dt,
		float save_at,
		float max_angle,
		uint seeds_per_thread,
        __global float hist[$nx][$ny][$nz][$n_coefficients])
{
    uint gid = get_global_id(0);
	uint2 state = randoms[gid];
	float4 local_fod_inverse_affine[4] = {
		fod_inverse_affine[0],
		fod_inverse_affine[1],
		fod_inverse_affine[2],
		fod_inverse_affine[3]
	};
	float4 local_seed_fod_affine[4] = {
		seed_fod_affine[0],
		seed_fod_affine[1],
		seed_fod_affine[2],
		seed_fod_affine[3]
	};

	for (size_t j = 0; j < seeds_per_thread; j++) {

		// Generate the seed.
		float4 location;
		float4 orientation;
		seed_from_fod(seed_fod, seed_fod_voxels, local_seed_fod_affine, &state, &location, &orientation);

		float ylm[$n_coefficients];
		float ylm_dt[$n_coefficients];
		float ylm_dp[$n_coefficients]; 
		float coefficients[$n_coefficients] = {0};
		uint3 previous_index = {0, 0, 0};
		uint3 index = {0, 0, 0};

		float time = 0;
		size_t n = 1;
		while (n < $n_steps) {

			// Go back to voxel space.
			float3 voxel = to_voxel(local_fod_inverse_affine, location);
			previous_index = index;
			index = to_index(voxel);

			// Compute the coefficients of the Dirac associated with the current
			// orientation an add it to the histogram.
			float2 angles = cart2sph(orientation);
			ishtmtx(angles.x, angles.y, ylm, ylm_dp, ylm_dt);

			// Add the current orientation to the histogram.
			if (previous_index.x == index.x && previous_index.y == index.y && previous_index.z == index.z) {
				for (size_t i = 0; i < $n_coefficients; i++) {
					coefficients[i] += ylm[i];
				}
			}
			else {
				for (size_t i = 0; i < $n_coefficients; i++) {
					atomic_add_global_float(hist[previous_index.x][previous_index.y][previous_index.z] + i, coefficients[i]);
					coefficients[i] = ylm[i];
				}
			}

			// Check if we are still in the image and have an fODF.
			if (!in_image(voxel, $nx, $ny, $nz)) {
				break;
			}

			// Pick the next direction. If the orientation is 0, there is nowhere to go.
			float rand = randu(&state);
			orientation = pick_orientation(fod_values, directions, orientation, index, rand, max_angle);
			if (length(orientation) < 0.5f) {
				break;
			}

			// Move the point forwared and add it to the streamline.
			location += dt * orientation;
			
			// Move time forward and record point if necessary.
			time += dt;
			if (time >= save_at) {
				time -= save_at;
				n++;
			}
		}
		if (previous_index.x == index.x && previous_index.y == index.y && previous_index.z == index.z) {
			for (size_t i = 0; i < $n_coefficients; i++) {
				atomic_add_global_float(hist[previous_index.x][previous_index.y][previous_index.z] + i, coefficients[i]);
			}
		}
	}
	randoms[gid] = state;
}
