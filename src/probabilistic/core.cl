/**
 * PICK_ORIENTATION
 * Selects a next orientation probabilistically, weighted by the FOD values,
 * and constrained by a maximum angular difference.
 *
 * fod: Pointer to the flat FOD array with a shape specified by dims.
 * orientations: Pointer to the global array of discrete direction vectors. It
 * 	must contains dims.w orientations.
 * current_orientation: The orientation vector to align against.
 * index: The 3D coordinate (x, y, z) of the voxel currently being processed.
 * dims: The dimensions {nx, ny, nz, n_orientations} of the flattened FOD array.
 * rand: A pseudo-random float [0.0, 1.0) used for selection.
 * max_cos_angle: The *minimum* dot product (cosine) value required for alignment.
 * @return float4 The probabilistically selected orientation vector.
 */
inline float4 pick_orientation(
    __global const float *restrict fod,
    __global const float4 *restrict orientations,
    float4 current_orientation,
    uint4 dims,
    uint3 index,
    float rand,
    float max_cos_angle)
{
    // Local copies of dimensions for clarity and potential compiler optimization.
    const uint ny = dims.y;
    const uint nz = dims.z;
    const uint n_directions = dims.w;

    // Calculate the base index for the current voxel in the flattened FOD array.
    const size_t base_index = (size_t)index.x * (ny * nz * n_directions) + 
                              (size_t)index.y * (nz * n_directions) + 
                              (size_t)index.z * n_directions;

    // Calculate total sum of valid weights.
    float total_sum = 0.0f;
    for (uint i = 0; i < n_directions; i++) {
        const float fod_value = fod[base_index + i];
        
        // Check the angular constraint.
        if (dot(orientations[i], current_orientation) >= max_cos_angle) {
            total_sum += fod_value;
        }
    }

    if (total_sum <= 0.0f) {
        return (float4)(0.0f); // Nowhere to go (e.g., in a background area).
    }

    // The probabilistic check must be done in a second pass because we need 'total_sum'.
    float cumulative_sum = 0.0f;
    float target = rand * total_sum;
    for (uint i = 0; i < n_directions; i++) {
        const float fod_value = fod[base_index + i];

        if (dot(orientations[i], current_orientation) >= max_cos_angle) {
            // Only add valid values to the cumulative sum.
            cumulative_sum += fod_value;
            
            // If the cumulative sum exceeds the target, we found the direction.
            if (cumulative_sum > target) {
                return orientations[i];
            }
        }
    }

    // Safety return: this should only be reached due to float precision errors, 
    // but it ensures a non-crash path. We return the last valid direction.
    // Given the logic, the last direction should have been returned in the loop.
    return (float4)(0.0f); 
}
