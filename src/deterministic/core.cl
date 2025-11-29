#ifndef __DETERMINISTIC_CORE___
#define __DETERMINISTIC_CORE___

/**
 * PICK_ORIENTATION
 * Finds the orientation from the discrete set 'orientations' that 
 * aligns best with the 'current_orientation' within a maximum angle
 * constraint and has the maximum FOD scalar value at a specific voxel index.
 *
 * fod: Pointer to the flat FOD array with a shape specified by dims.
 * orientations: Pointer to the global array of discrete direction vectors. It
 * 	must contains dims.w orientations.
 * current_orientation: The orientation vector to align against.
 * index: The 3D coordinate (x, y, z) of the voxel currently being processed.
 * dims: The dimensions {nx, ny, nz, n_orientations} of the flattened FOD array.
 * max_cos_angle: The *minimum* dot product (cosine) value required for alignment.
 *
 * Returns the best-fitting orientation vector. Is 0 if no valid orientation can
 * be found.
 */
inline float4 pick_orientation(
    __global const float *restrict fod,
    __global const float4 *restrict orientations,
    float4 current_orientation,
    uint4 dims,
    uint3 index,
    float max_cos_angle)
{
    // Local copies of dimensions for clarity and potential compiler optimization.
    const uint nx = dims.x;
    const uint ny = dims.y;
    const uint nz = dims.z;
    const uint n_orientations = dims.w;

    // Calculate the base index for the current voxel (x, y, z) in the flattened FOD
	// array.
    const size_t base_index = (size_t) index.x * (ny * nz * n_orientations) + 
                              (size_t) index.y * (nz * n_orientations) + 
                              (size_t) index.z * n_orientations;

    float4 best_orientation = (float4)(0.0f);
    float current_max_cs = -1.0f;
    for (uint i = 0; i < n_orientations; i++) {

        // Check orientation constraint.
        if (dot(orientations[i], current_orientation) < max_cos_angle) {
            continue;
        }

        // Check Maximum FOD value.
        float cs = fod[base_index + i];
        if (cs > current_max_cs) {
            current_max_cs = cs;
            best_orientation = orientations[i];
        }
    }

    return best_orientation;
}

#endif
