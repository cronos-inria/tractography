#ifndef __PROBABILISTIC_CONNECTOME___
#define __PROBABILISTIC_CONNECTOME___

#include "utils/core.cl"
#include "algorithms/probabilistic/core.cl"

__kernel void connectome(
        __global const float fod_values[$nx][$ny][$nz][$n_directions],
        __global const float4 fod_inverse_affine[4],
        __global const float4 directions[$n_directions],
        __global const float seed_fod[$nnz][$n_coefficients],
        __global const float4 seed_fod_voxels[$nnz],
        __global const float4 seed_fod_affine[4],
        __global uint2 randoms[$n_seeds],
        __global const float4 vertices[$n_vertices],
        __global const int vertex_labels[$n_vertices],
        float dt,
        float save_at,
        uint min_n_steps,
        float max_angle,
        float distance_upper_bound,
        uint seeds_per_thread,
        __global uint matrix[$n_labels][$n_labels])
{
    uint gid = get_global_id(0);
    if (gid >= $n_seeds) return;

    uint4 dims = {$nx, $ny, $nz, $n_directions};

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

        // Record the starting location.
        float4 start_location = location;

        float time = 0;
        size_t n = 1;
        while (n < $n_steps) {

            // Go back to voxel space.
            float3 voxel = to_voxel(local_fod_inverse_affine, location);

            // Check if we are still in the image.
            if (!in_image(voxel, $nx, $ny, $nz)) {
                break;
            }
            uint3 index = to_index(voxel);

            // Pick the next direction.
            float rand = randu(&state);
            orientation = pick_orientation(fod_values, directions, orientation, dims, index, rand, max_angle);
            if (length(orientation) < 0.5) {
                break;
            }

            // Move the point forward.
            location += dt * orientation;

            // Move time forward.
            time += dt;
            if (time >= save_at) {
                time -= save_at;
                n++;
            }
        }

        // Only record if the streamline actually propagated.
        if (n < min_n_steps) continue;

        // Find the nearest labelled vertex to the start and end points.
        int start_label = nearest_vertex_label(vertices, vertex_labels, $n_vertices, start_location, distance_upper_bound);
        int end_label = nearest_vertex_label(vertices, vertex_labels, $n_vertices, location, distance_upper_bound);

        // Only count if both endpoints are near labelled vertices.
        if (start_label < 0 || end_label < 0) continue;

        // Atomically increment the matrix entry.
        atomic_inc(&matrix[start_label][end_label]);
    }
    randoms[gid] = state;
}

#endif