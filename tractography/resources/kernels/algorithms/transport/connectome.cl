#ifndef __TRANSPORT_CONNECTOME__
#define __TRANSPORT_CONNECTOME__

#include "utils/core.cl"
#define $model
#include "models/select.cl"
#include "algorithms/diffusion/core.cl"

__kernel void connectome(
        __global const float fod[$nx][$ny][$nz][$n_coefficients],
        __global const float4 fod_inverse_affine[4],
        __global const float seed_fod[$nnz][$n_coefficients],
        __global const float4 seed_fod_voxels[$nnz],
        __global const float4 seed_fod_affine[4],
        __global uint2 randoms[$n_seeds],
        __global const float4 vertices[$n_vertices],
        __global const int vertex_labels[$n_vertices],
        float dt,
        float save_at,
        uint min_n_points,
        float gamma,
        float distance_upper_bound,
        uint seeds_per_thread,
        __global uint matrix[$n_labels][$n_labels])
{
    uint gid = get_global_id(0);
    if (gid >= $n_seeds) return;

    uint4 dims = {$nx, $ny, $nz, $n_coefficients};

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

        float4 location;
        float4 orientation;
        seed_from_fod(
            seed_fod,
            seed_fod_voxels,
            local_seed_fod_affine,
            &state,
            &location,
            &orientation
        );

        float4 start_location = location;

        float ylm[$n_coefficients];
        float ylm_dt[$n_coefficients];
        float ylm_dp[$n_coefficients];
        float time = 0;
        size_t n = 1;
        while (n < $n_steps) {

            float3 voxel = to_voxel(local_fod_inverse_affine, location);
            if (!in_image(voxel, $nx, $ny, $nz)) {
                break;
            }
            uint3 index = to_index(voxel);

            if (fod[index.x][index.y][index.z][0] <= 0.0f) {
                break;
            }

            // Update the orientation.
            model_value_t evaluated_model = evaluate_model(fod, dims, voxel, orientation);
            orientation = update_orientation(evaluated_model, orientation, 0, dt, gamma, 0.0f);
            if (length(orientation) < 0.5f) {
                break;
            }

            location += dt * orientation;

            time += dt;
            if (time >= save_at) {
                time -= save_at;
                n++;
            }
        }

        if (n < min_n_points) continue;

        int start_label = nearest_vertex_label(
            vertices,
            vertex_labels,
            $n_vertices,
            start_location,
            distance_upper_bound
        );
        int end_label = nearest_vertex_label(
            vertices,
            vertex_labels,
            $n_vertices,
            location,
            distance_upper_bound
        );

        if (start_label < 0 || end_label < 0) continue;

        atomic_inc(&matrix[start_label][end_label]);
    }
    randoms[gid] = state;
}

#endif