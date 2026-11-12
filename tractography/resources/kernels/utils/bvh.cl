#ifndef __UTILS_BVH__
#define __UTILS_BVH__

// Bounding-volume hierarchy over a triangle mesh. Leaf nodes have left/right
// set to -1, with first/count identifying a range in ordered_triangles.
typedef struct {
    __global const float4* vertices;
    __global const uint4* triangles;
    __global const float4* bounds_min;
    __global const float4* bounds_max;
    __global const int4* nodes;
    __global const uint* ordered_triangles;
    uint n_nodes;
} BVH;

inline bool bvh_segment_intersects_bounds(
    BVH bvh,
    uint node_id,
    float3 start,
    float3 direction,
    float max_t)
{
    float t_min = 0.0f;
    float t_max = max_t;
    const float3 bounds_min = bvh.bounds_min[node_id].xyz;
    const float3 bounds_max = bvh.bounds_max[node_id].xyz;
    for (uint axis = 0; axis < 3; ++axis) {
        if (fabs(direction[axis]) < 1e-12f) {
            if (start[axis] < bounds_min[axis] || start[axis] > bounds_max[axis])
                return false;
        } else {
            const float inverse_direction = 1.0f / direction[axis];
            float near_t = (bounds_min[axis] - start[axis]) * inverse_direction;
            float far_t = (bounds_max[axis] - start[axis]) * inverse_direction;
            if (near_t > far_t) {
                const float swap_t = near_t;
                near_t = far_t;
                far_t = swap_t;
            }
            t_min = max(t_min, near_t);
            t_max = min(t_max, far_t);
            if (t_min > t_max) return false;
        }
    }
    return true;
}

inline bool bvh_intersect_triangle(
    BVH bvh,
    uint triangle_id,
    float3 start,
    float3 direction,
    float max_t,
    float* hit_t,
    float3* barycentric)
{
    const uint4 indices = bvh.triangles[triangle_id];
    const float3 a = bvh.vertices[indices.x].xyz;
    const float3 b = bvh.vertices[indices.y].xyz;
    const float3 c = bvh.vertices[indices.z].xyz;
    const float3 edge1 = b - a;
    const float3 edge2 = c - a;
    const float3 p = cross(direction, edge2);
    const float determinant = dot(edge1, p);
    if (fabs(determinant) < 1e-10f) return false;

    const float inverse_determinant = 1.0f / determinant;
    const float3 offset = start - a;
    const float u = dot(offset, p) * inverse_determinant;
    if (u < -1e-6f || u > 1.0f + 1e-6f) return false;
    const float3 q = cross(offset, edge1);
    const float v = dot(direction, q) * inverse_determinant;
    if (v < -1e-6f || u + v > 1.0f + 1e-6f) return false;
    const float t = dot(edge2, q) * inverse_determinant;
    if (t <= 1e-6f || t > max_t) return false;

    *hit_t = t;
    *barycentric = (float3)(1.0f - u - v, u, v);
    return true;
}

// Return the closest triangle intersection along the finite segment start-end.
inline bool bvh_segment_hit(
    BVH bvh,
    float3 start,
    float3 end,
    uint* hit_triangle,
    float3* hit_barycentric,
    float* hit_t)
{
    const float3 direction = end - start;
    int stack[64];
    uint stack_size = 1;
    stack[0] = 0;
    *hit_t = 1.0f;
    *hit_triangle = 0;
    bool found = false;

    while (stack_size > 0) {
        const uint node_id = (uint)stack[--stack_size];
        if (!bvh_segment_intersects_bounds(bvh, node_id, start, direction, *hit_t))
            continue;

        const int4 node = bvh.nodes[node_id];
        if (node.w > 0) {
            for (int i = 0; i < node.w; ++i) {
                const uint triangle_id = bvh.ordered_triangles[node.z + i];
                float triangle_t;
                float3 triangle_barycentric;
                if (bvh_intersect_triangle(
                        bvh,
                        triangle_id,
                        start,
                        direction,
                        *hit_t,
                        &triangle_t,
                        &triangle_barycentric)) {
                    *hit_t = triangle_t;
                    *hit_triangle = triangle_id;
                    *hit_barycentric = triangle_barycentric;
                    found = true;
                }
            }
        } else {
            if (stack_size + 2 > 64) return found;
            stack[stack_size++] = node.y;
            stack[stack_size++] = node.x;
        }
    }
    return found;
}

#endif
