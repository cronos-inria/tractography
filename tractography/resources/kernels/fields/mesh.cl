#ifndef __FIELDS_MESH__
#define __FIELDS_MESH__

#define MAX_N_VALUES_PER_VERTEX 16

/**
 * MESH
 * This kernel implements a mesh field, which is a field defined on a
 * tetrahedral mesh. The mesh is defined by a set of vertices and a set of
 * tetrahedra, where each tetrahedron is defined by four vertex indices.
 * 
 * The field values are multidimensional, each having n values defined at the
 * vertices of the mesh, and the field can be interpolated at any point within
 * the mesh using barycentric coordinates.
 */

 typedef struct {
     __global const float4 *vertices;
     __global const int4 *tetrahedra;
     __global const float *values;
     int n_values_per_vertex;
     int n_vertices;
     int n_tetrahedra;
 } Mesh;

 /**
 * COMPUTE_BARYCENTRIC_COORDINATES
 * Computes the barycentric coordinates of a point with respect to a
 * tetrahedron.
 */
inline float4 compute_barycentric_coordinates(
        float4 v0,
        float4 v1,
        float4 v2, 
        float4 v3,
        float4 point) {

    const float3 a = v0.xyz - v3.xyz;
    const float3 b = v1.xyz - v3.xyz;
    const float3 c = v2.xyz - v3.xyz;
    const float3 rhs = point.xyz - v3.xyz;

    float4 bary = (float4)(-1.0f, -1.0f, -1.0f, -1.0f);
    const float denom = dot(a, cross(b, c));
    if (fabs(denom) < 1e-8f) {
        return bary;
    }

    bary.x = dot(rhs, cross(b, c)) / denom;
    bary.y = dot(a, cross(rhs, c)) / denom;
    bary.z = dot(a, cross(b, rhs)) / denom;
    bary.w = 1.0f - bary.x - bary.y - bary.z;

    return bary;
}

/**
 * IS_INSIDE_TETRAHEDRON
 * Checks if the barycentric coordinates indicate that the point is inside the
 * tetrahedron.
 */
inline bool is_inside_tetrahedron(float4 bary) {
    const float epsilon = -1e-6f;
    return (
        bary.x >= epsilon &&
        bary.y >= epsilon &&
        bary.z >= epsilon &&
        bary.w >= epsilon);
}

/**
 * FIND_TETRAHEDRON_CONTAINING_POINT
 * Returns the index of the tetrahedron containing the point, or -1 if the point
 * is outside the mesh.
 */
int find_tetrahedron_containing_point(Mesh mesh, float4 point) {

    // Without a hint, we have to check all tetrahedra. This is not efficient,
    // but it is simple.
    for (int i = 0; i < mesh.n_tetrahedra; i++) {
        int4 tetrahedron = mesh.tetrahedra[i];
        float4 v0 = mesh.vertices[tetrahedron.x];
        float4 v1 = mesh.vertices[tetrahedron.y];
        float4 v2 = mesh.vertices[tetrahedron.z];
        float4 v3 = mesh.vertices[tetrahedron.w];

        // Compute barycentric coordinates.
        float4 bary = compute_barycentric_coordinates(v0, v1, v2, v3, point);
        if (is_inside_tetrahedron(bary)) {
            return i;
        }
    }
    return -1;
}

/**
 * INTERPOLATE_FIELD
 * Interpolates the field values at a point inside a tetrahedron using
 * barycentric coordinates.
 */
void interpolate_field(
        Mesh mesh,
        int tetrahedron_index,
        float4 bary,
        float* values_out) {
    
    int4 tetrahedron = mesh.tetrahedra[tetrahedron_index];

    // The values are stored in a flattened array, where each vertex has 
    // n_values.
    for (int i = 0; i < mesh.n_values_per_vertex; i++) {
        float v0 = mesh.values[tetrahedron.x * mesh.n_values_per_vertex + i];
        float v1 = mesh.values[tetrahedron.y * mesh.n_values_per_vertex + i];
        float v2 = mesh.values[tetrahedron.z * mesh.n_values_per_vertex + i];
        float v3 = mesh.values[tetrahedron.w * mesh.n_values_per_vertex + i];

        // We assume linear combination of the values is valid.
        values_out[i] = bary.x * v0 + bary.y * v1 + bary.z * v2 + bary.w * v3;
    }
}

/**
 * INTERPOLATE_FIELD_AT_POINT
 * Computes the interpolated field values at the point. Is NaN when the point
 * is outside the mesh. Return true if the point is inside the mesh, false
 * otherwise.
 */
bool interpolate_field_at_point(
        Mesh mesh,
        float4 point,
        float* values_out) {

    int tetrahedron_index = find_tetrahedron_containing_point(mesh, point);
    if (tetrahedron_index == -1) {
        // Point is outside the mesh. Set values to zero or some default.
        for (int i = 0; i < mesh.n_values_per_vertex; i++) {
            values_out[i] = nan(0u);
        }
        return false;
    }

    // Compute barycentric coordinates. Note: some optimization could be done
    // here to avoid recomputing the barycentric coordinates.
    int4 tetrahedron = mesh.tetrahedra[tetrahedron_index];
    float4 v0 = mesh.vertices[tetrahedron.x];
    float4 v1 = mesh.vertices[tetrahedron.y];
    float4 v2 = mesh.vertices[tetrahedron.z];
    float4 v3 = mesh.vertices[tetrahedron.w];
    float4 bary = compute_barycentric_coordinates(v0, v1, v2, v3, point);

    // Interpolate the field values using barycentric coordinates.
    interpolate_field(mesh, tetrahedron_index, bary, values_out);
    return true;
}

/**
 * FIND_TETRAHEDRON_CONTAINING_POINTS
 * Returns the indices of the tetrahedra containing the points.
 */
__kernel void find_tetrahedra_containing_points(
        __global const float4 *vertices,
        int n_vertices,
        __global const int4 *tetrahedra,
        int n_tetrahedra,
        __global const float *values,
        int n_values_per_vertex,
        __global const float4 *points,
        int n_points,
        __global int *tetrahedron_indices) {

    uint gid = get_global_id(0);
    if (gid >= n_points) return;

    // Assemble the mesh structure.
    Mesh mesh = {
        vertices, 
        tetrahedra, 
        values,
        n_values_per_vertex,
        n_vertices,
        n_tetrahedra};

    // Find the tetrahedron containing the point.
    tetrahedron_indices[gid] = find_tetrahedron_containing_point(
        mesh,
        points[gid]);
}

/**
 * INTERPOLATE_FIELD_AT_POINTS
 * Returns the interpolated field values at the points. Is NaN when the point
 * is outside the mesh.
 */
__kernel void interpolate_field_at_points(
        __global const float4 *vertices,
        int n_vertices,
        __global const int4 *tetrahedra,
        int n_tetrahedra,
        __global const float *values,
        int n_values_per_vertex,
        __global const float4 *points,
        int n_points,
        __global float *values_out) {

    uint gid = get_global_id(0);
    if (gid >= n_points) return;

    // Assemble the mesh structure.
        Mesh mesh = {
        vertices, 
        tetrahedra, 
        values,
        n_values_per_vertex,
        n_vertices,
        n_tetrahedra};

    // Interpolate the field values at the point.
    float local_values_out[MAX_N_VALUES_PER_VERTEX];
    interpolate_field_at_point(mesh, points[gid], local_values_out);
    for (int i = 0; i < n_values_per_vertex; i++) {
        values_out[gid * n_values_per_vertex + i] = local_values_out[i];
    }
}

#endif // __FIELDS_MESH__