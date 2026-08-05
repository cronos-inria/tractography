#ifndef __FIELDS_IMAGE__
#define __FIELDS_IMAGE__

#define MAX_N_VALUES_PER_VERTEX 16

/**
 * IMAGE
 * This kernel implements an image field, which is a field defined on a
 * regular grid. The values of the image are only valid within a specified
 * mask and withing the image boundary. The image and the mask do not need to
 * be sampled on the same grid.
 * 
 * The field values are multidimensional, each having n values defined at the
 * voxels of the image, and the field can be interpolated at any point within
 * the image using nearest neighbor interpolation.
 */

typedef struct {
     __global const float *values;
     uint4 image_shape;
    __global const float4 *image_affine; // From voxel coordinates to world coordinates!
    __global const uchar *mask;
     uint4 mask_shape;
    __global const float4 *mask_affine; // From world coordinates to voxel coordinates!
} Image;

/**
 * WORLD_TO_VOXEL
 * Converts a point in world coordinates to voxel coordinates using the
 * provided affine transformation.
 */
inline uint3 world_to_voxel(__global const float4 *affine, float4 point) {
	float3 voxel;
    for (size_t i = 0; i < 3; i++) {
        voxel[i] = dot(affine[i], point);
    }

    // The voxel center is at 0.0, the voxel goes from -0.5 to 0.5.
	return (uint3) {round(voxel.x), round(voxel.y), round(voxel.z)};
}

/**
 * VOXEL_IN_IMAGE
 * Returns true if the voxel coordinates are inside the image boundaries, false
 * otherwise.
 */
inline bool voxel_in_image(uint3 voxel, uint3 shape) {
	return !(
        voxel.x <= 0u || 
        voxel.x >= shape.x || 
        voxel.y <= 0u || 
        voxel.y >= shape.y ||
        voxel.z <= 0u ||
        voxel.z >= shape.z
    );
}

/**
 * IS_POINT_INSIDE_MASK
 * Returns true if the point is inside the mask, false otherwise.
 */
inline bool is_point_inside_mask(Image image, float4 point) {

    // Convert the point from world coordinates to mask voxel coordinates.
    uint3 mask_voxel = world_to_voxel(image.mask_affine, point);
    if (!voxel_in_image(mask_voxel, image.mask_shape.xyz)) {
        return false;
    }
    
    uint index = (int)mask_voxel.x +
                 (int)mask_voxel.y * image.mask_shape.x +
                 (int)mask_voxel.z * image.mask_shape.x * image.mask_shape.y;
    return (image.mask[index] != 0u);
}

/**
 * INTERPOLATE_FIELD_AT_POINT
 * Computes the interpolated field values at the point. Return true if the point
 * is inside the image, false
 * otherwise.
 */
bool interpolate_field_at_point(
        Image image,
        float4 point,
        float* values_out) {

    if (!is_point_inside_mask(image, point)) {
        for (uint i = 0; i < image.image_shape.w; i++) {
            values_out[i] = nan(0u);
        }
        return false;
    }

    // Convert the point from world coordinates to image voxel coordinates.
    uint3 image_voxel = world_to_voxel(image.image_affine, point);
    if (!voxel_in_image(image_voxel, image.image_shape.xyz)) {
        for (uint i = 0; i < image.image_shape.w; i++) {
            values_out[i] = nan(0u);
        }
        return false;
    }
    
    uint index = (int)image_voxel.x +
                 (int)image_voxel.y * image.image_shape.x +
                 (int)image_voxel.z * image.image_shape.x * image.image_shape.y;
    for (uint i = 0; i < image.image_shape.w; i++) {
        values_out[i] = image.values[index * image.image_shape.w + i];
    }

    return true;
}

/**
 * INTERPOLATE_FIELD_AT_POINTS
 * Returns the interpolated field values at the points. Is NaN when the point
 * is outside the mesh.
 */
__kernel void interpolate_field_at_points(
        __global const float *values,
        __global const uint4 *image_shape,
        __global const float4 image_affine[4],
        __global const uchar *mask,
        __global const uint4 *mask_shape,
        __global const float4 mask_affine[4],
        __global const float4 *points,
        int n_points,
        __global float *values_out) {

    uint gid = get_global_id(0);
    if (gid >= n_points) return;

    // Assemble the mesh structure.
    Image image = {
        values,
        *image_shape,
        image_affine,
        mask,
        *mask_shape,
        mask_affine
    };

    // Interpolate the field values at the point.
    float local_values_out[MAX_N_VALUES_PER_VERTEX];
    interpolate_field_at_point(image, points[gid], local_values_out);
    for (int i = 0; i < image.image_shape.w; i++) {
        values_out[gid * image.image_shape.w + i] = local_values_out[i];
    }
}

#endif // __FIELDS_IMAGE__