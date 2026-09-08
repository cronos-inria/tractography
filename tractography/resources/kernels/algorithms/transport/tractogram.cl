#ifndef __TRANSPORT_TRACTOGRAM__
#define __TRANSPORT_TRACTOGRAM__

#include "utils/core.cl"

// Select the model used based on preprocessor directives. All models provide
// their own evaluate_interpolated_model function.
#include "models/select.cl"
#include "algorithms/diffusion/core.cl"

// Select the type of field to use based on preprocessor directives. All fields
// provide their own interpolate_field_at_point function.
#include "fields/select.cl"

void track_streamline(
		Field field,
		float4 location,
		float4 orientation,
		float dt,
		float save_at,
		float gamma,
        __global float4 streamlines[$n_steps],
        __global uint* length)
{

	// Initialize the first streamline location and orientation with the
	// provided location and orientation.
	streamlines[0] = location;
	size_t n = 1;
	float time = 0.0f;
	while (n < $n_steps) {

		// Evaluate the model at the current particle location. Stop if the 
		// particle is not in the domain.
		float interpolated_model[45];
		if (!interpolate_field_at_point(field, location, interpolated_model))
			break;
		model_value_t evaluated_model = evaluate_interpolated_model(
			interpolated_model,
			$n_coefficients,
			orientation
		);

		// Update the orientation.
		orientation = update_orientation(
			evaluated_model, 
			orientation, 
			0, 
			dt, 
			gamma, 
			0.0f
		);

		// Move the location forward and add it to the streamline.
		location = mad(dt, orientation, location);
		
		// Move time forward and record the location if necessary.
		time += dt;
		if (time >= save_at) {
			time -= save_at;
			streamlines[n] = mad(-time, orientation, location);
			n++;
		}

	}
	*length = n;
}

// Define a thin wrapper which depends on the field type.
#if defined(FIELD_TETRAHEDRAL_MESH)

__kernel void tractogram(
     __global const float4 *vertices,
     __global const int4 *tetrahedra,
     __global const float *values,
     int n_values_per_vertex,
     int n_vertices,
     int n_tetrahedra,
	__global const float4 seeds[$n_streamlines][2],
	float dt,
	float save_at,
	float gamma,
	__global float4 streamlines[$n_streamlines][$n_steps],
	__global uint lengths[$n_streamlines])
{
    uint gid = get_global_id(0);
	if (gid >= $n_streamlines) return;

	// Assemble the field structure.
    Field field = {
        vertices,
		tetrahedra,
		values,
		n_values_per_vertex,
		n_vertices,
		n_tetrahedra,
    };

	track_streamline(
		field,
		seeds[gid][0],
		seeds[gid][1],
		dt,
		save_at,
		gamma,
		streamlines[gid],
		lengths + gid
	);
}

#elif defined(FIELD_IMAGE)

__kernel void tractogram(
        __global const float fod[$nx][$ny][$nz][$n_coefficients],
        __global const float4 affine[4],
		__global const uchar *mask,
        __global const uint4 *mask_shape,
        __global const float4 mask_affine[4],
        __global const float4 seeds[$n_streamlines][2],
        float dt,
		float save_at,
		float gamma,
        __global float4 streamlines[$n_streamlines][$n_steps],
        __global uint lengths[$n_streamlines])
{
    uint gid = get_global_id(0);
	if (gid >= $n_streamlines) return;

	// Assemble the field structure.
    Field field = {
        fod,
        {$nx, $ny, $nz, $n_coefficients},
        affine,
        mask,
        *mask_shape,
        mask_affine
    };

	// Each thread tracks one streamline.
	track_streamline(
		field,
		seeds[gid][0],
		seeds[gid][1],
		dt,
		save_at,
		gamma,
		streamlines[gid],
		lengths + gid
	);
}

#endif
#endif
