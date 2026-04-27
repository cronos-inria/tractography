#ifndef __UTILS_CORE__
#define __UTILS_CORE__

#define PI 3.14159265359f

// For now, the histograms are always expressed in spherical harmonics with 45 coefficients.
#define HISTOGRAM_N_COEFFICIENTS 45

/**
 * MODEL_VALUE_T
 * A struct to hold the value of a local model (DTI, Spherical Harmonics, etc.) and its
 * derivatives with respect to theta and phi.
 *
 * IMPORTANT: The derivative with respect to phi must be pre-divided by sin(theta).
**/
typedef struct {
    float value;
    float dtheta;
    float dphi;
} model_value_t;

// NEAREST_VERTEX_LABEL
// Finds the nearest vertex to a given point and returns its label.
// Returns -1 if no vertex is within the distance bound.
//
// vertices: An array of vertices.
// vertex_labels: An array of vertex labels.
// n_vertices: The number of vertices.
// point: The point to find the nearest vertex to.
// distance_upper_bound: The maximum distance between the point and the nearest vertex.
//
// Returns: The label of the nearest vertex.
//
inline int nearest_vertex_label(
        __global const float4 *vertices,
        __global const int *vertex_labels,
        uint n_vertices,
        float4 point,
        float distance_upper_bound)
{
    float min_dist = distance_upper_bound * distance_upper_bound;
    int best_label = -1;
    for (uint i = 0; i < n_vertices; i++) {
        float4 diff = vertices[i] - point;
        float dist2 = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
        if (dist2 < min_dist) {
            min_dist = dist2;
            best_label = vertex_labels[i];
        }
    }
    return best_label;
}


// ATOMIC_ADD_GLOBAL_FLOAT
// Adds two values and atomically store the result in the
// first value. This is needed because OpenCL does
// not natively provide atomic operations on floats.
//
// left: The left operand. The result will be store here.
// right: The right operand.
//
inline void atomic_add_global_float(__global float *left, float right)
{
	// A union is used to safely convert the float's bit pattern 
    // to an unsigned int for use with atomic_cmpxchg.
    union {
        unsigned int iv;
        float fv;
    } old, newval;

    do {
		// Read the current value from global memory and compute
		// the desired new value.
        old.fv = *left;
        newval.fv = old.fv + right;

		// Attempt to write the new value IF the address still holds the old value.
        // If the return value is not equal to old.iv, it means another work-item 
        // changed the value.
    } while (atomic_cmpxchg((__global volatile unsigned int *)left, old.iv, newval.iv) != old.iv);
}

float4 exps2(float4 p, float4 x, float t) {
    float n = length(x);
    if (n <= 0.0f)
        return p;

	float c;
    float s = sincos(t * n, &c);
    return c * p + x * (s / n);
}

float softmax(float x, float k) {
	if (x * k < 30.0f)
		return log(1 + exp(k * x)) / k;
	return x;
}


float dsoftmax(float x, float k) {
	return 1.0f / (1 + exp(-k * x));
}

inline float modulus(float a, float b) {
	return a - b * floor(a / b);
}

inline void wrap(float* azimuth, float* colatitude) {
	*colatitude = modulus(*colatitude, 2.0f * PI);
	if (*colatitude >= PI) {
		*colatitude = PI - modulus(*colatitude, PI);
		*azimuth = *azimuth + PI;
	}
	*azimuth = modulus(*azimuth, 2.0f * PI);
}

inline float4 sph2cart(float azimuth, float colatitude) {
	float sc, sa, ca, cc;
	sc = sincos(colatitude, &cc);
	sa = sincos(azimuth, &ca);
	return (float4) (sc * ca, sc * sa, cc, 0);
}

inline float2 cart2sph(float4 cart) {
	float azimuth = atan2(cart.y, cart.x);
	float colatitude = acos(cart.z);
	wrap(&azimuth, &colatitude);
    return (float2) (azimuth, colatitude);
}

inline float3 to_voxel(const float4 affine[4], float4 point) {
	float3 voxel;
    for (size_t i = 0; i < 3; i++) {
        voxel[i] = dot(affine[i], point);
    }
	return voxel;
}

// Applies the affine transport to a point in homogeneous coordinates.
inline float4 apply_affine(const float4 affine[4], float4 point) {
	float4 new_point;
    for (size_t i = 0; i < 4; i++) {
        new_point[i] = dot(affine[i], point);
    }
	return new_point;
}

inline bool in_image(float3 voxel, uint nx, uint ny, uint nz) {
	return !(voxel.x <= -0.5f || voxel.x >= nx - 0.5f || voxel.y <= -0.5f || voxel.y >= ny - 0.5f || voxel.z <= -0.5f || voxel.z >= nz - 0.5f);
}

inline uint3 to_index(float3 voxel) {
	return (uint3) {round(voxel.x), round(voxel.y), round(voxel.z)};
}

inline uint MWC64X(uint2 *state)
{
    enum {A=4294883355U};
    uint x=(*state).x, c=(*state).y;  // Unpack the state
    uint res=x^c;                     // Calculate the result
    uint hi=mul_hi(x,A);              // Step the RNG
    x=x*A+c;
    c=hi+(x<c);
    *state=(uint2)(x,c);              // Pack the state back up
    return res;                       // Return the next result
}

// Returns a float in the range [0, 1) with uniform distribution.
//
inline float randu(uint2 *state) {
	return (MWC64X(state) >> 8) * 0x1p-24f;
}

// Returns an array of floats in the range [0, 1). For testing
// mostly.
__kernel void randus(__global float* values, uint n_values) {
	uint2 state = {10, 4000};
	for (size_t i = 0; i < n_values; i++) {
		values[i] = randu(&state);
	}
}

// Returns an integer in the range [0, max - 1] with uniform distribution.
//
inline uint randi(uint2 *state, uint max) {
	return (uint) (randu(state) * max);
}

// Returns an array of integer in the range [0, max - 1]. For testing
// mostly.
__kernel void randis(__global uint* values, uint n_values, uint max) {
	uint2 state = {10, 4000};
	for (size_t i = 0; i < n_values; i++) {
		values[i] = randi(&state, max);
	}
}

inline float randn(uint2 *state) {
	float n = randu(state);
	while (n <= 0.0f)
		n = randu(state);
	return sqrt(-2.0f * log(n)) * cos(2 * PI * randu(state));
}

#endif
