#define PI 3.14159265359f

float4 exps2(float4 p, float4 x, float t) {
    float n = length(x);
    if (n == 0)
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
	return !(voxel.x < -0.5f || voxel.x >= nx - 0.5f || voxel.y < -0.5f || voxel.y >= ny - 0.5f || voxel.z < -0.5f || voxel.z >= nz - 0.5f);
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

// Returns an integer in the range (0, max) with uniform distribution.
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
	while (n == 0)
		n = randu(state);
	return sqrt(-2.0f * log(n)) * cos(2 * PI * randu(state));
}
