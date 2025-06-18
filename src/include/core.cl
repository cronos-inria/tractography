#define PI 3.14159265359f

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

inline float3 to_voxel(float4 affine[4], float4 point) {
	float3 voxel;
    for (size_t i = 0; i < 3; i++) {
        voxel[i] = dot(affine[i], point);
    }
	return voxel;
}

inline bool in_image(float3 voxel, uint nx, uint ny, uint nz) {
	return !(voxel.x < 0 || voxel.x >= nx || voxel.y < 0 || voxel.y >= ny || voxel.z < 0 || voxel.z >= nz);
}

inline uint3 to_index(float3 voxel) {
	uint3 index = {(uint) rint(voxel.x), (uint) rint(voxel.y), (uint) rint(voxel.z)};
	return index;
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

inline float randu(uint2 *state) {
	return (float) MWC64X(state) / 4294967295.0f;
}

inline float randn(uint2 *state) {
	return sqrt(-2.0f * log(randu(state))) * cos(2 * PI * randu(state)); 
}
