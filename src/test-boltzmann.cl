#include "core.cl"

void sample_fod(
	__global const float fod[3][4][5][6],
	float voxel[3],
	__global float coefficients[6])
{
	if (voxel[0] < 0 || voxel[0] >= 3 || voxel[1] < 0 || voxel[1] >= 4 || voxel[2] < 0 || voxel[2] >= 5) {
		for (size_t i = 0; i < 6; i++) {
			coefficients[i] = 0;
		}
	}
	else {
		for (size_t i = 0; i < 6; i++) {
			coefficients[i] = fod[(size_t) round(voxel[0])][(size_t) round(voxel[1])][(size_t) round(voxel[2])][i];
		}
	}

}

__kernel void test_modulus(
	__global const float* values,
	int n_values,
	__global float* result)
{
	for (size_t i = 0; i < n_values; i++) {
		result[i] = modulus(values[i], 2 * PI);
	}
}

__kernel void test_wrap(
	__global float* azimuths,
	__global float* colatitudes,
	int n_values)
{
	for (size_t i = 0; i < n_values; i++) {
		float a = azimuths[i];
		float c = colatitudes[i];
		wrap(&a, &c);
		azimuths[i] = a;
		colatitudes[i] = c;
	}
}

__kernel void test_sph2cart(
	__global const float* azimuths,
	__global const float* colatitudes,
	int n_values,
	__global float* x,
	__global float* y,
	__global float* z)
{
	for (size_t i = 0; i < n_values; i++) {
		float4 r = sph2cart(azimuths[i], colatitudes[i]);
		x[i] = r.x;
		y[i] = r.y;
		z[i] = r.z;
	}
}

__kernel void test_cart2sph(
	__global const float4* cart,
	int n_values,
	__global float2* sph)
{
	for (size_t i = 0; i < n_values; i++) {
		sph[i] = cart2sph(cart[i]);
	}
}

__kernel void test_sample_fod(
	__global const float fod[3][4][5][6],
	__global const float voxel[3],
	__global float coefficients[6])
{
	float v[3] = {voxel[0], voxel[1], voxel[2]};
	sample_fod(fod, v, coefficients);
}

__kernel void test_randu(
	__global float *values,
	uint n_values)
{
	uint2 state = {1, 0};
	for (size_t i = 0; i < n_values; i++) {
		values[i] = randu(&state);
	}
}

__kernel void test_randn(
	__global float *values,
	uint n_values)
{
	uint2 state = {1, 0};
	for (size_t i = 0; i < n_values; i++) {
		values[i] = randn(&state);
	}
}
