#define PI 3.14159265359f


float modulus(float a, float b)
{
	return a - b * floor(a / b);
}

void wrap(float* azimuth, float* colatitude)
{

	*colatitude = modulus(*colatitude, 2.0f * PI);
	if (*colatitude >= PI) {
		*colatitude = PI - modulus(*colatitude, PI);
		*azimuth = *azimuth + PI;
	}
	*azimuth = modulus(*azimuth, 2.0f * PI);
}

void sph2cart(float azimuth, float colatitude, float* x, float* y, float* z) {
	float sc, sa, ca;
	sc = sincos(colatitude, z);
	sa = sincos(azimuth, &ca);
	*x = sc * ca;
	*y = sc * sa;
}

void to_voxel(__global const float affine[4][4], float point[3], float voxel[3]) {
    for (size_t i = 0; i < 3; i++) {
        voxel[i] = affine[i][0] * point[0] +
                   affine[i][1] * point[1] +
                   affine[i][2] * point[2] +
                   affine[i][3];
    }
}


float norm(float v[3]) {
	return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

size_t pick_orientation(
		__global const float vertices[$n_directions][3],
		float orientation[3])
{
	float best_angle = -2;
	size_t best_index = -1;
	for (size_t i = 0; i < $n_directions; i++) {
		float angle = vertices[i][0] * orientation[0] + vertices[i][1] * orientation[1] + vertices[i][2] * orientation[2];
		if (angle > best_angle) {
			best_angle = angle;
			best_index = i;
		}
	}
	return best_index;
}


void sample_fod(
        __global const float fod[$nx][$ny][$nz][$n_coefficients],
		float voxel[3],
		float coefficients[$n_coefficients])
{
	if (voxel[0] < 0 || voxel[0] > $nx - 1 || voxel[1] < 0 || voxel[1] > $ny - 1 || voxel[2] < 0 || voxel[2] > $nz - 1) {
		for (size_t i = 0; i < $n_coefficients; i++) {
			coefficients[i] = 0;
		}
	}
	else {
		for (size_t i = 0; i < $n_coefficients; i++) {
			coefficients[i] = fod[(int) round(voxel[0])][(int) round(voxel[1])][(int) round(voxel[2])][i];
		}
	}

}

__kernel void tractography(
        __global const float fod[$nx][$ny][$nz][$n_coefficients],
        __global const float affine[4][4],
        __global const float vertices[$n_directions][3],
		__global const float matrix[$n_directions][$n_coefficients],
		__global const float dmatrix[2][$n_directions][$n_coefficients],
        __global const float seeds[$n_streamlines][6],
        __global float streamlines[$n_streamlines][$n_steps][3],
		float step_size,
		float acceleration_factor)
{
    uint gid = get_global_id(0);

    // Initialize the first streamline point with the seed.
	float location[3] = {seeds[gid][0], seeds[gid][1], seeds[gid][2]};
    float orientation[3] = {seeds[gid][3], seeds[gid][4], seeds[gid][5]};
	float colatitude = acos(orientation[2]);
	float azimuth = atan2(orientation[1], orientation[0]);
	if (azimuth < 0) {
		azimuth = 2 * PI + azimuth;
	}

	size_t index = 0;
	float coefficients[$n_coefficients];
    for (size_t n = 0; n < $n_steps; n++) {

		streamlines[gid][n][0] = location[0];
		streamlines[gid][n][1] = location[1];
		streamlines[gid][n][2] = location[2];

        // Go back to voxel space.
        float voxel[3];
        to_voxel(affine, location, voxel);

		// Check if we still have an FOD.
		sample_fod(fod, voxel, coefficients);
        if (coefficients[0] <= 0.0f) {
            continue;
        }

		// Update the orientation displacement.
		index = pick_orientation(vertices, orientation);
		float fod_value = 0.0f;
		float fod_colatitude_value = 0.0f;
		float fod_azimuth_value = 0.0f;
		for (size_t i = 0; i < $n_coefficients; i++) {
			fod_value += coefficients[i] * matrix[index][i];
			fod_colatitude_value += coefficients[i] * dmatrix[0][index][i];
			fod_azimuth_value += coefficients[i] * dmatrix[1][index][i];
		}
		fod_value = fmax(fod_value,  0.001f);

		// Displace angles and fix wrapping of the angles.
		float sc = fmax(sin(colatitude), 0.001f);
		azimuth = azimuth + fod_azimuth_value / fod_value / sc * step_size * acceleration_factor;
		colatitude = colatitude + fod_colatitude_value / fod_value * step_size * acceleration_factor;
		wrap(&azimuth, &colatitude);

		// Move foward.
		sph2cart(azimuth, colatitude, orientation, orientation + 1, orientation + 2);
		location[0] = location[0] + orientation[0] * step_size;
		location[1] = location[1] + orientation[1] * step_size;
		location[2] = location[2] + orientation[2] * step_size;
	}
}
