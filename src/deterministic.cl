void to_voxel(__global const float affine[4][4], float point[3], uint voxel[3]) {
    for (size_t i = 0; i < 3; i++) {
        voxel[i] = affine[i][0] * point[0] +
                   affine[i][1] * point[1] +
                   affine[i][2] * point[2] +
                   affine[i][3];
    }
}

size_t pick_direction(__global const float fod[$nx][$ny][$nz][$n_directions], float valid[$n_directions], uint voxel[3]) {

	// Pick the direction with max value.
	float cs;
	size_t j = 0;
	float current_max = 0;
	for (size_t i = 0; i < $n_directions; i++) {
	    cs = fod[voxel[0]][voxel[1]][voxel[2]][i] * valid[i];
	    if (cs > current_max) {
		current_max = cs;
		j = i;
	    }
	}

	return j;
}

__kernel void tractography(
        __global const float fod[$nx][$ny][$nz][$n_directions],
        __global const float affine[4][4],
        __global const float vertices[$n_directions][3],
        __global const float seeds[$n_streamlines][6],
        float dt,
        float max_angle,
        __global float streamline[$n_streamlines][$n_steps][3])
{
    uint gid = get_global_id(0);

    // Initialize the first streamline point with the seed.
    streamline[gid][0][0] = seeds[gid][0];
    streamline[gid][0][1] = seeds[gid][1];
    streamline[gid][0][2] = seeds[gid][2];

    float orientation[3] = {seeds[gid][3], seeds[gid][4], seeds[gid][5]};

    for (size_t n = 1; n < $n_steps; n++ ) {

        // Go back to voxel space.
        float point[3] = {streamline[gid][n-1][0], streamline[gid][n-1][1], streamline[gid][n-1][2]};
        uint voxel[3];
        to_voxel(affine, point, voxel);

        // Find the valid orientations.
        float valid[$n_directions];
        for (size_t i = 0; i < $n_directions; i++) {
            if ((vertices[i][0] * orientation[0] + vertices[i][1] * orientation[1] + vertices[i][2] * orientation[2]) > max_angle) {
                valid[i] = 1.0;
            }
            else {
                valid[i] = 0.0;
            }
        }

        // To pick, we need the sum as a normalization factor.
        float sum = 0;
        for (size_t i = 0; i < $n_directions; i++) {
            sum += fod[voxel[0]][voxel[1]][voxel[2]][i] * valid[i];
        }

        // If the normalization factor is 0, there is nowhere to go. Stay
        // on the same point.
        if (sum == 0.0) {
            streamline[gid][n][0] = streamline[gid][n-1][0];
            streamline[gid][n][1] = streamline[gid][n-1][1];
            streamline[gid][n][2] = streamline[gid][n-1][2];
            continue;
        }

	// Pick the next direction.	
	size_t j = pick_direction(fod, valid, voxel);

        // Add the point to the streamline.
        streamline[gid][n][0] = streamline[gid][n-1][0] + dt * vertices[j][0];
        streamline[gid][n][1] = streamline[gid][n-1][1] + dt * vertices[j][1];
        streamline[gid][n][2] = streamline[gid][n-1][2] + dt * vertices[j][2];

        orientation[0] = vertices[j][0];
        orientation[1] = vertices[j][1];
        orientation[2] = vertices[j][2];
    }
}
