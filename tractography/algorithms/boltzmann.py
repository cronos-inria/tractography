import numpy as np
import scipy.interpolate as si

import tractography as tg


def boltzmann(fod, affine, seeds, config):
    """The reference algorithm for Boltzmann tractography"""

    n_points = 1000

    # Generate a set of orientation where the FODs are evaluated.
    vertices = tg.core.fibonacci_sphere(n_points)
    azimuths, colatitudes, _ = tg.core.cart2sph(*vertices.T)

    # Generate the spherical harmonic matrix and it's derivative for
    # the selected orientations.
    n_coefficients = fod.shape[-1]
    matrix, matrix_der = tg.core.ishtmtx(azimuths, colatitudes, n_coefficients)

    # Interpolate the FODs.
    x = np.arange(fod.shape[0])
    y = np.arange(fod.shape[1])
    z = np.arange(fod.shape[2])
    fod = si.RegularGridInterpolator(
        (x, y, z), fod, method="nearest", bounds_error=False, fill_value=0
    )

    # Precompute the inverse affine.
    iaffine = np.linalg.inv(affine)

    streamlines = []
    for seed in seeds:

        # Initialize the first point of the streamline from the seed.
        location = seed.location
        orientation = np.array([seed.orientation])
        *angles, _ = tg.core.cart2sph(*orientation.T)
        angles = np.array(angles).squeeze()

        streamline = np.zeros((config.n_steps, 3))
        streamlines.append(streamline)
        for i, points in enumerate(streamline):

            # Go back to voxel space.
            voxel = tg.utils.to_voxel(iaffine, location)

            # Check if we still have an FOD.
            if fod(voxel)[0][0] == 0:
                streamline[i:] = location[None, :]
                break

            # Record the new location to the output array.
            points[:] = location

            # Update the orientation displacement.
            coefficients = fod(voxel)[0]
            index = np.argmax(np.dot(vertices, orientation.T))
            fod_value = np.dot(matrix[index], coefficients)
            fod_der_value = np.dot(matrix_der[:, index], coefficients)
            delta_angles = fod_der_value / np.maximum(fod_value, 0.01)

            # Move forward and fix wrapping of the angles.
            angles += delta_angles[::-1] * config.step_size
            angles = tg.utils.wrap(angles[0], angles[1])
            orientation = np.array(tg.core.sph2cart(*angles, 1))
            location += orientation * config.step_size

    return streamlines
