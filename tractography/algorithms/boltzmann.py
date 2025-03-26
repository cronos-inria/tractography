from pathlib import Path
from string import Template

import numpy as np
import scipy.interpolate as si

import tractography as tg

import pyopencl as cl

# Create the global OpenCL context.
_context = cl.create_some_context(interactive=False)
_queue = cl.CommandQueue(_context)

# Get the number of compute units for the selected device.
_device = _context.devices[0]
_nb_units = _device.max_compute_units

_OPENCL_DIR = Path(__file__).parents[2] / "src"


def boltzmann(fod, affine, seeds, config):

    n_points = 1000

    # Generate a set of orientation where the FODs are evaluated.
    vertices = tg.core.fibonacci_sphere(n_points)
    azimuths, colatitudes, _ = tg.core.cart2sph(*vertices.T)

    # Generate the spherical harmonic matrix and its derivative for
    # the selected orientations.
    n_coefficients = fod.shape[-1]
    matrix, dmatrix = tg.core.ishtmtx(azimuths, colatitudes, n_coefficients)

    # Precompute the inverse affine.
    iaffine = np.linalg.inv(affine)

    # Send readonly data to the device.
    flags = cl.mem_flags.READ_ONLY
    vertices = vertices.astype(np.float32)
    vertex_buffer = cl.Buffer(_context, flags, size=vertices.nbytes)
    cl.enqueue_copy(_queue, vertex_buffer, np.ascontiguousarray(vertices))

    matrix = matrix.astype(np.float32)
    matrix_buffer = cl.Buffer(_context, flags, size=matrix.nbytes)
    cl.enqueue_copy(_queue, matrix_buffer, np.ascontiguousarray(matrix))

    dmatrix = dmatrix.astype(np.float32)
    dmatrix_buffer = cl.Buffer(_context, flags, size=dmatrix.nbytes)
    cl.enqueue_copy(_queue, dmatrix_buffer, np.ascontiguousarray(dmatrix))

    iaffine = iaffine.astype(np.float32)
    affine_buffer = cl.Buffer(_context, flags, size=iaffine.nbytes)
    cl.enqueue_copy(_queue, affine_buffer, np.ascontiguousarray(iaffine))

    fod = fod.astype(np.float32)
    fod_buffer = cl.Buffer(_context, flags, size=fod.nbytes)
    cl.enqueue_copy(_queue, fod_buffer, np.ascontiguousarray(fod))

    seeds_array = tg.seeds.to_array(seeds).astype(np.float32)
    seeds_buffer = cl.Buffer(_context, flags, size=seeds_array.nbytes)
    cl.enqueue_copy(_queue, seeds_buffer, np.ascontiguousarray(seeds_array))

    # Reserve space for the streamlines on the device.
    n_streamlines = len(seeds)
    streamlines = np.zeros((n_streamlines, config.n_steps, 3), dtype=np.float32)
    flags = cl.mem_flags.WRITE_ONLY
    streamlines_buffer = cl.Buffer(_context, flags, size=streamlines.nbytes)

    # Compile the OpenCL program that implements Boltzmann tractography.
    with open(_OPENCL_DIR / "boltzmann.cl") as f:
        kernel = f.read()
    template = Template(kernel)

    # Set constants in the OpenCL code.
    values = {
        "nx": fod.shape[0],
        "ny": fod.shape[1],
        "nz": fod.shape[2],
        "n_directions": len(vertices),
        "n_coefficients": n_coefficients,
        "n_steps": config.n_steps,
        "n_streamlines": n_streamlines,
    }
    source = template.safe_substitute(values)
    program = cl.Program(_context, source).build()

    # Track streamlines.
    args = (
        fod_buffer,
        affine_buffer,
        vertex_buffer,
        matrix_buffer,
        dmatrix_buffer,
        seeds_buffer,
        streamlines_buffer,
        np.float32(config.step_size),
    )
    program.tractography(_queue, (n_streamlines,), None, *args)
    cl.enqueue_copy(_queue, streamlines, streamlines_buffer)

    return streamlines


def boltzmann_reference(fod, affine, seeds, config):
    """The reference algorithm for Boltzmann tractography"""

    n_points = 1000

    # Generate a set of orientation where the FODs are evaluated.
    vertices = tg.core.fibonacci_sphere(n_points)
    azimuths, colatitudes, _ = tg.core.cart2sph(*vertices.T)

    # Generate the spherical harmonic matrix and its derivative for
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
    fod_value = 0
    fod_der_value = [0, 0]
    index = 0
    coefficients = [0]
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
            if fod(voxel)[0][0] <= 0:
                streamline[i:] = location[None, :]
                break

            # Record the new location to the output array.
            points[:] = location

            # Update the orientation displacement.
            coefficients = fod(voxel)[0]
            index = np.argmax(np.dot(vertices, orientation.T))
            fod_value = np.maximum(np.dot(matrix[index], coefficients), 0.01)
            fod_der_value = np.dot(matrix_der[:, index], coefficients)
            delta_angles = fod_der_value / fod_value

            # Move forward and fix wrapping of the angles.
            gamma = 0.05
            angles = angles + delta_angles[::-1] * config.step_size * gamma
            angles = tg.utils.wrap(angles[0], angles[1])
            orientation = np.array(tg.core.sph2cart(*angles, 1))
            location = location + orientation * config.step_size

    return streamlines
