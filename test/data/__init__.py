import numpy as np

import tractography as tg


# Some commons tensor shapes.
_BALL = [
    [5.0, 0, 0],
    [0, 5.0, 0],
    [0, 0, 5.0],
]
_X_TENSOR = [
    [5.0, 0, 0],
    [0, 1.0, 0],
    [0, 0, 5.0],
]
_Y_TENSOR = [
    [1.0, 0, 0],
    [0, 5.0, 0],
    [0, 0, 5.0],
]


def uniform_isotropic(shape=(10, 10, 10), n_coefficients=45):
    """Generate an uniform isotropic fODF field"""
    bvectors = tg.core.fibonacci_sphere(n_coefficients * 3)
    fod_values = _uniform_tensor_field(_BALL, shape, bvectors)
    return _fit_spherical_harmonics(fod_values, bvectors, n_coefficients)


def cross(shape=(10, 10, 1), n_coefficients=45):
    """Generate a crossing fiber fODF field"""
    bvectors = tg.core.fibonacci_sphere(n_coefficients * 3)
    fod_values = np.zeros(shape + (len(bvectors),))

    x_offset = shape[0] // 3
    x_size = shape[0] - 2 * x_offset
    fod_values[x_offset:-x_offset] = _uniform_tensor_field(
        _X_TENSOR, (x_size, shape[1], shape[2]), bvectors
    )

    y_offset = shape[1] // 3
    y_size = shape[1] - 2 * y_offset
    fod_values[:, y_offset:-y_offset] += _uniform_tensor_field(
        _Y_TENSOR, (shape[0], y_size, shape[2]), bvectors
    )

    return _fit_spherical_harmonics(fod_values, bvectors, n_coefficients)


def _uniform_tensor_field(tensor, shape, bvectors):
    """Generate a uniform tensor field (the tensor is the same in all voxels)"""
    field = np.zeros(shape + (len(bvectors),))
    field[..., :] = np.exp(-np.diag(np.dot(np.dot(bvectors, tensor), bvectors.T)))
    return field


def _fit_spherical_harmonics(fod_values, bvectors, n_coefficients):
    """Compute the spherical harmonics coefficients from fOD values"""
    azimuth, colatitude, _ = tg.core.cart2sph(*bvectors.T)
    ishtmtx, _ = tg.core.ishtmtx(azimuth, colatitude, n_coefficients)
    shtmtx = np.linalg.pinv(ishtmtx)
    fod = np.tensordot(fod_values, shtmtx, axes=(3, 1))
    return tg.utils.normalize_odf(fod)


def _generate_cross():

    n_coefficients = 45
    bvectors = tg.core.fibonacci_sphere(100)
    shape = (10, 10, 10, 100)

    # Add a ball everywhere.
    fod_values = np.zeros(shape)
    tensor = np.array(
        [
            [5.0, 0, 0],
            [0, 5.0, 0],
            [0, 0, 5.0],
        ]
    )
    fod_values[..., :] = np.exp(-np.diag(np.dot(np.dot(bvectors, tensor), bvectors.T)))

    tensor = np.array(
        [
            [1.0, 0, 0],
            [0, 5.0, 0],
            [0, 0, 5.0],
        ]
    )
    fod_values[:, 4:6, 4:6, :] += np.exp(
        -np.diag(np.dot(np.dot(bvectors, tensor), bvectors.T))
    )[None, None, None, :]

    tensor = np.array(
        [
            [5.0, 0, 0],
            [0, 1.0, 0],
            [0, 0, 5.0],
        ]
    )
    fod_values[4:6, :, 4:6, :] += np.exp(
        -np.diag(np.dot(np.dot(bvectors, tensor), bvectors.T))
    )[None, None, None, :]

    tensor = np.array(
        [
            [5.0, 0, 0],
            [0, 5.0, 0],
            [0, 0, 1.0],
        ]
    )
    fod_values[4:6, 4:6, :, :] += np.exp(
        -np.diag(np.dot(np.dot(bvectors, tensor), bvectors.T))
    )[None, None, None, :]

    # Compute SH representation.
    azimuth, colatitude, _ = tg.core.cart2sph(*bvectors.T)
    ishtmtx, _ = tg.core.ishtmtx(azimuth, colatitude, n_coefficients)
    shtmtx = np.linalg.pinv(ishtmtx)
    fod = np.tensordot(fod_values, shtmtx, axes=(3, 1))
    fod /= fod[..., 0:1] * np.sqrt(4 * np.pi)

    path = _DATA_DIR / "cross-fod.nii.gz"
    os.makedirs(path.parent, exist_ok=True)
    nib.save(nib.Nifti1Image(fod, np.eye(4)), path)

    # Generate the WM mask.
    mask = np.zeros(shape[:3], dtype=np.uint8)
    mask[:, 4:6, 4:6] = 1
    mask[4:6, :, 4:6] = 1
    mask[4:6, 4:6, :] = 1
    path = _DATA_DIR / "cross-wm.nii.gz"
    os.makedirs(path.parent, exist_ok=True)
    nib.save(nib.Nifti1Image(mask, np.eye(4)), path)

    # Generate the seed mask.
    seed_mask = np.zeros(shape[:3], dtype=np.uint8)
    seed_mask[[0, -1], 4:6, 4:6] = 1
    seed_mask[4:6, [0, -1], 4:6] = 1
    seed_mask[4:6, 4:6, [0, -1]] = 1
    path = _DATA_DIR / "cross-seed.nii.gz"
    os.makedirs(path.parent, exist_ok=True)
    nib.save(nib.Nifti1Image(seed_mask, np.eye(4)), path)
