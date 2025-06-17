import os
from pathlib import Path

import nibabel as nib
import numpy as np

import tractography as tg


_DATA_DIR = Path(__file__).parents[1] / "data"


def generate_cross_test_data():

    if os.path.exists(_DATA_DIR / "cross-fod.nii.gz"):
        return

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
