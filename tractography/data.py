"""Synthetic test data generators for tractography algorithms.

This module provides functions to generate synthetic datasets with known 
fiber orientations and geometric structures. These datasets are useful for
validating tractography algorithms and demonstrating workflows.

Available datasets include crossing fibers (cross_sh) with fiber-specific
fOD (fiber orientation distributions) and region labels for connectivity
analysis.

"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from nibabel.nifti1 import Nifti1Image

from .core import cart2sph, fibonacci_sphere, ishtmtx
from .utils import normalize_odf


# Common diffusion tensor shapes used in synthetic data generation.
# _BALL: isotropic tensor (equal diffusivity in all directions)
# _X_TENSOR: anisotropic tensor with preference along x-axis
# _Y_TENSOR: anisotropic tensor with preference along y-axis
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


def cross_sh(shape=(150, 150, 1), n_coefficients: int = 45) -> tuple[Nifti1Image, Nifti1Image, Nifti1Image]:
    """Generate a synthetic crossing fiber dataset with fiber orientation distributions.
    
    Creates a 3D volume with two perpendicular crossing fiber bundles (one along x-axis,
    one along y-axis) embedded in an isotropic background. Fiber orientations are 
    represented as spherical harmonic (SH) coefficients.
    
    Args:
        shape: The 3D dimensions of the generated volume (default 150×150×1 voxels).
        n_coefficients: Number of spherical harmonic coefficients (default 45, lmax=8).
    
    Returns:
        A tuple of three Nifti1Image objects:
        - fod: Fiber orientation distributions as SH coefficients (shape + (n_coefficients,))
        - mask: Binary white matter mask indicating valid tracking regions
        - segmentation: Four labeled regions at the ends of each fiber bundle
    """

    affine = np.diag([1.25, 1.25, 1.25, 1])
    affine[:3, 3] = [-100, 20, 33]

    bvectors = fibonacci_sphere(n_coefficients * 3)
    fod_values = _uniform_tensor_field(_BALL, shape, bvectors)

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

    fod_data = _fit_spherical_harmonics(fod_values, bvectors, n_coefficients)
    fod = Nifti1Image(fod_data, affine)

    # The segmentation mask is cross shaped.
    mask_data = np.zeros(shape, dtype=np.uint8)
    mask_data[x_offset:-x_offset] = 1
    mask_data[:, y_offset:-y_offset] = 1
    mask = Nifti1Image(mask_data, affine)

    # Add ROIs at each end of the cross.
    segmentation_data = np.zeros(shape, dtype=np.uint8)
    segmentation_data[x_offset:-x_offset, 0] = 1
    segmentation_data[x_offset:-x_offset, -1] = 2
    segmentation_data[0, y_offset:-y_offset] = 3
    segmentation_data[-1, y_offset:-y_offset] = 4
    segmentation = Nifti1Image(segmentation_data, affine)

    return fod, mask, segmentation


# Provide convenient namespace access to data generators.
# This allows users to call tg.data.sh.cross() if preferred.
sh = SimpleNamespace(cross=cross_sh)


def _uniform_tensor_field(tensor, shape, bvectors):
    """Generate a uniform diffusion tensor field across all voxels.
    
    Evaluates a single 3×3 diffusion tensor at N sampling directions (bvectors)
    to compute signal attenuation values. The same tensor is replicated across
    all voxels in the volume.
    
    Args:
        tensor: 3×3 diffusion tensor matrix defining anisotropic diffusion.
        shape: 3D voxel dimensions of the output volume.
        bvectors: N×3 array of unit vectors defining sampling directions.
    
    Returns:
        Field array of shape (X, Y, Z, N) containing tensor-derived attenuation
        values at each voxel and direction.
    """
    field = np.zeros(shape + (len(bvectors),))
    field[..., :] = np.exp(-np.diag(np.dot(np.dot(bvectors, tensor), bvectors.T)))
    return field


def _fit_spherical_harmonics(fod_values, bvectors, n_coefficients):
    """Convert fiber orientation distribution values to spherical harmonic coefficients.
    
    Fits spherical harmonic basis functions to discretized ODF values across
    the volume. Uses the MRtrix3-compatible inverse SH transform matrix.
    
    Args:
        fod_values: Array of shape (X, Y, Z, N) containing ODF values at N directions.
        bvectors: N×3 array of unit vectors corresponding to the ODF sampling directions.
        n_coefficients: Number of SH coefficients to compute (determines maximum degree).
    
    Returns:
        Normalized SH coefficient array of shape (X, Y, Z, n_coefficients).
    """
    azimuth, colatitude, _ = cart2sph(*bvectors.T)
    imtx, _ = ishtmtx(azimuth, colatitude, n_coefficients)
    mtx = np.linalg.pinv(imtx)
    fod = np.tensordot(fod_values, mtx, axes=(3, 1))
    return normalize_odf(fod)