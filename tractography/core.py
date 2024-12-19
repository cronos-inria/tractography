import nibabel as nib
import numpy as np
import scipy


def cart2sph(x, y, z):
    radius = np.sqrt(x**2 + y**2 + z**2)
    azimuth = np.arctan2(y, x)
    azimuth[azimuth < 0] = 2 * np.pi + azimuth[azimuth < 0]
    colatitude = np.arccos(z / radius)
    return azimuth, colatitude, radius


def sph2cart(azimuth, colatitude, radius):
    x = radius * np.sin(colatitude) * np.cos(azimuth)
    y = radius * np.sin(colatitude) * np.sin(azimuth)
    z = radius * np.cos(colatitude)
    return x, y, z


def ishtmtx(azimuths, colatitudes, n_coefficients):
    """Generates the mrtrix3 compatible inverse SH transform matrix"""

    matrix = np.zeros((n_coefficients, len(colatitudes)))
    degree = 0
    order = 0
    for i in range(n_coefficients):
        if order < 0:
            matrix[i] = np.sqrt(2) * np.imag(
                scipy.special.sph_harm(-order, degree, azimuths, colatitudes)
            )
        elif order == 0:
            matrix[i] = np.real(
                scipy.special.sph_harm(order, degree, azimuths, colatitudes)
            )
        else:
            matrix[i] = np.sqrt(2) * np.real(
                scipy.special.sph_harm(order, degree, azimuths, colatitudes)
            )

        if degree == order:
            degree += 2
            order = -degree
        else:
            order += 1

    return matrix.T


def remove_duplicate_endpoints(streamlines, dt):
    new_streamlines = []
    for streamline in streamlines:
        diff = np.sum(np.diff(streamline, axis=0) ** 2, axis=1)
        i = np.searchsorted(diff[::-1], dt**2 / 2)
        new_streamlines.append(streamline[: len(streamline) - i - 1])

    return new_streamlines


def apply_mask(image, image_affine, mask, mask_affine):
    """Apply the mask to the image in world space"""

    # Get all voxel coordinates of the image.
    x, y, z = np.meshgrid(
        range(image.shape[0]),
        range(image.shape[1]),
        range(image.shape[2]),
        indexing="ij",
    )
    image_voxels = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
    affine = np.dot(np.linalg.inv(mask_affine), image_affine)
    mask_voxels = nib.affines.apply_affine(affine, image_voxels).astype(int)
    mask_values = mask[*mask_voxels.T].reshape(image.shape[:3])
    nib.save(
        nib.Nifti1Image(mask_values.astype(np.uint8), image_affine), "test-mask.nii.gz"
    )

    return image * mask_values[..., None]
