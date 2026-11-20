"""Generate seeds for tractography"""

from pathlib import Path

import nibabel as nib
from nibabel.nifti1 import Nifti1Image
import nimesh
import numpy as np
import numpy.typing as npt

from . import core
from . import local as local_models
from .opencl import BufferAccess, OpenCLRuntime


def _from_components(
    locations: npt.ArrayLike, orientations: npt.ArrayLike
) -> npt.NDArray[np.float32]:
    """Build the canonical ``(n_seeds, 8)`` homogeneous seed array."""
    locations = np.asarray(locations, dtype=np.float32)
    orientations = np.asarray(orientations, dtype=np.float32)
    if locations.ndim != 2 or locations.shape[1] != 3:
        raise ValueError("locations must have shape (n_seeds, 3).")
    if orientations.shape != locations.shape:
        raise ValueError("orientations must have the same shape as locations.")

    result = np.empty((len(locations), 8), dtype=np.float32)
    result[:, :3] = locations
    result[:, 3] = 1.0
    result[:, 4:7] = orientations
    result[:, 7] = 0.0
    return result


def validate_seed_array(seeds: npt.ArrayLike) -> npt.NDArray[np.float32]:
    """Validate and normalize a canonical seed array.

    Args:
        seeds: Array with shape ``(n_seeds, 8)``. Each row contains a
            homogeneous position ``(x, y, z, 1)`` and direction
            ``(dx, dy, dz, 0)``.

    Returns:
        A C-contiguous ``float32`` array with the same seed values.

    Raises:
        ValueError: If the shape or homogeneous-coordinate columns are invalid.
    """
    result = np.ascontiguousarray(seeds, dtype=np.float32)
    if result.ndim != 2 or result.shape[1] != 8:
        raise ValueError("seeds must have shape (n_seeds, 8).")
    if not np.all(result[:, 3] == 1) or not np.all(result[:, 7] == 0):
        raise ValueError("seed position and orientation homogeneous values must be 1 and 0.")
    return result


def from_surface(
    surface: nimesh.Mesh, n_seeds: int, cone_angle: float = 0
) -> npt.NDArray[np.float32]:
    """Generate seeds from a surface

    The seeds are generated uniformly over the triangles of the
    surface. The orientation of each seed is in a cone centered on
    the normal of the surface.

    It is assumed the ordering of the triangle indices follows the
    right-hand rule and the normals point outward.

    Args:
        surface: The surface from which to generate the seeds, for example the
            lh.white from FreeSurfer.
        n_seeds: The number of seeds to generate.
        cone_angle: The opening angle of the cone in degrees.

    Return:
        An array of shape ``(n_seeds, 8)`` suitable for tractography.

    """

    if n_seeds == 0:
        return np.empty((0, 8), dtype=np.float32)

    # First choose a triangle for every seed and generate a random point
    # inside that triangle.
    vertices, triangles = surface.vertices, surface.triangles
    triangle_vertices = vertices[triangles]
    areas = (
        np.linalg.norm(
            np.cross(
                triangle_vertices[:, 0] - triangle_vertices[:, 1],
                triangle_vertices[:, 0] - triangle_vertices[:, 2],
            ),
            axis=-1,
        )
        / 2
    )
    rng = np.random.default_rng()
    indices = rng.choice(len(triangles), size=n_seeds, p=areas / np.sum(areas))
    barycentric = _random_barycentric_coordinates(n_seeds)
    triangle_indices = triangles[indices]
    triangle_vertices = vertices[triangle_indices]
    locations = np.sum(triangle_vertices * barycentric[..., None], axis=1)

    # The orientations of the seeds are random but oriented with the normals.
    normals = _triangle_normals(vertices, triangles)
    if cone_angle == 0:
        orientations = -normals[indices]
    else:
        orientations = _sample_cone(rng, -normals[indices], cone_angle)

    return _from_components(locations, orientations)


def from_mask(mask: Nifti1Image, n_seeds: int) -> npt.NDArray[np.float32]:
    """Generate seeds from a 3D mask

    The seeds are generated uniformly inside voxels with non-zero
    values in the mask. The orientations are uniform on the sphere.

    Args:
        mask: The Nifti1Image containing the mask.
        n_seeds: The number of seeds to generate.

    Return:
        An array of shape ``(n_seeds, 8)`` suitable for tractography.

    """

    if n_seeds == 0:
        return np.empty((0, 8), dtype=np.float32)

    # Get the non-zero voxels and generate random points inside them.
    voxels = np.array(list(zip(*np.nonzero(mask.get_fdata()))))
    indices = np.random.randint(len(voxels), size=n_seeds)
    locations_voxel = voxels[indices] + np.random.rand(n_seeds, 3) - [0.5, 0.5, 0.5]
    locations = nib.affines.apply_affine(mask.affine, locations_voxel)

    # The orientation are uniformly distributed on the sphere.
    orientations = np.random.randn(n_seeds, 3)
    orientations /= np.linalg.norm(orientations, axis=1, keepdims=True)

    return _from_components(locations, orientations)


def from_fod(
    fod: Nifti1Image,
    n_seeds: int,
    use_opencl: bool = True,
    *,
    runtime: OpenCLRuntime | None = None,
) -> npt.NDArray[np.float32]:
    """Generate seeds from fibre orientation distributions (FOD)

    The seeds are generated uniformly in voxels with non-zero average
    FOD with the orientations importance sampled according to the local
    FOD.

    Args:
        fod: The Nifti1Image containing the FOD.
        n_seeds: The number of seeds to generate.
        use_opencl: If True, the seeds will be generated using the OpenCL
            implementation which can run on GPU.
        runtime: Optional runtime to reuse. One is created when omitted and
            ``use_opencl`` is True.

    Return:
        An array of shape ``(n_seeds, 8)`` suitable for tractography.

    """

    if n_seeds == 0:
        return np.empty((0, 8), dtype=np.float32)

    if use_opencl:
        if runtime is None:
            runtime = OpenCLRuntime()

        # Use a sparse format for the FOD.
        mask = np.any(fod.get_fdata() != 0, axis=-1)
        indices = np.array(np.nonzero(mask), dtype=np.float32).T
        indices = np.hstack((indices, np.ones((len(indices), 1)))).astype(np.float32)
        inline_fod = fod.get_fdata()[mask].astype(np.float32)
        affine = fod.affine.astype(np.float32)

        randoms = np.random.randint(4294967295, size=(n_seeds, 2)).astype(np.uint32)

        fod_buffer = runtime.buffer_from_array(inline_fod, BufferAccess.READ_ONLY)
        indices_buffer = runtime.buffer_from_array(indices, BufferAccess.READ_ONLY)
        affine_buffer = runtime.buffer_from_array(affine, BufferAccess.READ_ONLY)
        randoms_buffer = runtime.buffer_from_array(randoms, BufferAccess.READ_ONLY)
        seeds_buffer = runtime.allocate_buffer(4 * 8 * n_seeds)

        # Compile the OpenCL program that generates seeds.
        model = local_models.from_shape(fod.shape)
        values = {
            "model": model.opencl_model,
            "nnz": len(indices),
            "n_coefficients": fod.shape[-1],
            "n_seeds": n_seeds,
        }
        program = runtime.build_kernel_program("utils/seeds.cl", values)

        program.seeds_from_fod(
            runtime.queue,
            (n_seeds,),
            None,
            fod_buffer,
            indices_buffer,
            affine_buffer,
            randoms_buffer,
            seeds_buffer,
        )
        seeds = np.zeros((n_seeds, 8), np.float32)
        runtime.copy_from_buffer(seeds_buffer, seeds, is_blocking=True)

        return seeds

    else:
        # Get the non-zero voxels.
        voxels = np.array(list(zip(*np.nonzero(np.any(fod.get_fdata() != 0, axis=-1)))))
        indices = np.random.randint(len(voxels), size=n_seeds)
        locations_voxel = voxels[indices] + np.random.rand(n_seeds, 3) - [0.5, 0.5, 0.5]
        locations = nib.affines.apply_affine(fod.affine, locations_voxel)

        # Preprare discretization of the ODFs.
        vertices = core.fibonacci_sphere(1000)
        azimuths, colatitudes, _ = core.cart2sph(*vertices.T)
        ishtmtx, _ = core.ishtmtx(azimuths, colatitudes, fod.shape[-1])

        # Importance sample the ODFs based on the discretization.
        orientations = []
        for index in indices:
            voxel = voxels[index]
            local = fod.get_fdata()[tuple(voxel)]
            values = np.dot(ishtmtx, local)
            cumsum = np.cumsum(np.maximum(values, 0.0))
            i = np.searchsorted(cumsum, np.random.rand() * cumsum[-1])
            orientations.append(vertices[i])

        return _from_components(locations, orientations)


def save(filename: Path, seeds: npt.ArrayLike):
    """Save an ``(n_seeds, 8)`` seed array to file.

    Saves the seeds to a file which can either be a text file
    or a tractogram .tck.

    Args:
        filename: The name of the file to save. The extension
            decides the file type.
        seeds: Seed array with homogeneous positions and directions.

    """

    seeds = validate_seed_array(seeds)

    match filename.suffix:
        case ".txt":
            data = np.hstack((seeds[:, :3], seeds[:, 4:7]))
            np.savetxt(filename, data)
        case ".tck":
            data = [
                np.vstack((seed[:3], seed[:3] + seed[4:7]))
                for seed in seeds
            ]
            tractogram = nib.streamlines.Tractogram(data, affine_to_rasmm=np.eye(4))
            nib.streamlines.TckFile(tractogram).save(filename)
        case _:
            raise ValueError(f"Unknown file type {filename.suffix}")


def load(filename: Path) -> npt.NDArray[np.float32]:
    """Load seeds from a file into the canonical ``(n_seeds, 8)`` array.

    Loads the seeds from a file which can either be a text file
    or a tractogram .tck.

    Args:
        filename: The name of the file to load.

    Return
        seeds: Loaded seed array.

    """

    match filename.suffix:
        case ".txt":
            data = np.loadtxt(filename, dtype=np.float32, ndmin=2)
            if data.size == 0:
                return np.empty((0, 8), dtype=np.float32)
            if data.shape[1] != 6:
                raise ValueError("Text seed files must contain six columns.")
            return _from_components(data[:, :3], data[:, 3:6])
        case ".tck":
            streamlines = nib.streamlines.load(filename).streamlines
            locations = np.asarray([s[0] for s in streamlines], dtype=np.float32).reshape(-1, 3)
            orientations = np.asarray(
                [s[1] - s[0] for s in streamlines], dtype=np.float32
            ).reshape(-1, 3)
            return _from_components(locations, orientations)
        case _:
            raise ValueError(f"Unknown file type {filename.suffix}")


def _triangle_normals(vs, ts):
    """Compute the normal of triangles"""
    t = vs[ts]
    ns = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0], axis=1)
    norms = np.linalg.norm(ns, axis=1)
    ns[norms != 0] /= norms[norms != 0, None]
    return ns


def _random_barycentric_coordinates(n):
    """Generate n random points uniformly in a triangle"""
    a, b = np.random.rand(2, n)
    to_update = a + b > 1
    a[to_update] = 1 - a[to_update]
    b[to_update] = 1 - b[to_update]
    return np.c_[1 - a - b, a, b]


def _sample_cone(rng, orientations, angle):
    """Uniformly sample a cone around an orientation"""

    samples = np.zeros((len(orientations), 3))
    valid = np.sum(samples * orientations, axis=1) >= np.cos(np.deg2rad(angle))
    while not np.all(valid):
        samples[~valid] = _sample_sphere(rng, np.sum(~valid))
        valid = np.sum(samples * orientations, axis=1) >= np.cos(np.deg2rad(angle))

    return samples


def _sample_sphere(rng, n_samples):
    """Generate uniform samples on the sphere"""
    u = rng.uniform(size=(2, n_samples))
    phi = 2 * np.pi * u[0]
    theta = np.arccos(1 - 2 * u[1])
    return np.vstack(
        [np.sin(theta) * np.sin(phi), np.sin(theta) * np.cos(phi), np.cos(theta)]
    ).T
