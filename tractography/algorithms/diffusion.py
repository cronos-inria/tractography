import numpy as np
import pydantic
import trimesh

import tractography as tg
from . import opencl as cl
from .configuration import Algorithm, BaseConfiguration


class Configuration(BaseConfiguration):
    inverse_curvature: pydantic.PositiveFloat
    noise_variance: pydantic.PositiveFloat

    @property
    def implementation(self):
        return Diffusion

    @classmethod
    def load(cls):
        return super().load(Algorithm.DIFFUSION)


def histogram(fod, fod_affine, seed_fod, seed_fod_affine, n_seeds, config):
    """Generate a streamline histogram

    The histogram correponds to the FOD associated to a particular tracgogram. That is,
    the distribution of streamline orientations, for each voxel. This function
    generates the histogram directly, without saving the intermediate streamlines and
    therefore allows a much larger number of seeds to be used.

    """

    # Determine the number of seeds per thread. We want 1k threads.
    n_threads = 1000
    n_seeds_per_thread = np.ceil(n_seeds / n_threads)

    # Prepare the data necessary for the seeds.
    mask = seed_fod[..., 0] > 0
    voxels = np.array(np.nonzero(mask), dtype=np.float32).T
    voxels = np.hstack((voxels, np.ones((len(voxels), 1))))
    inline_fod = fod[mask]
    randoms = np.random.randint(4294967295, size=(n_threads, 2))

    # Send the data to the device.
    fod_buffer = cl.new_read_only_buffer(fod.astype(np.float32))
    fod_inverse_affine_buffer = cl.new_read_only_buffer(np.linalg.inv(fod_affine).astype(np.float32))
    seed_fod_buffer = cl.new_read_only_buffer(inline_fod.astype(np.float32))
    seed_fod_voxels_buffer = cl.new_read_only_buffer(voxels.astype(np.float32))
    seed_fod_affine_buffer = cl.new_read_only_buffer(seed_fod_affine.astype(np.float32))
    randoms_buffer = cl.new_buffer(randoms.astype(np.uint32))

    hist = np.zeros(fod.shape, dtype=np.float32)
    hist_buffer = cl.new_buffer(hist)

    # Set constants in the OpenCL code.
    values = {
        "nx": fod.shape[0],
        "ny": fod.shape[1],
        "nz": fod.shape[2],
        "nnz": len(voxels),
        "n_coefficients": fod.shape[-1],
        "n_steps": config.n_steps,
        "n_seeds": n_threads,
    }
    program = cl.build_program(values, ["utils/seeds.cl", "diffusion/histogram.cl"])

    args = (
        fod_buffer,
        fod_inverse_affine_buffer,
        seed_fod_buffer,
        seed_fod_voxels_buffer,
        seed_fod_affine_buffer,
        randoms_buffer,
        np.float32(config.step_size),
        np.float32(config.save_at),
        np.float32(config.inverse_curvature),
        np.float32(config.noise_variance),
        np.uint32(n_seeds_per_thread),
        hist_buffer,
    )
    cl.run_histogram(program, args, n_threads)
    cl.copy_from_buffer(hist_buffer, hist)

    hist = tg.utils.normalize_odf(hist)
    return hist


class Diffusion:

    def __init__(self, odf, affine, n_streamlines, config):

        self._odf_shape = odf.shape
        self._n_streamlines = n_streamlines
        self._config = config

        # Precompute the inverse affine.
        iaffine = np.linalg.inv(affine).astype(np.float32)
        self._iaffine = cl.new_read_only_buffer(iaffine)

        # Check the shape of the fODF data. For now, only 45 coefficients are
        # supported.
        if odf.shape[-1] != 45:
            raise ValueError(
                "For now, only fODF with 45 coefficients are supported (lmax=8)."
            )
        self._odf = cl.new_read_only_buffer(odf.astype(np.float32))

        # Add the random number states.
        randoms_array = np.random.randint(4294967295, size=(n_streamlines, 2)).astype(
            np.uint32
        )
        self._randoms = cl.new_read_only_buffer(randoms_array)

        # Create the seed buffer on the device. They are stored as two float4.
        seeds_array = np.empty((n_streamlines, 8), dtype=np.float32)
        self._seeds = cl.new_read_only_buffer(seeds_array)

        # Reserve space for the streamlines on the device. The are
        # stored as float4.
        streamlines_nbytes = n_streamlines * config.n_steps * 4 * 4
        self._streamlines = cl.new_write_only_buffer(streamlines_nbytes)

        # Reserve space for the length of the streamlines on the device.
        self._lengths = cl.new_write_only_buffer(n_streamlines * 4)

        # Set constants in the OpenCL code.
        values = {
            "nx": odf.shape[0],
            "ny": odf.shape[1],
            "nz": odf.shape[2],
            "n_coefficients": odf.shape[-1],
            "n_steps": config.n_steps,
            "n_streamlines": n_streamlines,
        }
        self._program = cl.build_program(values, "diffusion.cl")

    def histogram(self, seeds):

        # Transfer the seeds to the buffer.
        array = tg.seeds.to_array(seeds).astype(np.float32)
        array = np.c_[array[:, :3], self._ones, array[:, 3:], self._zeros]
        cl.copy_to_buffer(self._seeds, array)

        # Reserve space on the device for the histogram.
        hist_nbytes = np.prod(self._odf_shape[:3]) * 162 * 4
        hist_buffer = cl.new_write_only_buffer(hist_nbytes)

        # Reserve space for the bin centers.
        mesh = trimesh.creation.icosphere(2)
        bin_areas = np.zeros(len(mesh.vertices))
        for i, indices in enumerate(mesh.vertex_faces):
            for index in indices:
                if index != -1:
                    bin_areas[i] += mesh.area_faces[index] / 3
        bin_centers = mesh.vertices / np.linalg.norm(
            mesh.vertices, axis=1, keepdims=True
        )
        bin_centers = np.c_[bin_centers, np.zeros((162, 1))].astype(np.float32)
        bin_centers_buffer = cl.new_read_only_buffer(bin_centers)

        args = (
            self._odf,
            self._iaffine,
            self._seeds,
            self._randoms,
            bin_centers_buffer,
            np.float32(self._config.step_size),
            np.float32(self._config.save_at),
            np.float32(self._config.inverse_curvature),
            hist_buffer,
        )
        cl.run_histogram(self._program, args, self._n_streamlines)
        hist = np.zeros(self._odf_shape[:3] + (162,), dtype=np.uint32)
        cl.copy_from_buffer(hist_buffer, hist)

        # Normalize the histogram.
        hist = hist / np.expand_dims(bin_areas, axis=(0, 1, 2))
        hist_norm = np.linalg.norm(hist, axis=-1, keepdims=True)
        np.divide(hist, hist_norm, where=hist_norm != 0, out=hist)

        return hist, bin_centers[:, :3]

    def run(self, seeds):

        # Transfer the seeds to the buffer.
        array = tg.seeds.to_array(seeds).astype(np.float32)
        cl.copy_to_buffer(self._seeds, array)

        # Track streamlines.
        args = (
            self._odf,
            self._iaffine,
            self._seeds,
            self._randoms,
            np.float32(self._config.step_size),
            np.float32(self._config.save_at),
            np.float32(self._config.inverse_curvature),
            np.float32(self._config.noise_variance),
            self._streamlines,
            self._lengths,
        )
        cl.run_program(self._program, args, self._n_streamlines)
        streamlines = np.zeros(
            (self._n_streamlines, self._config.n_steps, 4), dtype=np.float32
        )
        cl.copy_from_buffer(self._streamlines, streamlines)
        lengths = np.zeros((self._n_streamlines,), dtype=np.uint32)
        cl.copy_from_buffer(self._lengths, lengths)

        return [streamlines[i, :n, :3] for i, n in enumerate(lengths)]
