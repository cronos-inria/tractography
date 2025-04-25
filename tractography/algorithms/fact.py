import numpy as np

import tractography as tg
from . import opencl as cl


class FACT:
    """OpenCL-accelerated implementation of the FACT algorithm

    The FACT (Fiber Assignment by Continuous Tracking) algorithm
    deterministically traces white matter pathways by propagating streamlines
    along the most colinear fiber orientation (peak direction) at each voxel.
    This implementation leverages GPU acceleration using OpenCL for efficient,
    large-scale tractography.

    """

    def __init__(self, peaks, affine, n_streamlines, config):
        """Initializes the FACT algorithm with tracking data and parameters

        Prepares OpenCL buffers for peaks, seed points, output streamlines,
        and tracking parameters. Normalizes input peak vectors, reshapes them
        for OpenCL compatibility, and allocates memory for results on the device.

        Args:
            peaks: 4D array of peak directions with shape (X, Y, Z, N*3),
                where N is the number of peaks per voxel.
            affine: 4x4 affine matrix mapping voxel indices to world coordinates.
            n_streamlines: Number of streamlines to generate when running the
                algorithm.
            config: Configuration object containing tracking parameters (e.g.,
                step size, number of steps).

        """

        self._config = config
        self._n_streamlines = n_streamlines

        # Normalize the peaks.
        n_peaks = peaks.shape[-1] // 3
        peaks = peaks.reshape(peaks.shape[:3] + (3, n_peaks), order="F")
        peaks = np.transpose(peaks, (0, 1, 2, 4, 3))
        norms = np.linalg.norm(peaks, axis=-1, keepdims=True)
        peaks = np.divide(peaks, norms, where=norms != 0)

        # Precompute the inverse affine.
        iaffine = np.linalg.inv(affine).astype(np.float32)
        self._iaffine = cl.new_read_only_buffer(iaffine)

        # Augment the vectors to 4 elements because OpenCL only has float4.
        peaks = np.pad(peaks, ((0, 0),) * 4 + ((0, 1),)).astype(np.float32)
        self._peaks = cl.new_read_only_buffer(peaks)

        # Also augment the seeds array to two float4 per seed.
        seeds_array = np.ones((n_streamlines, 8), dtype=np.float32)
        self._seeds = cl.new_read_only_buffer(seeds_array)

        # Reserve space for the streamlines on the device.
        # Streamlines are also float4.
        streamlines_nbytes = n_streamlines * config.n_steps * 4 * 4
        self._streamlines = cl.new_write_only_buffer(streamlines_nbytes)

        # Reserve space for the length of the streamlines on the device.
        self._lengths = cl.new_write_only_buffer(n_streamlines * 4)

        # Fill columns for seeds.
        self._zeros = np.zeros(self._n_streamlines, dtype=np.float32)
        self._ones = np.ones(self._n_streamlines, dtype=np.float32)

        # Set constants in the OpenCL code.
        values = {
            "nx": peaks.shape[0],
            "ny": peaks.shape[1],
            "nz": peaks.shape[2],
            "n_peaks": n_peaks,
            "n_steps": config.n_steps,
            "n_streamlines": n_streamlines,
        }
        self._program = cl.build_program(values, "fact.cl")

    def run(self, seeds):
        """Run the FACT algorithm on the given seed points

        Args:
            seeds: List of seed points in world coordinates with associated
                direction vectors.

        Returns:
            A list of streamlines, each represented as a 2D array of 3D
                coordinates (N x 3), where N is the number of steps
                successfully tracked for that streamline.

        """

        # Transfer the seeds to the buffer.
        array = tg.seeds.to_array(seeds).astype(np.float32)
        array = np.c_[array[:, :3], self._ones, array[:, 3:], self._zeros]
        cl.copy_to_buffer(self._seeds, array)

        # Track streamlines.
        max_angle = self._config.algorithms.fact.maximum_angle
        args = (
            self._peaks,
            self._iaffine,
            self._seeds,
            np.float32(self._config.step_size),
            np.float32(np.cos(np.deg2rad(max_angle))),
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
