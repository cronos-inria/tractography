import numpy as np
import pydantic

import tractography as tg
from . import opencl as cl

from .configuration import Algorithm, BaseConfiguration


class Configuration(BaseConfiguration):
    inverse_curvature: pydantic.PositiveFloat

    @property
    def implementation(self):
        return Transport

    @classmethod
    def load(cls):
        return super().load(Algorithm.TRANSPORT)


class Transport:

    def __init__(self, odf, affine, n_streamlines, config):

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

        # Create the seed buffer on the device. They are stored as two float4.
        seeds_array = np.empty((n_streamlines, 8), dtype=np.float32)
        self._seeds = cl.new_read_only_buffer(seeds_array)

        # Reserve space for the streamlines on the device. The are
        # stored as float4.
        streamlines_nbytes = n_streamlines * config.n_steps * 4 * 4
        self._streamlines = cl.new_write_only_buffer(streamlines_nbytes)

        # Reserve space for the length of the streamlines on the device.
        self._lengths = cl.new_write_only_buffer(n_streamlines * 4)

        # Fill columns for seeds.
        self._zeros = np.zeros(self._n_streamlines, dtype=np.float32)
        self._ones = np.ones(self._n_streamlines, dtype=np.float32)

        # Set constants in the OpenCL code.
        values = {
            "nx": odf.shape[0],
            "ny": odf.shape[1],
            "nz": odf.shape[2],
            "n_coefficients": odf.shape[-1],
            "n_steps": config.n_steps,
            "n_streamlines": n_streamlines,
        }
        self._program = cl.build_program(values, "transport.cl")

    def run(self, seeds):

        # Transfer the seeds to the buffer.
        array = tg.seeds.to_array(seeds).astype(np.float32)
        array = np.c_[array[:, :3], self._ones, array[:, 3:], self._zeros]
        cl.copy_to_buffer(self._seeds, array)

        # Track streamlines.
        args = (
            self._odf,
            self._iaffine,
            self._seeds,
            np.float32(self._config.step_size),
            np.float32(self._config.save_at),
            np.float32(self._config.inverse_curvature),
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
