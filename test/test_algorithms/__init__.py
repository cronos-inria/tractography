import numpy as np

import tractography as tg


def make_algorithm(field, algorithm_type, **options):
	"""Construct an algorithm using the former test-suite defaults."""
	model_type = tg.algorithms.core.LocalModel.from_shape(field.shape)
	if model_type == tg.algorithms.core.LocalModel.DTI:
		local_model = tg.local.DiffusionTensorImaging()
	else:
		local_model = tg.local.RealSymmetricSphericalHarmonics(field.shape[-1])

	defaults = {
		tg.algorithm.Transport: {"step_size": 0.001, "gamma": 0.5},
		tg.algorithm.Deterministic: {
			"step_size": 0.25,
			"max_step_size": np.deg2rad(30.0),
		},
		tg.algorithm.Diffusion: {
			"step_size": 0.001,
			"gamma": 0.5,
			"noise_variance": 2.0,
		},
		tg.algorithm.Probabilistic: {
			"step_size": 0.25,
			"max_angle": np.deg2rad(20.0),
		},
	}
	algorithm_options = defaults[algorithm_type].copy()
	algorithm_options.update(options)
	return algorithm_type(local_model, **algorithm_options)


def run_tractogram(
	field,
	seeds,
	algorithm_type,
	*,
	algorithm_options=None,
	save_at=0.25,
	maximum_streamline_length=300.0,
	minimum_streamline_length=0.0,
	batch_size=50000,
):
	"""Run the public tractogram API with an explicitly selected algorithm."""
	algorithm = make_algorithm(field, algorithm_type, **(algorithm_options or {}))
	return tg.tractogram(
		field,
		seeds,
		algorithm=algorithm,
		save_at=save_at,
		maximum_streamline_length=maximum_streamline_length,
		minimum_streamline_length=minimum_streamline_length,
		batch_size=batch_size,
	)
