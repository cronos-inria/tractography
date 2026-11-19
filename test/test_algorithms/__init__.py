import tractography as tg


def run_configured_tractogram(field, seeds, config):
	"""Run the public tractogram API with parameters from a legacy config."""
	return tg.tractogram(
		field,
		seeds,
		algorithm=tg._algorithm_from_config(field, config),
		save_at=config.save_at,
		maximum_streamline_length=config.streamline.length.maximum,
		# The old low-level implementation returned every nonempty track.
		# Keep that behavior in tests that assert one result per seed.
		minimum_streamline_length=0.0,
		batch_size=config.batch_size,
	)
