from nibabel.nifti1 import Nifti1Image
import numpy as np
from scipy.sparse import csr_matrix

import tractography as tg
from tractography.tracking import tractogram
from tractography.cli.connectome import _aggregate_by_segmentation
from test import data


def test_tractogram_empty_seeds_returns_empty_tractogram():
	"""Empty input returns an empty tractogram without running OpenCL."""
	domain = tg.domain.Image(Nifti1Image(np.ones((1, 1, 1)), np.eye(4)))
	field = tg.field.Image(
		Nifti1Image(np.ones((1, 1, 1, 1)), np.eye(4))
	)
	algorithm = tg.algorithm.Transport(tg.local.RealSymmetricSphericalHarmonics(1))
	result = tractogram(
		algorithm,
		domain,
		field,
		np.empty((0, 6), dtype=np.float32),
	)

	assert len(result.streamlines) == 0
	np.testing.assert_array_equal(result.affine_to_rasmm, np.eye(4))


def test_tractogram_tracks_seed_until_leaving_domain():
	"""Track one straight streamline and preserve its RAS+ coordinates."""
	domain_image = Nifti1Image(
		np.ones((1, 1, 1), dtype=np.uint8), np.eye(4)
	)
	domain = tg.domain.Image(domain_image)

	field_data = np.zeros((1, 1, 1, 15), dtype=np.float32)
	field_data[..., 0] = 1.0
	field_image = Nifti1Image(field_data, np.eye(4))
	field = tg.field.Image(field_image)
	local_model = tg.local.RealSymmetricSphericalHarmonics(15)
	algorithm = tg.algorithm.Transport(
		local_model, step_size=1.0, gamma=0.0
	)
	seeds = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32)

	result = tractogram(algorithm, domain, field, seeds)

	assert len(result.streamlines) == 1
	np.testing.assert_allclose(
		np.asarray(result.streamlines[0]),
		[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
		atol=1e-6,
	)
	np.testing.assert_array_equal(result.affine_to_rasmm, np.eye(4))


def test_tractogram_rejects_incompatible_local_model_and_field():
	"""Reject fields whose parameter count differs from the local model."""
	domain = tg.domain.Image(Nifti1Image(np.ones((1, 1, 1)), np.eye(4)))
	field = tg.field.Image(Nifti1Image(np.ones((1, 1, 1, 1)), np.eye(4)))
	algorithm = tg.algorithm.Transport(tg.local.RealSymmetricSphericalHarmonics(15))

	with np.testing.assert_raises_regex(ValueError, "15 != 1"):
		tractogram(
			algorithm,
			domain,
			field,
			np.empty((0, 6), dtype=np.float32),
		)


def test_tractogram_respects_save_interval_and_maximum_length():
	"""Sample at save_at intervals and stop when the point buffer is full."""
	domain = tg.domain.Image(
		Nifti1Image(np.ones((20, 1, 1), dtype=np.uint8), np.eye(4))
	)
	field_data = np.zeros((20, 1, 1, 15), dtype=np.float32)
	field_data[..., 0] = 1.0
	field = tg.field.Image(Nifti1Image(field_data, np.eye(4)))
	algorithm = tg.algorithm.Transport(
		tg.local.RealSymmetricSphericalHarmonics(15),
		step_size=0.25,
		gamma=0.0,
	)
	seeds = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32)

	result = tractogram(
		algorithm,
		domain,
		field,
		seeds,
		save_at=0.5,
		maximum_streamline_length=3.0,
	)

	streamline = np.asarray(result.streamlines[0])
	assert len(streamline) == 6
	np.testing.assert_allclose(
		streamline[:, 0], np.arange(0.0, 3.0, 0.5), atol=1e-6
	)
	np.testing.assert_allclose(streamline[:, 1:], 0.0, atol=1e-6)


def test_tractogram_accepts_diffusion_algorithm():
	"""Run the stochastic algorithm through the shared tracking kernel."""
	domain = tg.domain.Image(
		Nifti1Image(np.ones((1, 1, 1), dtype=np.uint8), np.eye(4))
	)
	field_data = np.zeros((1, 1, 1, 15), dtype=np.float32)
	field_data[..., 0] = 1.0
	field = tg.field.Image(Nifti1Image(field_data, np.eye(4)))
	algorithm = tg.algorithm.Diffusion(
		tg.local.RealSymmetricSphericalHarmonics(15),
		step_size=1.0,
		gamma=0.0,
		noise_variance=0.0,
		seed=1234,
	)
	seeds = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32)

	result = tractogram(algorithm, domain, field, seeds)

	assert len(result.streamlines) == 1
	np.testing.assert_allclose(
		np.asarray(result.streamlines[0]),
		[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
		atol=1e-6,
	)


def test_connectome_returns_piecewise_linear_boundary_matrix():
	"""The kernel deposits barycentric endpoint weights into a directed CSR."""
	domain = tg.domain.Image(
		Nifti1Image(np.ones((1, 1, 1), dtype=np.uint8), np.eye(4))
	)
	field_data = np.zeros((1, 1, 1, 15), dtype=np.float32)
	field_data[..., 0] = 1.0
	field = tg.field.Image(Nifti1Image(field_data, np.eye(4)))
	algorithm = tg.algorithm.Transport(
		tg.local.RealSymmetricSphericalHarmonics(15),
		step_size=0.1,
		gamma=0.0,
	)

	matrix = tg.tracking.connectome(
		algorithm,
		domain,
		field,
		n_seeds=16,
		maximum_streamline_length=2.0,
	)

	assert isinstance(matrix, csr_matrix)
	assert matrix.shape == (len(domain.boundary.vertices),) * 2
	assert matrix.nnz > 0
	assert matrix.dtype == np.float32
	assert 0.0 < matrix.sum() <= 16.0 + 1e-5


def test_connectome_zero_seeds_returns_empty_csr_without_opencl():
	"""An empty seed request returns an empty sparse matrix immediately."""
	domain = tg.domain.Image(
		Nifti1Image(np.ones((1, 1, 1), dtype=np.uint8), np.eye(4))
	)
	field = tg.field.Image(Nifti1Image(np.zeros((1, 1, 1, 15)), np.eye(4)))
	algorithm = tg.algorithm.Transport(tg.local.RealSymmetricSphericalHarmonics(15))

	matrix = tg.tracking.connectome(algorithm, domain, field, n_seeds=0)

	assert isinstance(matrix, csr_matrix)
	assert matrix.shape == (len(domain.boundary.vertices),) * 2
	assert matrix.nnz == 0


def test_connectome_cross_dataset_connects_expected_regions():
	"""The cross connectome should link opposite ends of each fiber bundle."""
	fod_image, mask_image, segmentation = data.cross()
	domain = tg.domain.Image(mask_image)
	field = tg.field.Image(fod_image)
	algorithm = tg.algorithm.Transport(
		tg.local.RealSymmetricSphericalHarmonics(fod_image.shape[-1]),
		step_size=0.25,
		gamma=0.0,
	)

	boundary_matrix = tg.tracking.connectome(
		algorithm,
		domain,
		field,
		n_seeds=10000,
		maximum_streamline_length=12.0,
	)
	matrix, labels = _aggregate_by_segmentation(
		boundary_matrix,
		domain.boundary.vertices,
		segmentation,
		assignment_radius=2.0,
	)

	np.testing.assert_array_equal(labels, [1, 2, 3, 4])
	undirected = matrix.toarray() + matrix.toarray().T
	expected_partners = [1, 0, 3, 2]
	for region, partner in enumerate(expected_partners):
		competing_weights = np.delete(undirected[region], [region, partner])
		assert undirected[region, partner] > competing_weights.max()
