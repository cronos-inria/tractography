import unittest
from pathlib import Path
from typing import cast
from unittest.mock import patch

import nibabel as nib
import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

import tractography as tg
import test
import test.data
from test.test_algorithms import make_algorithm

_TEST_RESULTS_DIR = (
    Path(__file__).parents[1] / "test-results" / "tractography" / "connectome"
)
_TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

class TestTOD(unittest.TestCase):
    """Test the tg.tod function"""

    def test_tod_with_explicit_domain_and_algorithm(self):
        """Test the simplest use case"""

        fod, seed_mask, _ = test.data.cross()
        n_seeds = 32
        tracking_algorithm = make_algorithm(fod, tg.algorithm.Diffusion)
        tod = tg.tod(
            fod,
            n_seeds,
            seed_mask,
            tracking_algorithm,
            maximum_streamline_length=20.0,
        )
        tod_data = tod.get_fdata()
        self.assertEqual(tod.shape, fod.shape[:3] + (45,))
        self.assertTrue(np.isfinite(tod_data).all())
        populated = tod_data[..., 0] != 0
        self.assertTrue(np.any(populated))
        np.testing.assert_allclose(
            tod_data[..., 0][populated], 1.0 / np.sqrt(4.0 * np.pi)
        )

    def test_tod_from_image_field_uses_defaults(self):
        fod, _, _ = test.data.cross()
        result = tg.tod(
            tg.field.Image(fod), n_seeds=16, maximum_streamline_length=2.0
        )
        self.assertEqual(result.shape, fod.shape[:3] + (45,))


class TestTractography(unittest.TestCase):
    """Test the tg.tractogram function"""

    def test_image_default_algorithm_and_boundary_seeds(self):
        fod, _, _ = test.data.cross()
        result = tg.tractogram(
            fod,
            8,
            maximum_streamline_length=2.0,
        )
        self.assertEqual(len(result), 8)

    def test_field_and_explicit_domain(self):
        fod, seed_mask, _ = test.data.cross()
        result = tg.tractogram(
            tg.field.Image(fod),
            8,
            tg.domain.Image(seed_mask),
            maximum_streamline_length=2.0,
        )
        self.assertEqual(len(result), 8)

    def test_simple(self):
        """Test the simplest use case"""

        fod, _, _ = test.data.cross()
        n_seeds = 100  # The number of seeds can be smaller than the batch size.
        seeds = cast(list[tg.seeds.Seed], tg.seeds.from_fod(fod, n_seeds))
        tracking_algorithm = make_algorithm(fod, tg.algorithm.Transport)
        streamlines = tg.tractogram(
            fod,
            seeds,
            None,
            tracking_algorithm,
            save_at=0.25,
            maximum_streamline_length=300.0,
            minimum_streamline_length=10.0,
            batch_size=50000,
        )
        self.assertLessEqual(len(streamlines), n_seeds)


class TestConnectome(unittest.TestCase):
    """Test the tg.connectome function"""

    def test_without_segmentation_preserves_directionality(self):
        fod, _, _ = test.data.cross()
        n_vertices = len(tg.domain.Image(
            nib.Nifti1Image(
                np.any(np.asanyarray(fod.dataobj) != 0, axis=-1).astype(np.uint8),
                fod.affine,
            )
        ).boundary.vertices)
        directed = csr_matrix(
            ([1.0], ([0], [1])), shape=(n_vertices, n_vertices)
        )

        with patch.object(tg.tracking, "connectome", return_value=directed):
            matrix, labels = tg.connectome(fod, n_seeds=1)

        self.assertIs(matrix, directed)
        self.assertEqual(matrix[0, 1], 1.0)
        self.assertEqual(matrix[1, 0], 0.0)
        np.testing.assert_array_equal(labels, np.arange(n_vertices))

    def test_segmentation_aggregation_preserves_directionality(self):
        fod = nib.Nifti1Image(np.ones((2, 1, 1, 6), dtype=np.float32), np.eye(4))
        segmentation_data = np.array([[[1]], [[2]]], dtype=np.int16)
        segmentation = nib.Nifti1Image(segmentation_data, np.eye(4))
        domain = tg.domain.Image(
            nib.Nifti1Image(np.ones((2, 1, 1), dtype=np.uint8), np.eye(4))
        )
        voxel_indices = np.argwhere(segmentation_data > 0)
        voxel_centers = nib.affines.apply_affine(
            segmentation.affine, voxel_indices
        )
        nearest = cKDTree(voxel_centers).query(domain.boundary.vertices)[1]
        vertex_labels = segmentation_data[tuple(voxel_indices[nearest].T)]
        first = np.flatnonzero(vertex_labels == 1)[0]
        second = np.flatnonzero(vertex_labels == 2)[0]
        directed = csr_matrix(
            ([1.0], ([first], [second])),
            shape=(len(vertex_labels), len(vertex_labels)),
        )

        with patch.object(tg.tracking, "connectome", return_value=directed):
            matrix, labels = tg.connectome(
                fod, n_seeds=1, segmentation=segmentation
            )

        np.testing.assert_array_equal(labels, [1, 2])
        self.assertEqual(matrix[0, 1], 1.0)
        self.assertEqual(matrix[1, 0], 0.0)

    def test_without_segmentation_returns_boundary_connectome(self):
        fod, _, _ = test.data.cross()
        matrix, labels = tg.connectome(
            fod,
            n_seeds=32,
            maximum_streamline_length=2.0,
        )

        self.assertIsInstance(matrix, csr_matrix)
        self.assertEqual(matrix.shape, (len(labels), len(labels)))
        np.testing.assert_array_equal(labels, np.arange(len(labels)))

    def test_cross_deterministic(self):
        """Test that the connectivity matrix from the deterministic algorithm has the expected shape and labels"""

        fod, wm, segmentation = test.data.cross()
        nib.save(fod, _TEST_RESULTS_DIR / "cross-fod.nii.gz")
        nib.save(wm, _TEST_RESULTS_DIR / "cross-wm.nii.gz")
        nib.save(segmentation, _TEST_RESULTS_DIR / "cross-segmentation.nii.gz")
        fod = tg.nifti.multiply(fod, wm)
        n_seeds = 10000
        tracking_algorithm = make_algorithm(fod, tg.algorithm.Deterministic)
        matrix, labels = tg.connectome(
            fod,
            n_seeds=n_seeds,
            algorithm=tracking_algorithm,
            segmentation=segmentation,
            maximum_streamline_length=300.0,
            minimum_streamline_length=5.0,
        )

        # The crossing dataset has labels 1, 2, 3, 4.
        np.testing.assert_array_equal(labels, [1, 2, 3, 4])

        # The matrix should be square with one row/column per label.
        self.assertEqual(matrix.shape, (4, 4))

        # The diagonal should be zero or very small (streamlines connecting a
        # region to itself are uncommon in a crossing geometry).
        self.assertTrue(np.all(matrix.diagonal() < 0.1 * matrix.sum()))

        # The total number of connections should be positive, meaning some
        # streamlines did reach labeled regions.
        self.assertGreater(matrix.sum(), 0)

        # Region 1 should be connected mostly to 2. Region 3 should be connected mostly to 4.
        self.assertTrue(np.argmax(matrix[0]) == 1)
        self.assertTrue(np.argmax(matrix[2]) == 3)

    def test_cross_probabilistic(self):
        """Test that the connectivity matrix from the probabilistic algorithm has the expected shape and labels"""

        fod, wm, segmentation = test.data.cross()
        nib.save(fod, _TEST_RESULTS_DIR / "cross-fod.nii.gz")
        nib.save(wm, _TEST_RESULTS_DIR / "cross-wm.nii.gz")
        nib.save(segmentation, _TEST_RESULTS_DIR / "cross-segmentation.nii.gz")
        fod = tg.nifti.multiply(fod, wm)
        n_seeds = 10000
        tracking_algorithm = make_algorithm(fod, tg.algorithm.Probabilistic)
        matrix, labels = tg.connectome(
            fod,
            n_seeds=n_seeds,
            algorithm=tracking_algorithm,
            segmentation=segmentation,
            maximum_streamline_length=300.0,
            minimum_streamline_length=5.0,
        )

        # The crossing dataset has labels 1, 2, 3, 4.
        np.testing.assert_array_equal(labels, [1, 2, 3, 4])

        # The matrix should be square with one row/column per label.
        self.assertEqual(matrix.shape, (4, 4))

        # The diagonal should be zero or very small (streamlines connecting a
        # region to itself are uncommon in a crossing geometry).
        self.assertTrue(np.all(matrix.diagonal() < 0.1 * matrix.sum()))

        # The total number of connections should be positive, meaning some
        # streamlines did reach labeled regions.
        self.assertGreater(matrix.sum(), 0)

        # Region 1 should be connected mostly to 2. Region 3 should be connected mostly to 4.
        self.assertTrue(np.argmax(matrix[0]) == 1)
        self.assertTrue(np.argmax(matrix[2]) == 3)

    def test_cross_diffusion(self):
        """Test that the connectivity matrix from the diffusion algorithm has the expected shape and labels"""

        fod, wm, segmentation = test.data.cross()
        fod = tg.nifti.multiply(fod, wm)
        n_seeds = 10000
        tracking_algorithm = make_algorithm(
            fod, tg.algorithm.Diffusion, noise_variance=0.01
        )
        matrix, labels = tg.connectome(
            fod,
            n_seeds=n_seeds,
            algorithm=tracking_algorithm,
            segmentation=segmentation,
            maximum_streamline_length=300.0,
            minimum_streamline_length=5.0,
        )

        np.testing.assert_array_equal(labels, [1, 2, 3, 4])
        self.assertEqual(matrix.shape, (4, 4))
        self.assertTrue(np.all(matrix.diagonal() < 0.1 * matrix.sum()))
        self.assertGreater(matrix.sum(), 0)
        self.assertTrue(np.argmax(matrix[0]) == 1)
        self.assertTrue(np.argmax(matrix[2]) == 3)

    def test_cross_transport(self):
        """Test that the connectivity matrix from the transport algorithm has the expected shape and labels"""

        fod, wm, segmentation = test.data.cross()
        fod = tg.nifti.multiply(fod, wm)
        n_seeds = 10000
        tracking_algorithm = make_algorithm(fod, tg.algorithm.Transport)
        matrix, labels = tg.connectome(
            fod,
            n_seeds=n_seeds,
            algorithm=tracking_algorithm,
            segmentation=segmentation,
            maximum_streamline_length=300.0,
            minimum_streamline_length=5.0,
        )

        np.testing.assert_array_equal(labels, [1, 2, 3, 4])
        self.assertEqual(matrix.shape, (4, 4))
        self.assertTrue(np.all(matrix.diagonal() < 0.1 * matrix.sum()))
        self.assertGreater(matrix.sum(), 0)
        self.assertTrue(np.argmax(matrix[0]) == 1)
        self.assertTrue(np.argmax(matrix[2]) == 3)
