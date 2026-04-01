"""Tests for core functions.

Uses small synthetic data to verify correctness without
requiring the full dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core import (
    compute_artist_centroids,
    compute_sigma_c,
    compute_sigma_c_global,
    cosine_distance_matrix,
    purity_score,
    spearman_upper_triangle,
)


@pytest.fixture
def simple_embeddings() -> tuple[np.ndarray, pd.DataFrame]:
    """3 artists, 2 images each, 4-dim embeddings."""
    emb = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],  # artist_a, img 1
            [0.9, 0.1, 0.0, 0.0],  # artist_a, img 2
            [0.0, 1.0, 0.0, 0.0],  # artist_b, img 1
            [0.0, 0.9, 0.1, 0.0],  # artist_b, img 2
            [0.0, 0.0, 1.0, 0.0],  # artist_c, img 1
            [0.0, 0.0, 0.9, 0.1],  # artist_c, img 2
        ]
    )
    manifest = pd.DataFrame({"labelsCat": ["a", "a", "b", "b", "c", "c"]})
    return emb, manifest


class TestComputeArtistCentroids:
    def test_shape(self, simple_embeddings: tuple[np.ndarray, pd.DataFrame]) -> None:
        emb, manifest = simple_embeddings
        centroids = compute_artist_centroids(emb, manifest, ["a", "b", "c"])
        assert centroids.shape == (3, 4)

    def test_values(self, simple_embeddings: tuple[np.ndarray, pd.DataFrame]) -> None:
        emb, manifest = simple_embeddings
        centroids = compute_artist_centroids(emb, manifest, ["a", "b", "c"])
        np.testing.assert_allclose(centroids[0], [0.95, 0.05, 0.0, 0.0])
        np.testing.assert_allclose(centroids[1], [0.0, 0.95, 0.05, 0.0])

    def test_missing_artist(
        self, simple_embeddings: tuple[np.ndarray, pd.DataFrame]
    ) -> None:
        emb, manifest = simple_embeddings
        centroids = compute_artist_centroids(emb, manifest, ["a", "missing"])
        assert centroids.shape == (2, 4)
        np.testing.assert_allclose(centroids[1], [0.0, 0.0, 0.0, 0.0])


class TestSigmaC:
    def test_identical_images(self) -> None:
        """If all images per artist are identical, sigma_c should be 0."""
        emb = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        manifest = pd.DataFrame({"labelsCat": ["a", "a", "b", "b"]})
        mean, _std = compute_sigma_c(emb, manifest)
        assert mean == pytest.approx(0.0, abs=1e-10)

    def test_positive(self, simple_embeddings: tuple[np.ndarray, pd.DataFrame]) -> None:
        emb, manifest = simple_embeddings
        mean, _std = compute_sigma_c(emb, manifest)
        assert mean > 0
        assert _std >= 0

    def test_single_image_artists_excluded(self) -> None:
        """Artists with only 1 image should be excluded from sigma_c."""
        emb = np.array(
            [
                [1.0, 0.0],  # artist_a, img 1
                [0.8, 0.2],  # artist_a, img 2
                [0.0, 1.0],  # artist_b, img 1 (single image)
            ]
        )
        manifest = pd.DataFrame({"labelsCat": ["a", "a", "b"]})
        mean_with_exclusion, _ = compute_sigma_c(emb, manifest)
        # Only artist_a contributes; artist_b (1 image) is excluded
        centroid_a = emb[:2].mean(axis=0)
        from scipy.spatial.distance import cosine

        expected = np.mean([cosine(centroid_a, emb[0]), cosine(centroid_a, emb[1])])
        assert mean_with_exclusion == pytest.approx(expected, abs=1e-10)


class TestSigmaCGlobal:
    def test_identical_images(self) -> None:
        emb = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        mean, _std = compute_sigma_c_global(emb)
        assert mean == pytest.approx(0.0, abs=1e-10)

    def test_positive(self, simple_embeddings: tuple[np.ndarray, pd.DataFrame]) -> None:
        emb, _ = simple_embeddings
        mean, _std = compute_sigma_c_global(emb)
        assert mean > 0


class TestCosineDistanceMatrix:
    def test_shape(self) -> None:
        emb = np.random.randn(10, 5)
        dist = cosine_distance_matrix(emb)
        assert dist.shape == (10, 10)

    def test_diagonal_zero(self) -> None:
        emb = np.random.randn(10, 5)
        dist = cosine_distance_matrix(emb)
        np.testing.assert_allclose(np.diag(dist), 0.0, atol=1e-10)

    def test_symmetric(self) -> None:
        emb = np.random.randn(10, 5)
        dist = cosine_distance_matrix(emb)
        np.testing.assert_allclose(dist, dist.T)


class TestSpearmanUpperTriangle:
    def test_perfect_correlation(self) -> None:
        dist = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]], dtype=float)
        rho, _p = spearman_upper_triangle(dist, dist)
        assert rho == pytest.approx(1.0)

    def test_returns_tuple(self) -> None:
        a = np.random.randn(5, 5)
        a = (a + a.T) / 2
        np.fill_diagonal(a, 0)
        b = np.random.randn(5, 5)
        b = (b + b.T) / 2
        np.fill_diagonal(b, 0)
        rho, _p = spearman_upper_triangle(a, b)
        assert isinstance(rho, float)
        assert isinstance(_p, float)


class TestPurityScore:
    def test_perfect(self) -> None:
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = np.array([0, 0, 1, 1, 2, 2])
        assert purity_score(y_true, y_pred) == pytest.approx(1.0)

    def test_random(self) -> None:
        y_true = [0, 0, 1, 1]
        y_pred = np.array([0, 1, 0, 1])
        score = purity_score(y_true, y_pred)
        assert 0.0 < score <= 1.0
