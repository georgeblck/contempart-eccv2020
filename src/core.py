"""Shared functions for the contempArt reproduction pipeline.

Provides reusable building blocks for loading data, computing
embeddings, and aggregating per-artist features.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine, pdist, squareform
from scipy.stats import spearmanr


def load_manifest(path: Path, sep: str = "\t") -> pd.DataFrame:
    """Load the image manifest CSV."""
    return pd.read_csv(path, sep=sep)


def load_artist_metadata(path: Path, sep: str = ";") -> pd.DataFrame:
    """Load the artist metadata CSV."""
    return pd.read_csv(path, sep=sep)


def compute_artist_centroids(
    embeddings: np.ndarray,
    manifest: pd.DataFrame,
    artist_ids: list[str],
    label_col: str = "labelsCat",
) -> np.ndarray:
    """Compute per-artist centroid (mean embedding).

    Paper Eq. 7: c^l = (1/N^l) sum_i e^l_i
    """
    centroids = []
    for artist in artist_ids:
        mask = manifest[label_col] == artist
        if mask.sum() > 0:
            centroids.append(embeddings[np.asarray(mask)].mean(axis=0))
        else:
            centroids.append(np.zeros(embeddings.shape[1]))
    return np.array(centroids)


def compute_sigma_c(
    embeddings: np.ndarray,
    manifest: pd.DataFrame,
    label_col: str = "labelsCat",
) -> tuple[float, float]:
    """Compute local style variance (paper Eq. 8).

    sigma_c = mean across artists of (mean cosine distance to centroid).
    Returns (mean, std) across artists.

    Artists with only one image are excluded (their distance to centroid
    is trivially 0, and the original R code sets them to NA).
    """
    artists = manifest[label_col].unique()
    artist_sigmas: list[float] = []

    for artist in artists:
        mask = manifest[label_col] == artist
        artist_emb = embeddings[np.asarray(mask)]
        if len(artist_emb) <= 1:
            continue
        centroid = artist_emb.mean(axis=0)
        dists = np.array([cosine(centroid, e) for e in artist_emb])
        artist_sigmas.append(float(dists.mean()))

    return float(np.mean(artist_sigmas)), float(np.std(artist_sigmas))


def compute_sigma_c_global(
    embeddings: np.ndarray,
) -> tuple[float, float]:
    """Compute global style variance (paper Eq. 9).

    sigma_c_global = mean cosine distance of all images to the global centroid.
    Returns (mean, std) across all images.
    """
    c_global = embeddings.mean(axis=0)
    dists = np.array([cosine(c_global, e) for e in embeddings])
    return float(np.mean(dists)), float(np.std(dists))


def cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine distance matrix."""
    return np.asarray(squareform(pdist(embeddings, metric="cosine")), dtype=np.float64)


def spearman_upper_triangle(
    dist_a: np.ndarray, dist_b: np.ndarray
) -> tuple[float, float]:
    """Spearman rho on the upper triangle of two distance matrices.

    Matches the original paper's methodology (p.13):
    'Spearman's rank coefficient is used to compute the correlation
     between the flattened upper triangular parts'
    """
    idx = np.triu_indices(dist_a.shape[0], k=1)
    rho, p = spearmanr(dist_a[idx], dist_b[idx])
    return float(rho), float(p)


def purity_score(y_true: list[int], y_pred: np.ndarray) -> float:
    """Purity: fraction correctly assigned by majority vote per cluster."""
    from sklearn.metrics.cluster import contingency_matrix

    cm = contingency_matrix(y_true, y_pred)
    return float(np.sum(np.amax(cm, axis=0)) / np.sum(cm))
