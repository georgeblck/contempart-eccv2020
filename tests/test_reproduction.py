"""Tests that verify reproduced results against the paper.

These tests require the Zenodo data to be available in data/.
They are skipped if the data is not present.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.core import (
    compute_artist_centroids,
    compute_sigma_c_global,
    cosine_distance_matrix,
    spearman_upper_triangle,
)
from src.paths import (
    ARCHETYPE_PATH,
    ARTIST_META_PATH,
    GU_DIST_PATH,
    MANIFEST_PATH,
    SMALL_GRAPH_PATH,
    TEXTURE_EMB_PATH,
    VGG_EMB_PATH,
)

DATA_AVAILABLE = MANIFEST_PATH.exists() and VGG_EMB_PATH.exists()
skip_no_data = pytest.mark.skipif(
    not DATA_AVAILABLE, reason="Zenodo data not available"
)


@skip_no_data
class TestTable2:
    """Verify Table 2: style variance (sigma_c_global)."""

    def test_vgg_sigma_c_global(self) -> None:
        emb = np.load(VGG_EMB_PATH)
        mean, _ = compute_sigma_c_global(emb)
        assert mean == pytest.approx(0.435, abs=0.001)

    def test_texture_sigma_c_global(self) -> None:
        emb = np.load(TEXTURE_EMB_PATH)
        mean, _ = compute_sigma_c_global(emb)
        assert mean == pytest.approx(0.211, abs=0.001)

    def test_archetype_sigma_c_global(self) -> None:
        emb = np.load(ARCHETYPE_PATH)
        mean, _ = compute_sigma_c_global(emb)
        # Wider tolerance: py_pcha produces different archetypes than SPAMS
        assert mean == pytest.approx(0.323, abs=0.05)


@skip_no_data
class TestTable3:
    """Verify Table 3: Spearman rho between style and social distance."""

    @pytest.fixture
    def artist_order(self) -> list[str]:
        metadata = pd.read_csv(ARTIST_META_PATH, sep=";")
        metadata.rename(columns={"instagramHandle.y": "instagramHandleY"}, inplace=True)
        insta_dat = metadata.dropna(subset=["instagramHandleY"])
        insta2id: dict[str, str] = dict(
            zip(insta_dat["instagramHandleY"], insta_dat["ID"], strict=True)
        )
        n2v_dat = pd.read_csv(SMALL_GRAPH_PATH)
        n2v_dat["ID"] = n2v_dat["instagramHandleCheck2"].map(insta2id)
        return n2v_dat["ID"].tolist()

    def _get_style_rho(
        self, emb_path: Path, artist_order: list[str], dist_path: Path
    ) -> float:
        manifest = pd.read_csv(MANIFEST_PATH, sep="\t")
        emb = np.load(emb_path)
        centroids = compute_artist_centroids(emb, manifest, artist_order)
        style_dist = cosine_distance_matrix(centroids)
        social_dist = np.load(dist_path)
        rho, _ = spearman_upper_triangle(style_dist, social_dist)
        return rho

    def test_vgg_vs_gu(self, artist_order: list[str]) -> None:
        rho = self._get_style_rho(VGG_EMB_PATH, artist_order, GU_DIST_PATH)
        assert rho == pytest.approx(0.007, abs=0.002)

    def test_texture_vs_gu(self, artist_order: list[str]) -> None:
        rho = self._get_style_rho(TEXTURE_EMB_PATH, artist_order, GU_DIST_PATH)
        assert rho == pytest.approx(0.043, abs=0.002)

    def test_archetype_vs_gu(self, artist_order: list[str]) -> None:
        rho = self._get_style_rho(ARCHETYPE_PATH, artist_order, GU_DIST_PATH)
        assert rho == pytest.approx(0.013, abs=0.002)
