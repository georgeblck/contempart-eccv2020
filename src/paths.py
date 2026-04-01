"""Shared data paths for all reproduction scripts.

All paths are relative to the repo root. The data/ directory
is downloaded from Zenodo and is not tracked in git.
"""

from __future__ import annotations

from pathlib import Path

# Root data directory (Zenodo download, gitignored)
DATA_DIR = Path("data")

# Metadata
MANIFEST_PATH = DATA_DIR / "images_manifest.csv"
ARTIST_META_PATH = DATA_DIR / "artists.csv"
SMALL_GRAPH_PATH = DATA_DIR / "graph_gu_node2vec.csv"
WIKIART_META_PATH = DATA_DIR / "wikiart_metadata.csv"

# contempArt embeddings
VGG_EMB_PATH = DATA_DIR / "embeddings" / "vgg_fc7.npy"
TEXTURE_EMB_PATH = DATA_DIR / "embeddings" / "texture_gram_svd.npy"
ARCHETYPE_PATH = DATA_DIR / "embeddings" / "archetype_m36.npy"

# WikiArt embeddings
WIKIART_VGG_PATH = DATA_DIR / "embeddings" / "wikiart_vgg_fc7.npy"
WIKIART_TEXTURE_PATH = DATA_DIR / "embeddings" / "wikiart_texture_gram_svd.npy"
WIKIART_ARCHETYPE_PATH = DATA_DIR / "embeddings" / "wikiart_archetype_m35.npy"

# Social network distance matrices
GU_DIST_PATH = DATA_DIR / "distances" / "gu_node2vec_cosine.npy"
GY_DIST_PATH = DATA_DIR / "distances" / "gy_node2vec_cosine.npy"

# Output directories
RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots")
