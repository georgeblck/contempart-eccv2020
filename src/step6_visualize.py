"""
Step 6: Reproduce Figure 3 from the paper.

Artist-level t-SNE of VGG FC7 embeddings, colored by art school.
Also produces t-SNE colored by gender and nationality for Section 6.2.

Methodology (from paper p.12-14):
  - Per-artist centroid of VGG FC7 embeddings (contempArtv2.npy)
  - t-SNE with cosine distance
  - "we extract a two-dimensional feature space from the VGG embeddings
     with t-SNE, both per image and per artist" (p.14)
  - "There were no visible patterns for any of the available variables" (p.14)

Usage:
    uv run python -m src.step6_visualize
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

ANALYSIS_DIR = Path("/Users/nikolaihuckle/Documents/projects/artAnalysis/visart2020")
RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots")

# Paul Tol muted + extras for 15 schools
SCHOOL_COLORS = [
    "#332288",
    "#88CCEE",
    "#44AA99",
    "#117733",
    "#999933",
    "#DDCC77",
    "#CC6677",
    "#882255",
    "#AA4499",
    "#DDDDDD",
    "#661100",
    "#6699CC",
    "#AA4466",
    "#BBCC33",
    "#AAAA00",
]


def plot_tsne(
    coords: np.ndarray,
    labels: np.ndarray,
    title: str,
    palette: list[str],
    output_path: Path,
    legend_cols: int = 2,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    unique_labels = sorted(set(labels))
    markers = [
        "o",
        "^",
        "s",
        "D",
        "v",
        "P",
        "X",
        "*",
        "p",
        "h",
        "d",
        "<",
        ">",
        "8",
        "H",
    ]
    for i, lab in enumerate(unique_labels):
        mask = np.array(labels) == lab
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=palette[i % len(palette)],
            marker=markers[i % len(markers)],
            label=lab,
            s=30,
            alpha=0.7,
            edgecolors="none",
        )
    ax.set_title(title, fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=7,
        markerscale=1.5,
        frameon=False,
        ncol=legend_cols,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)

    # Load data
    manifest = pd.read_csv(ANALYSIS_DIR / "contempArtv2.csv", sep="\t")
    metadata = pd.read_csv(ANALYSIS_DIR / "nodeResults/finalData.csv", sep=";")
    emb = np.load(ANALYSIS_DIR / "rawVGG/contempArtv2.npy")

    print(f"Images: {len(manifest)}, Artists: {len(metadata)}, Embeddings: {emb.shape}")

    # Per-artist centroid
    artists = sorted(metadata["ID"].unique())
    centroid_list: list[np.ndarray] = []
    for artist in artists:
        mask = manifest["labelsCat"] == artist
        if mask.sum() > 0:
            centroid_list.append(emb[mask.values].mean(axis=0))
        else:
            centroid_list.append(np.zeros(emb.shape[1]))
    centroids = np.array(centroid_list)
    print(f"Centroids: {centroids.shape}")

    # t-SNE (cosine distance, matching original)
    print("Running t-SNE...")
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        random_state=42,
        metric="cosine",
        max_iter=5000,
    )
    coords = tsne.fit_transform(centroids)
    np.save(RESULTS_DIR / "artist_tsne_coords.npy", coords)

    # Build artist metadata lookup
    artist_meta = pd.DataFrame({"ID": artists}).merge(metadata, on="ID", how="left")

    # Figure 3: t-SNE by school
    schools = np.asarray(artist_meta["university"].fillna("Unknown").values)
    plot_tsne(
        coords,
        schools,
        "t-SNE by art school (VGG FC7)",
        SCHOOL_COLORS,
        PLOTS_DIR / "tsne_school.png",
    )

    # Section 6.2: t-SNE by gender
    genders = np.asarray(artist_meta["gender"].fillna("Unknown").values)
    plot_tsne(
        coords,
        genders,
        "t-SNE by gender (VGG FC7)",
        ["#E69F00", "#56B4E9", "#999999"],
        PLOTS_DIR / "tsne_gender.png",
        legend_cols=1,
    )

    # Section 6.2: t-SNE by nationality/continent
    continents = np.asarray(artist_meta["continent"].fillna("Unknown").values)
    plot_tsne(
        coords,
        continents,
        "t-SNE by continent (VGG FC7)",
        ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#999999"],
        PLOTS_DIR / "tsne_continent.png",
        legend_cols=1,
    )

    print("Step 6 complete.")


if __name__ == "__main__":
    main()
