"""
Step 5: Reproduce Table 3 from the paper.

"Rank correlations of style and network distances."

Computes Spearman rho between per-artist centroid cosine distances
(for each of the three embedding types) and node2vec social distances
(for both G^U and G^Y).

Methodology (from paper Section 6.1, p.13):
  - Per-artist centroid: c^l = (1/N^l) sum_i e^l_i  (p.12)
  - Pairwise cosine distance between centroids
  - "Spearman's rank coefficient is used to compute the correlation
     between the flattened upper triangular parts of the described
     distance matrices." (p.13)

Expected results (Table 3):
  |           | G^U   | G^Y   |
  |-----------|-------|-------|
  | VGG       | .007  | -.032 |
  | Texture   | .043  | -.025 |
  | Archetype | .012  | -.057 |

Usage:
    uv run python -m src.step5_correlations
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr


ANALYSIS_DIR = Path("/Users/nikolaihuckle/Documents/projects/artAnalysis/visart2020")
ORIGINAL_2020 = Path("data/original_2020")
RESULTS_DIR = Path("results")


def load_364_artist_order() -> list[str]:
    """Get the 364-artist ordering matching the social distance matrices."""
    allDat = pd.read_csv(ANALYSIS_DIR / "nodeResults/finalData.csv", sep=";")
    allDat.rename(
        columns={"instagramHandle.x": "instagramHandleX", "instagramHandle.y": "instagramHandleY"},
        inplace=True,
    )
    insta_dat = allDat.dropna(subset=["instagramHandleY"])
    insta2ID = insta_dat.set_index("instagramHandleY")["ID"].to_dict()
    n2vDat = pd.read_csv(ANALYSIS_DIR / "nodeResults/smallGraphs/smallGraph.csv")
    n2vDat["ID"] = n2vDat["instagramHandleCheck2"].map(insta2ID)
    return n2vDat["ID"].tolist()


def compute_centroid_cosine(emb: np.ndarray, manifest: pd.DataFrame, artist_order: list[str]) -> np.ndarray:
    """Compute per-artist centroid, then pairwise cosine distance matrix."""
    centroids = []
    for artist_id in artist_order:
        mask = manifest["labelsCat"] == artist_id
        if mask.sum() == 0:
            centroids.append(np.zeros(emb.shape[1]))
            continue
        centroids.append(emb[mask.values].mean(axis=0))
    return squareform(pdist(np.array(centroids), metric="cosine"))


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    manifest = pd.read_csv(ANALYSIS_DIR / "contempArtv2.csv", sep="\t")
    artist_order = load_364_artist_order()

    # Load pre-computed social distance matrices
    small_n2v = np.load(ANALYSIS_DIR / "distances/small_n2v_cos.npy")
    big_n2v = np.load(ANALYSIS_DIR / "distances/big_n2v_cos.npy")
    idx = np.triu_indices(364, k=1)

    # Three embedding types
    embeddings = {
        "VGG": np.load(ANALYSIS_DIR / "rawVGG/contempArtv2.npy"),
        "Texture": np.load(ANALYSIS_DIR / "styleSVD/contempArtV3_conv_01234.npy"),
    }

    # Archetype (M=36, so 2M=72 dims)
    with open(ANALYSIS_DIR / "archeTypes/contempArtV3_conv_01234_36.pickle", "rb") as f:
        Z = pickle.load(f)
        A = pickle.load(f)
        B = pickle.load(f)
    embeddings["Archetype"] = np.hstack([A, B.T])

    print("=" * 60)
    print("Table 3: Rank correlations of style and network distances")
    print("=" * 60)
    print(f"{'':12s} {'G^U':>8s} {'G^Y':>8s}")
    print("-" * 30)

    rows = []
    for name, emb in embeddings.items():
        cos_dist = compute_centroid_cosine(emb, manifest, artist_order)
        rho_gu, _ = spearmanr(small_n2v[idx], cos_dist[idx])
        rho_gy, _ = spearmanr(big_n2v[idx], cos_dist[idx])
        print(f"{name:12s} {rho_gu:8.3f} {rho_gy:8.3f}")
        rows.append({"embedding": name, "G_U": round(rho_gu, 3), "G_Y": round(rho_gy, 3)})

    print("-" * 30)
    print("\nPaper Table 3:")
    print(f"{'VGG':12s} {'0.007':>8s} {'-0.032':>8s}")
    print(f"{'Texture':12s} {'0.043':>8s} {'-0.025':>8s}")
    print(f"{'Archetype':12s} {'0.012':>8s} {'-0.057':>8s}")

    # Save
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "table3.csv", index=False)
    print(f"\nSaved to {RESULTS_DIR / 'table3.csv'}")


if __name__ == "__main__":
    main()
