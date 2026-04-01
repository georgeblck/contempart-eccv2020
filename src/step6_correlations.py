"""
Step 6: Reproduce Table 3 from the paper.

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
    uv run python -m src.step6_correlations
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .core import (
    compute_artist_centroids,
    cosine_distance_matrix,
    load_artist_metadata,
    load_manifest,
    spearman_upper_triangle,
)
from .paths import (
    ARCHETYPE_PATH,
    ARTIST_META_PATH,
    GU_DIST_PATH,
    GY_DIST_PATH,
    MANIFEST_PATH,
    RESULTS_DIR,
    SMALL_GRAPH_PATH,
    TEXTURE_EMB_PATH,
    VGG_EMB_PATH,
)


def load_364_artist_order() -> list[str]:
    """Get the 364-artist ordering matching the social distance matrices."""
    allDat = load_artist_metadata(ARTIST_META_PATH)
    allDat.rename(
        columns={
            "instagramHandle.x": "instagramHandleX",
            "instagramHandle.y": "instagramHandleY",
        },
        inplace=True,
    )
    insta_dat = allDat.dropna(subset=["instagramHandleY"])
    insta2ID = insta_dat.set_index("instagramHandleY")["ID"].to_dict()
    n2vDat = pd.read_csv(SMALL_GRAPH_PATH)
    n2vDat["ID"] = n2vDat["instagramHandleCheck2"].map(insta2ID)
    return n2vDat["ID"].tolist()


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    manifest = load_manifest(MANIFEST_PATH)
    artist_order = load_364_artist_order()

    # Load pre-computed social distance matrices
    small_n2v = np.load(GU_DIST_PATH)
    big_n2v = np.load(GY_DIST_PATH)

    # Three embedding types
    embeddings = {
        "VGG": np.load(VGG_EMB_PATH),
        "Texture": np.load(TEXTURE_EMB_PATH),
    }

    # Archetype (M=36, so 2M=72 dims)
    embeddings["Archetype"] = np.load(ARCHETYPE_PATH)

    print("=" * 60)
    print("Table 3: Rank correlations of style and network distances")
    print("=" * 60)
    print(f"{'':12s} {'G^U':>8s} {'G^Y':>8s}")
    print("-" * 30)

    rows = []
    for name, emb in embeddings.items():
        centroids = compute_artist_centroids(emb, manifest, artist_order)
        cos_dist = cosine_distance_matrix(centroids)
        rho_gu, _ = spearman_upper_triangle(small_n2v, cos_dist)
        rho_gy, _ = spearman_upper_triangle(big_n2v, cos_dist)
        print(f"{name:12s} {rho_gu:8.3f} {rho_gy:8.3f}")
        rows.append(
            {"embedding": name, "G_U": round(rho_gu, 3), "G_Y": round(rho_gy, 3)}
        )

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
