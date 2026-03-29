"""
Step 4: Reproduce Table 2 from the paper.

"Local and global style variation with standard deviation."

Computes two measures of style variance:
  sigma_c       = mean intra-artist cosine distance to centroid (Eq. 8)
  sigma_c_global = mean cosine distance of all images to the global centroid (Eq. 9)

Methodology (from paper p.12):
  c^l = (1/N^l) sum_i e^l_i                          (Eq. 7, per-artist centroid)
  sigma_c = (1/N_a) sum_j (1/N^j) sum_i D_C(c^l, e^j_i)   (Eq. 8, local variance)
  sigma_c_global = (1/N) sum_j sum_i D_C(c_N, e^l_i)       (Eq. 9, global variance)

Expected results (Table 2):
  |           | sigma_c       | sigma_c_global  |
  |-----------|---------------|-----------------|
  | VGG       | .283 +/- .080 | .435 +/- .101   |
  | Texture   | .137 +/- .049 | .211 +/- .094   |
  | Archetype | .195 +/- .121 | .323 +/- .326   |

Usage:
    uv run python -m src.step4_distances
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine


ANALYSIS_DIR = Path("/Users/nikolaihuckle/Documents/projects/artAnalysis/visart2020")
RESULTS_DIR = Path("results")


def compute_table2(emb: np.ndarray, manifest: pd.DataFrame) -> tuple[float, float, float, float]:
    """Compute sigma_c and sigma_c_global for one embedding type.

    Returns: (sigma_c_mean, sigma_c_std, sigma_c_global_mean, sigma_c_global_std)
    """
    artists = manifest["labelsCat"].unique()

    # Per-artist: centroid and mean cosine distance to centroid
    artist_sigmas = []
    all_distances_to_global = []

    # Global centroid c_N (mean of all image embeddings)
    c_global = emb.mean(axis=0)

    for artist in artists:
        mask = manifest["labelsCat"] == artist
        artist_emb = emb[mask.values]
        n = len(artist_emb)
        if n == 0:
            continue

        # Per-artist centroid (Eq. 7)
        centroid = artist_emb.mean(axis=0)

        # Intra-artist cosine distances to centroid
        dists = np.array([cosine(centroid, e) for e in artist_emb])
        sigma_artist = dists.mean()
        artist_sigmas.append(sigma_artist)

        # Global distances (for sigma_c_global)
        global_dists = np.array([cosine(c_global, e) for e in artist_emb])
        all_distances_to_global.extend(global_dists)

    sigma_c_mean = np.mean(artist_sigmas)
    sigma_c_std = np.std(artist_sigmas)
    sigma_c_global_mean = np.mean(all_distances_to_global)
    sigma_c_global_std = np.std(all_distances_to_global)

    return sigma_c_mean, sigma_c_std, sigma_c_global_mean, sigma_c_global_std


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    manifest = pd.read_csv(ANALYSIS_DIR / "contempArtv2.csv", sep="\t")
    print(f"Images: {len(manifest)}, Artists: {manifest['labelsCat'].nunique()}")

    # Three embedding types
    embeddings = {
        "VGG": np.load(ANALYSIS_DIR / "rawVGG/contempArtv2.npy"),
        "Texture": np.load(ANALYSIS_DIR / "styleSVD/contempArtV3_conv_01234.npy"),
    }

    with open(ANALYSIS_DIR / "archeTypes/contempArtV3_conv_01234_36.pickle", "rb") as f:
        Z = pickle.load(f)
        A = pickle.load(f)
        B = pickle.load(f)
    embeddings["Archetype"] = np.hstack([A, B.T])

    print()
    print("=" * 60)
    print("Table 2: Local and global style variation")
    print("=" * 60)
    print(f"{'':12s} {'sigma_c':>15s} {'sigma_c_global':>18s}")
    print("-" * 48)

    rows = []
    for name, emb in embeddings.items():
        sc_m, sc_s, sg_m, sg_s = compute_table2(emb, manifest)
        print(f"{name:12s} {sc_m:.3f} +/- {sc_s:.3f}   {sg_m:.3f} +/- {sg_s:.3f}")
        rows.append({
            "embedding": name,
            "sigma_c": round(sc_m, 3),
            "sigma_c_std": round(sc_s, 3),
            "sigma_c_global": round(sg_m, 3),
            "sigma_c_global_std": round(sg_s, 3),
        })

    print("-" * 48)
    print("\nPaper Table 2:")
    print(f"{'VGG':12s} {'0.283 +/- 0.080':>15s}   {'0.435 +/- 0.101':>18s}")
    print(f"{'Texture':12s} {'0.137 +/- 0.049':>15s}   {'0.211 +/- 0.094':>18s}")
    print(f"{'Archetype':12s} {'0.195 +/- 0.121':>15s}   {'0.323 +/- 0.326':>18s}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "table2.csv", index=False)
    print(f"\nSaved to {RESULTS_DIR / 'table2.csv'}")


if __name__ == "__main__":
    main()
