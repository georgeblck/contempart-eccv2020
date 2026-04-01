"""
Step 5: Reproduce Table 2 from the paper.

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
    uv run python -m src.step5_variance
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .core import compute_sigma_c, compute_sigma_c_global, load_manifest
from .paths import (
    ARCHETYPE_PATH,
    MANIFEST_PATH,
    RESULTS_DIR,
    TEXTURE_EMB_PATH,
    VGG_EMB_PATH,
)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    manifest = load_manifest(MANIFEST_PATH)
    print(f"Images: {len(manifest)}, Artists: {manifest['labelsCat'].nunique()}")

    # Three embedding types
    embeddings = {
        "VGG": np.load(VGG_EMB_PATH),
        "Texture": np.load(TEXTURE_EMB_PATH),
    }

    embeddings["Archetype"] = np.load(ARCHETYPE_PATH)

    print()
    print("=" * 60)
    print("Table 2: Local and global style variation")
    print("=" * 60)
    print(f"{'':12s} {'sigma_c':>15s} {'sigma_c_global':>18s}")
    print("-" * 48)

    rows = []
    for name, emb in embeddings.items():
        sc_m, sc_s = compute_sigma_c(emb, manifest)
        sg_m, sg_s = compute_sigma_c_global(emb)
        print(f"{name:12s} {sc_m:.3f} +/- {sc_s:.3f}   {sg_m:.3f} +/- {sg_s:.3f}")
        rows.append(
            {
                "embedding": name,
                "sigma_c": round(sc_m, 3),
                "sigma_c_std": round(sc_s, 3),
                "sigma_c_global": round(sg_m, 3),
                "sigma_c_global_std": round(sg_s, 3),
            }
        )

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
