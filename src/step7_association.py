"""
Step 7: Reproduce Section 6.2 association tests from the paper.

k-means clustering of image-level VGG embeddings, then Cramer's V
association test between cluster membership and demographics.

Methodology (from paper p.14):
  - k-means on image-level VGG embeddings (k=5 in the original code)
  - Cramer's V between cluster labels and: school, gender, region, east/west
  - "There were no visible patterns for any of the available variables" (p.14)

Usage:
    uv run python -m src.step7_association
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans

ANALYSIS_DIR = Path("/Users/nikolaihuckle/Documents/projects/artAnalysis/visart2020")
RESULTS_DIR = Path("results")


def cramers_v(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """Compute Cramer's V statistic for two categorical variables."""
    ct = pd.crosstab(x, y)
    chi2, p, _dof, _expected = chi2_contingency(ct)
    n = int(ct.sum().sum())
    min_dim = min(ct.shape) - 1
    if min_dim == 0 or n == 0:
        return 0.0, float(p)
    v = float(np.sqrt(chi2 / (n * min_dim)))
    return v, float(p)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    manifest = pd.read_csv(ANALYSIS_DIR / "contempArtv2.csv", sep="\t")
    metadata = pd.read_csv(ANALYSIS_DIR / "nodeResults/finalData.csv", sep=";")
    emb = np.load(ANALYSIS_DIR / "rawVGG/contempArtv2.npy")

    # Map images to artist metadata
    id2meta = metadata.set_index("ID").to_dict("index")

    manifest["university"] = manifest["labelsCat"].map(
        lambda x: id2meta.get(x, {}).get("university", None)
    )
    manifest["gender"] = manifest["labelsCat"].map(
        lambda x: id2meta.get(x, {}).get("gender", None)
    )
    manifest["region"] = manifest["labelsCat"].map(
        lambda x: id2meta.get(x, {}).get("region", None)
    )
    manifest["univEastGerman"] = manifest["labelsCat"].map(
        lambda x: id2meta.get(x, {}).get("univEastGerman", None)
    )

    # k-means on image-level embeddings (k=5, matching original code)
    print("Running k-means (k=5) on 14,559 x 4,096 VGG embeddings...")
    km = KMeans(n_clusters=5, random_state=1991, n_init=10)
    km.fit(emb)
    manifest["cluster"] = km.labels_

    print()
    print("=" * 60)
    print("Section 6.2: Cramer's V association tests")
    print("=" * 60)
    print(f"{'Variable':20s} {'Cramer V':>10s} {'p-value':>10s} {'Significant?':>14s}")
    print("-" * 56)

    rows = []
    for var_name, col in [
        ("school", "university"),
        ("gender", "gender"),
        ("region", "region"),
        ("east/west", "univEastGerman"),
    ]:
        valid = manifest.dropna(subset=[col])
        v, p = cramers_v(valid["cluster"], valid[col])
        sig = "yes" if p < 0.05 else "no"
        print(f"{var_name:20s} {v:10.4f} {p:10.4f} {sig:>14s}")
        rows.append(
            {"variable": var_name, "cramers_v": round(v, 4), "p_value": round(p, 4)}
        )

    print("-" * 56)
    print(
        "Paper: 'There were no visible patterns for any of the available variables' (p.14)"
    )

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "section62_cramers_v.csv", index=False)
    print(f"\nSaved to {RESULTS_DIR / 'section62_cramers_v.csv'}")


if __name__ == "__main__":
    main()
