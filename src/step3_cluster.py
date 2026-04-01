"""
Step 3: Reproduce Figure 4 from the paper.

"Evaluation of style embeddings. Similarity between k-Means clustering
based on the three unsupervised embeddings and style labels from Wikiart."

Runs k-means (k=5,10,...,70) on WikiArt embeddings, computes AMI and
purity against known style labels.

Methodology (from paper p.9-10):
  - 20,000 WikiArt images, 1,000 per style, 20 styles
  - k-means with random_state=1991
  - Adjusted Mutual Information (AMI) score
  - Purity: assign each cluster to its most frequent label, count correct
  - "the highest AMI-score of 0.191" (p.10)

Usage:
    uv run python -m src.step3_cluster
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import sklearn.cluster
import sklearn.metrics

from .paths import (
    RESULTS_DIR,
    WIKIART_ARCHETYPE_PATH,
    WIKIART_META_PATH,
    WIKIART_TEXTURE_PATH,
    WIKIART_VGG_PATH,
)


def purity_score(y_true: list[int], y_pred: np.ndarray) -> float:
    """Purity: fraction of images correctly assigned by majority vote per cluster."""
    cm = sklearn.metrics.cluster.contingency_matrix(y_true, y_pred)
    return float(np.sum(np.amax(cm, axis=0)) / np.sum(cm))


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    # Load WikiArt labels
    wikiart = pd.read_csv(WIKIART_META_PATH, sep="\t")
    wikiart["Style1rCat"] = wikiart["Style1r"].astype("category")
    labels = wikiart["Style1rCat"].cat.codes.tolist()
    n_styles = wikiart["Style1r"].nunique()
    print(f"WikiArt: {len(wikiart)} images, {n_styles} styles")

    # Three embedding types
    embeddings = {
        "VGG": np.load(WIKIART_VGG_PATH),
        "Texture": np.load(WIKIART_TEXTURE_PATH),
    }

    # Archetype M=35 (WikiArt archetypes use step 5, so k=5,10,...,35)
    embeddings["Archetype"] = np.load(WIKIART_ARCHETYPE_PATH)

    k_range = list(range(5, 75, 5))  # k = 5, 10, ..., 70

    print()
    print("=" * 60)
    print("Figure 4: AMI and Purity vs k")
    print("=" * 60)

    all_results = []

    for name, emb in embeddings.items():
        print(f"\n{name} ({emb.shape}):")
        ami_scores = []
        pur_scores = []

        for k in k_range:
            km = sklearn.cluster.KMeans(n_clusters=k, random_state=1991, n_init=10)
            km.fit(emb)
            pred = km.labels_

            ami = sklearn.metrics.adjusted_mutual_info_score(labels, pred)
            pur = purity_score(labels, pred)
            ami_scores.append(ami)
            pur_scores.append(pur)

        max_ami = max(ami_scores)
        max_k = k_range[ami_scores.index(max_ami)]
        print(f"  Max AMI: {max_ami:.3f} at k={max_k}")
        print(f"  AMI at k=20: {ami_scores[3]:.3f}")

        for k, ami, pur in zip(k_range, ami_scores, pur_scores, strict=True):
            all_results.append(
                {
                    "embedding": name,
                    "k": k,
                    "AMI": round(ami, 4),
                    "Purity": round(pur, 4),
                }
            )

    # Paper reference
    print("\nPaper says: 'the highest AMI-score of 0.191' (p.10)")

    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_DIR / "figure4_clustering.csv", index=False)
    print(f"\nSaved to {RESULTS_DIR / 'figure4_clustering.csv'}")


if __name__ == "__main__":
    main()
