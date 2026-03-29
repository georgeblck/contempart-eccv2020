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

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn.cluster
import sklearn.metrics


ANALYSIS_DIR = Path("/Users/nikolaihuckle/Documents/projects/artAnalysis/visart2020")
RESULTS_DIR = Path("results")


def purity_score(y_true: list, y_pred: np.ndarray) -> float:
    """Purity: fraction of images correctly assigned by majority vote per cluster."""
    cm = sklearn.metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(cm, axis=0)) / np.sum(cm)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    # Load WikiArt labels
    wikiart = pd.read_csv(ANALYSIS_DIR / "rand1991_1000.csv", sep="\t")
    wikiart["Style1rCat"] = wikiart["Style1r"].astype("category")
    labels = wikiart["Style1rCat"].cat.codes.tolist()
    n_styles = wikiart["Style1r"].nunique()
    print(f"WikiArt: {len(wikiart)} images, {n_styles} styles")

    # Three embedding types
    embeddings = {
        "VGG": np.load(ANALYSIS_DIR / "rawVGG/rand1992_1000.npy"),
        "Texture": np.load(ANALYSIS_DIR / "styleSVD/rand1992_1000_conv_01234.npy"),
    }

    # Archetype M=35 (WikiArt archetypes use step 5, so k=5,10,...,35)
    arche_path = ANALYSIS_DIR / "archeTypes/rand1992_1000_conv_01234_35.pickle"
    with open(arche_path, "rb") as f:
        Z = pickle.load(f)
        A = pickle.load(f)
        B = pickle.load(f)
    embeddings["Archetype"] = np.hstack([A, B.T])

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

        for k, ami, pur in zip(k_range, ami_scores, pur_scores):
            all_results.append({
                "embedding": name,
                "k": k,
                "AMI": round(ami, 4),
                "Purity": round(pur, 4),
            })

    # Paper reference
    print("\nPaper says: 'the highest AMI-score of 0.191' (p.10)")

    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_DIR / "figure4_clustering.csv", index=False)
    print(f"\nSaved to {RESULTS_DIR / 'figure4_clustering.csv'}")


if __name__ == "__main__":
    main()
