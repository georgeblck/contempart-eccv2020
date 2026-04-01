"""Step 4: Build social graph and compute node2vec distance matrices.

Constructs G^U (artist-to-artist) from the Instagram edgelist, runs
node2vec to embed the graph, and computes pairwise cosine distance
matrices for both G^U and G^Y.

G^Y uses pre-computed node2vec embeddings from the original analysis
(included in the Zenodo download) since re-running node2vec on the
full 247k-node graph is stochastic and slow.

Paper reference (p.12-13):
  "We use the node2vec algorithm on both graphs G^U and G^Y to project
   their relational data into a low-dimensional feature space."

Usage:
    uv run python -m src.step4_network
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from node2vec import Node2Vec
from scipy.spatial.distance import pdist, squareform

from .paths import (
    ARTIST_META_PATH,
    DISTANCES_DIR,
    EDGELIST_PATH,
    GU_DIST_PATH,
    GY_DIST_PATH,
    SMALL_GRAPH_PATH,
)


def build_gu_graph(
    edgelist: pd.DataFrame,
    artist_handles: set[str],
) -> tuple[list[str], np.ndarray]:
    """Build G^U (artist-to-artist) and compute node2vec embeddings.

    Returns:
        nodes: list of artist handles with edges
        dist_matrix: pairwise cosine distance (N, N)
    """
    import networkx as nx

    # Filter to artist-to-artist edges
    a2a = edgelist[
        edgelist["target"].isin(artist_handles)
        & edgelist["source"].isin(artist_handles)
    ].copy()
    a2a = a2a[a2a["target"] != a2a["source"]]
    a2a = a2a[["target", "source"]].drop_duplicates()

    G: nx.Graph[str] = nx.Graph()
    G.add_nodes_from(artist_handles)
    for _, row in a2a.iterrows():
        G.add_edge(row["target"], row["source"])

    # Remove isolates for node2vec
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)
    print(f"  G^U: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  Isolated: {len(isolates)}")

    # Node2vec
    n2v = Node2Vec(
        G,
        dimensions=128,
        walk_length=30,
        num_walks=200,
        p=1.0,
        q=1.0,
        workers=4,
        quiet=True,
    )
    model = n2v.fit(window=10, min_count=1, batch_words=4)

    nodes = sorted(G.nodes())
    embeddings = np.array([model.wv[str(n)] for n in nodes])
    dist_matrix = squareform(pdist(embeddings, metric="cosine"))

    return nodes, dist_matrix


def main() -> None:
    DISTANCES_DIR.mkdir(parents=True, exist_ok=True)

    # Load edgelist and artist handles
    edgelist = pd.read_csv(EDGELIST_PATH)
    metadata = pd.read_csv(ARTIST_META_PATH, sep=";")
    artist_handles = set(metadata["instagramHandle.y"].dropna())

    print("Building G^U...")
    _nodes, gu_dist = build_gu_graph(edgelist, artist_handles)
    np.save(GU_DIST_PATH, gu_dist)
    print(f"  Saved: {GU_DIST_PATH} ({gu_dist.shape})")

    # G^Y: use pre-computed from Zenodo (stochastic, can't reproduce exactly)
    if GY_DIST_PATH.exists():
        print(f"  G^Y: using pre-computed {GY_DIST_PATH}")
    elif SMALL_GRAPH_PATH.exists():
        # Fall back to computing from the original node2vec CSV
        print("  Computing G^Y distances from pre-computed node2vec...")
        graph_df = pd.read_csv(SMALL_GRAPH_PATH)
        n2v_cols = [c for c in graph_df.columns if c.startswith("nodeDim")]
        n2v_emb = graph_df[n2v_cols].to_numpy()
        gy_dist = squareform(pdist(n2v_emb, metric="cosine"))
        np.save(GY_DIST_PATH, gy_dist)
        print(f"  Saved: {GY_DIST_PATH} ({gy_dist.shape})")
    else:
        print("  WARNING: no G^Y data available")

    print("Step 4 complete.")


if __name__ == "__main__":
    main()
