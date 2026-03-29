# Plan: Exact Reproduction of contempArt (ECCV Workshop 2020)

Goal: make this repo a clean, public, fully reproducible version of the original analysis. Same methods, same data, same results. Modern tooling (uv, renv) but identical methodology. Every number in the paper should be verifiable from this code.

## What the paper produces

| Output | Paper location | Source script (original) |
|--------|---------------|--------------------------|
| VGG FC7, Texture (Gram+SVD), Archetype embeddings | Section 4 | embedContemp.py, utils.py |
| WikiArt embeddings (same 3 types, 1000 sample) | Section 5 | embedWikiart.py |
| AMI, purity, DB, CH, silhouette vs k (WikiArt) | Table 1, Figure 2 | clusterStyle.py, clusterGenre.py, evalMetrics.R |
| MLP classification accuracy (WikiArt) | Table 1 | classifyStyle.py |
| Intra-artist style variance (sigma_c, sigma_c_global) | Table 2 | collapseArtists.py, distCentroids.R |
| Artist-level distance matrices (quantile aggregation) | Section 6 | collapseArtists.py, collapseInstaArtists.py |
| Social graph G^U (364 nodes, 5614 edges) | Section 3.2, Figure 3 | getSmallNetwork.py |
| Social graph G^Y (247k nodes, 745k edges) | Section 3.2 | saveBigNetwork.py, getBigNetwork.py |
| node2vec embeddings (128-dim, both graphs) | Section 6.1 | getSmallNetwork.py, getBigNetwork.py |
| Spearman rho: style vs social distance | Table 3 | mantelServer.R, compareDists.R |
| t-SNE by school, gender, nationality | Section 6.2, Figure 3 | artistTSNE.R |
| Network visualization (Figure 3) | Figure 3 | makeEdgesThird.R, makeNode2vec.R |
| Cramér-V association tests | Section 6.2 | vizContempArt.R |

## Pipeline steps (for the new repo)

### Step 0: Data setup
- Link or copy image data, metadata, social graph from contempart/
- Clean manifests: contempArtv2.csv (image paths, artist IDs)
- Clean metadata: finalData.csv (artist demographics)
- Verify: 442 artists, 14,559 images, 15 schools

### Step 1: Feature extraction (Python, GPU)
- VGG-19 FC7 features (4096-dim per image)
- Gram matrix texture features (5 conv layers, SVD to 4096-dim)
- Archetypal analysis on texture features (sweep k, select M=36)
- Same for WikiArt 1000-image sample
- Verify: embedding shapes match paper

### Step 2: Social network (Python)
- Build G^U from follower/following data (artist-to-artist)
- Build G^Y from full follow data (all accounts)
- Run node2vec (128-dim) on both
- Verify: G^U = 364 nodes, 5614 edges. G^Y = 247,087 nodes, 745,144 edges.

### Step 3: Clustering evaluation on WikiArt (Python + R)
- k-means (k=2..30) with AMI, purity, DB, CH, silhouette
- MLP classification (cross-validated)
- Plot AMI/purity curves
- Verify: Table 1 numbers

### Step 4: Artist distance matrices (Python)
- Aggregate per-image embeddings to per-artist (centroid)
- Compute pairwise cosine distance matrices (all 442, insta 364)
- Compute intra-artist variance (Table 2)
- Verify: Table 2 sigma values

### Step 5: Spearman / Mantel correlations (R)
- Spearman rho between style distance and social distance (Table 3)
- All 3 embeddings x 2 graphs = 6 comparisons
- Verify: Table 3 exact values

### Step 6: Visualization (R)
- t-SNE of artist embeddings colored by school, gender, nationality
- Network graph visualization (Figure 3)
- Verify: visual match with paper figures

### Step 7: Cramér-V association tests (R)
- k-means clustering of VGG embeddings
- Association of cluster membership with school, gender, region
- Verify: Section 6.2 claims

## Key decisions

1. Use uv for Python, renv for R. No conda.
2. VGG-19 pretrained on ImageNet (torchvision). Same architecture, same weights.
3. node2vec: use the node2vec Python package (same as original used GEM library). Verify same hyperparameters (d=128, walks=200, walk_length=30, p=1, q=1).
4. Archetypal analysis: original used SPAMS library. Check if still available, otherwise find equivalent.
5. Image paths: original used absolute Linux paths. New code uses relative paths.
6. Keep all intermediate outputs (.npy, .csv) in gitignored directories, but commit the final result tables and figures.

## Folder structure

```
contempart-eccv2020/
  src/
    step0_data.py          <- verify and prepare data
    step1_embed.py         <- VGG feature extraction
    step1_archetype.py     <- archetypal analysis
    step2_network.py       <- social graph + node2vec
    step3_cluster.py       <- WikiArt clustering evaluation
    step4_distances.py     <- artist distance matrices
    step5_correlations.R   <- Spearman/Mantel (Table 3)
    step6_visualize.R      <- t-SNE, network plots (Figure 3)
    step7_association.R    <- Cramér-V tests
    utils.py               <- VGG forward pass, gram matrices, SVD
  data/                    <- gitignored, symlinked from contempart/
  embeddings/              <- gitignored, regenerable
  distances/               <- gitignored, regenerable
  results/
    report.md              <- all results, tables, figures
    table1.csv             <- WikiArt clustering metrics
    table2.csv             <- intra-artist variance
    table3.csv             <- Spearman correlations
  plots/                   <- figures for paper
  R/                       <- any additional R scripts
```

## Dependencies to check

- SPAMS (archetypal analysis): may need manual install, check if pip/conda available
- GEM (node2vec): original used this, but node2vec package is simpler and equivalent
- MulticoreTSNE: original used this for speed, sklearn TSNE is fine for reproduction
- reticulate: original R scripts loaded numpy via reticulate. We can use readr/RcppCNPy instead.

## What we can reuse from the original

- The core VGG forward pass logic (utils.py makeRawVGG, makeStyleVGG, makeSVD)
- The distance matrix computation logic (collapseArtists.py)
- The Mantel/Spearman test setup (compareDists.R, mantelServer.R)
- The clustering evaluation (clusterStyle.py)

## What needs rewriting

- Image path handling (absolute Linux paths to relative)
- conda/reticulate dependencies (replace with uv/renv)
- Hardcoded server paths throughout
- Graph/network construction (clean up, make reproducible)
- All R scripts that source Python via reticulate (replace with CSV/npy I/O)
