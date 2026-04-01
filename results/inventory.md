# Complete Artifact Inventory

Everything needed to reproduce the paper and release the dataset.

## Images

- 14,559 images from 442 artists (contempArtv2.csv)
- Currently 14,398 on disk (161 missing from incomplete transfer)
- Missing: stefaniepojar (159/159, empty folder everywhere including external drive, images may be lost), anaiscousin (1/26, encoding issue with "anaïs"), tillmannziola (1/10, encoding issue with "Weichspülung")
- The 2 individual files exist on external drive but have filesystem encoding issues preventing copy
- stefaniepojar images appear to be permanently lost (folder empty on all sources)
- Source: external drive at /Volumes/artStorage/art_data/contempArt/visart2020/
- The pre-computed embeddings (14,559 rows) include all missing images, so the paper results are based on the full set

## Metadata

| File | Location | Rows | Description |
|------|----------|------|-------------|
| finalData.csv | artAnalysis/visart2020/nodeResults/ | 442 | Artist demographics (35 cols, semicolon-separated) |
| finalData2.csv | artAnalysis/visart2020/nodeResults/ | 442 | Same but slightly different column order |
| imageData.csv | contempart/ | 14,559 | Per-image metadata (13 cols, semicolon-separated) |
| contempArtv2.csv | artAnalysis/visart2020/ | 14,559 | Image manifest (paths, artist labels, tab-separated) |
| contempArt.xlsx | contempart/ | -- | Original spreadsheet |

## Embeddings (contempArt)

| File | Shape | Dims | Used for | Size |
|------|-------|------|----------|------|
| rawVGG/contempArtv2.npy | 14,559 x 4,096 | VGG FC7 | Table 2 (sigma_c), Table 3, t-SNE | 239 MB |
| styleSVD/contempArtV3_conv_01234.npy | 14,559 x 4,096 | Gram+SVD texture | Table 2, Table 3 | 239 MB |
| archeTypes/contempArtV3_conv_01234_36.pickle | 14,559 x 72 | Archetype (M=36) | Table 2, Table 3 | 10 MB |

Note: contempArtv2.npy and contempArtV3.npy are nearly identical (max diff 0.000006). collapseArtists.py uses v2, other scripts use V3.

## Embeddings (WikiArt, for Table 1 / Figure 4)

| File | Shape | Description | Size |
|------|-------|-------------|------|
| rawVGG/rand1992_1000.npy | 20,000 x 4,096 | VGG FC7 | 328 MB |
| styleSVD/rand1992_1000_conv_01234.npy | 20,000 x 4,096 | Gram+SVD texture | 328 MB |
| archeTypes/rand1992_1000_conv_01234_35.pickle | 20,000 x 70 | Archetype (M=35) | 13 MB |
| rand1991_1000.csv | 20,000 rows | WikiArt metadata (20 styles, 1000/style) | 5 MB |

## Social Network

| File | Description | Size |
|------|-------------|------|
| instagramFollowship.csv | Full edgelist (456,056 edges, 247,087 accounts) | 17 MB |
| nodeResults/smallGraphs/smallGraph.csv | G^U node2vec (364 artists, 128 dims) | 0.5 MB |
| nodeResults/smallGraphs/smallGraph.graphml | G^U graph file | 0.7 MB |
| nodeResults/bigGraphs/bigWithn2v.csv | G^Y node2vec (247,087 nodes, 128 dims) | 323 MB |
| nodeResults/bigGraphs/bigGraph.graphml | G^Y graph file | 326 MB |
| contempart/followDat/ | 860 per-artist follower CSVs | -- |
| contempart/followingDat/ | 918 per-artist following CSVs | -- |

## Distance Matrices

| File | Shape | Description |
|------|-------|-------------|
| distances/small_n2v_cos.npy | 364 x 364 | G^U node2vec cosine distance |
| distances/small_n2v_eucl.npy | 364 x 364 | G^U node2vec euclidean distance |
| distances/big_n2v_cos.npy | 364 x 364 | G^Y node2vec cosine distance (artist subset) |
| distances/big_n2v_eucl.npy | 364 x 364 | G^Y node2vec euclidean distance (artist subset) |
| distances/insta_cos_50_True.npy | 364 x 364 | VGG FC7 cosine, 50th percentile, symmetric (Instagram subset) |
| distances/all_cos_50_True.npy | 442 x 442 | VGG FC7 cosine, 50th percentile, symmetric (all artists) |

Plus ~90 more distance matrices with different metrics, quantiles, and symmetry settings.

## What maps to what in the paper

| Paper element | Source artifacts |
|---------------|-----------------|
| Table 1 (school breakdown) | finalData.csv (university column) |
| Table 2 (style variance) | contempArtv2.npy + styleSVD + archetype, contempArtv2.csv |
| Table 3 (style vs social) | same embeddings + small_n2v_cos + big_n2v_cos |
| Figure 2 (metadata distributions) | finalData.csv, imageData.csv |
| Figure 3 (network + t-SNE) | smallGraph.graphml, rawVGG + contempArtv2.csv + finalData.csv |
| Figure 4 (AMI/purity curves) | WikiArt embeddings + rand1991_1000.csv |
| Section 6.2 (demographic t-SNE) | rawVGG + contempArtv2.csv + finalData.csv |

## For the contempart dataset release

Minimum viable release:
1. Images (14,559 files in visart2020/ structure, need to recover 161 missing)
2. finalData.csv (artist metadata, cleaned)
3. imageData.csv (image metadata)
4. instagramFollowship.csv (social graph edgelist)
5. followDat/ and followingDat/ (per-artist follower lists)

Optional but valuable:
6. Pre-computed embeddings (VGG, Texture, Archetype .npy files, ~700 MB total)
7. Distance matrices (~4 MB total for the key ones)
8. Node2vec embeddings (smallGraph.csv, bigWithn2v.csv)
9. WikiArt sample + embeddings (for reproduction of Table 1)

## Deviations between the paper and the public dataset release

The paper reports 442 artists and 14,559 images. The public contempart dataset contains 441 artists and 14,398 images. Differences:

1. stefaniepojar (1 artist, 159 images): image files lost. Folder is empty on all sources including the external drive backup. The pre-computed embeddings (rawVGG, styleSVD, archetype .npy files at artAnalysis/) still include these 159 images in the original row ordering, so the paper's results are based on the full set. The contempart-eccv2020 reproduction should account for this discrepancy.

2. anaiscousin: 1 image missing (Cousin_anaïs.jpg). File exists on external drive but has HFS+ unicode encoding corruption preventing copy.

3. tillmannziola: 1 image missing (Weichspülung-2018-170x130-1-1870x2461.jpg). Same encoding issue.

4. The public dataset's artists.csv and images.csv reflect the reduced set (441/14,398). The embeddings in artAnalysis/ reflect the full set (442/14,559). Aligning these requires dropping rows from the embeddings or re-extracting from available images.

5. contempart-clip (the CLIP re-analysis) was run on 441 artists and 14,393 images (14,398 minus 5 corrupt PNGs from luanlamberty). Its results are aligned with the public dataset.

## File locations

- Public dataset: ~/Documents/projects/contempart-dataset/
- Images: ~/Documents/projects/contempart/visart2020/
- Raw metadata: ~/Documents/projects/contempart/
- Analysis artifacts: ~/Documents/projects/artAnalysis/visart2020/
- Social data: ~/Documents/projects/contempart/followDat/, followingDat/
