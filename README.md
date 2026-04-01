# contempart-eccv2020

Reproduction of the analysis from:

> Huckle, N., Garcia, N., & Nakashima, Y. (2020). *contempArt: A Multi-Modal Dataset of Contemporary Artworks and Socio-Demographic Data.* ECCV Workshop on Computer Vision for Fashion, Art and Design. [arXiv:2008.09558](https://arxiv.org/abs/2008.09558)

See also: [contempart](https://github.com/georgeblck/contempart) (dataset) and [contempart-clip](https://github.com/georgeblck/contempart-clip) (re-analysis with CLIP/SD embeddings).

## What this repo does

Reproduces every table and figure from the paper using the original pre-computed embeddings. The goal is to verify the published results and make the analysis transparent and auditable.

All results are reproduced from the original VGG FC7, Gram+SVD Texture, and Archetype embeddings computed in 2020. No features are re-extracted from images.

## Reproduction status

| Paper element | Script | Status | Match? |
|---------------|--------|--------|--------|
| Table 2 (style variance) | step4_distances.py | reproduced | sigma_c_global exact, sigma_c within 3% |
| Table 3 (style vs social network) | step5_correlations.py | reproduced | G^U matches to 3 decimal places |
| Figure 4 (WikiArt AMI/purity) | step3_cluster.py | reproduced | max AMI 0.191 matches paper |
| Figure 3 (t-SNE by school) | step6_visualize.py | reproduced | no visible school clustering, consistent with paper |
| Section 6.2 (association tests) | step7_association.py | reproduced | small Cramer's V (0.09-0.15), all significant but tiny |

## Pipeline

```bash
uv sync

uv run python -m src.step0_data              # verify data completeness
uv run python -m src.step3_cluster            # Figure 4: WikiArt clustering (~45 min)
uv run python -m src.step4_distances          # Table 2: style variance
uv run python -m src.step5_correlations       # Table 3: style vs social network
uv run python -m src.step6_visualize          # Figure 3: t-SNE plots
uv run python -m src.step7_association        # Section 6.2: Cramer's V tests
```

## Data dependencies

This repo reads pre-computed artifacts from the original 2020 analysis. These are not included in the repo and must be available locally:

- `artAnalysis/visart2020/rawVGG/contempArtv2.npy` (14,559 x 4,096, VGG FC7)
- `artAnalysis/visart2020/styleSVD/contempArtV3_conv_01234.npy` (14,559 x 4,096, Gram+SVD)
- `artAnalysis/visart2020/archeTypes/contempArtV3_conv_01234_36.pickle` (Archetype, M=36)
- `artAnalysis/visart2020/distances/small_n2v_cos.npy` (364 x 364, G^U)
- `artAnalysis/visart2020/distances/big_n2v_cos.npy` (364 x 364, G^Y)
- `artAnalysis/visart2020/contempArtv2.csv` (image manifest)
- `artAnalysis/visart2020/nodeResults/finalData.csv` (artist metadata)
- `artAnalysis/visart2020/nodeResults/smallGraphs/smallGraph.csv` (G^U node2vec)
- WikiArt: `rawVGG/rand1992_1000.npy`, `styleSVD/rand1992_1000_conv_01234.npy`, `archeTypes/rand1992_1000_conv_01234_35.pickle`, `rand1991_1000.csv`

## Notes

- The paper reports 442 artists and 14,559 images. The public [contempart dataset](https://github.com/georgeblck/contempart) contains 441 artists and 14,398 images (1 artist lost, 2 images with encoding issues). The pre-computed embeddings used here contain the full 442/14,559.
- See [results/inventory.md](results/inventory.md) for a complete mapping of every artifact to its file location.
