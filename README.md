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
| Table 2 (style variance) | step5_variance.py | reproduced | sigma_c_global exact, sigma_c within 3% |
| Table 3 (style vs social network) | step6_correlations.py | reproduced | G^U matches to 3 decimal places |
| Figure 4 (WikiArt AMI/purity) | step8_cluster.py | reproduced | max AMI 0.191 matches paper |
| Figure 3 (t-SNE by school) | step7_visualize.py | reproduced | no visible school clustering, consistent with paper |
| Section 6.2 (association tests) | step9_association.py | reproduced | small Cramer's V (0.09-0.15), all significant but tiny |

## Pipeline

```bash
uv sync

uv run python -m src.step0_data              # verify data completeness
uv run python -m src.step5_variance            # Table 2: style variance
uv run python -m src.step6_correlations        # Table 3: style vs social network
uv run python -m src.step7_visualize           # Figure 3: t-SNE plots
uv run python -m src.step8_cluster             # Figure 4: WikiArt clustering (~45 min)
uv run python -m src.step9_association         # Section 6.2: Cramer's V tests
```

## Data

Download the pre-computed artifacts from [Zenodo](https://doi.org/10.5281/zenodo.19367364) and extract into `data/`:

```
data/
  artists.csv                              <- artist metadata (442 rows)
  images_manifest.csv                      <- image manifest (14,559 rows)
  graph_gu_node2vec.csv                    <- G^U node2vec embeddings (364 artists)
  wikiart_metadata.csv                     <- WikiArt sample metadata (20,000 rows)
  embeddings/
    vgg_fc7.npy                            <- VGG FC7 (14,559 x 4,096)
    texture_gram_svd.npy                   <- Gram+SVD texture (14,559 x 4,096)
    archetype_m36.npy                      <- Archetype M=36 (14,559 x 72)
    wikiart_vgg_fc7.npy                    <- WikiArt VGG FC7 (20,000 x 4,096)
    wikiart_texture_gram_svd.npy           <- WikiArt Gram+SVD (20,000 x 4,096)
    wikiart_archetype_m35.npy              <- WikiArt Archetype M=35 (20,000 x 70)
  distances/
    gu_node2vec_cosine.npy                 <- G^U cosine distance (364 x 364)
    gy_node2vec_cosine.npy                 <- G^Y cosine distance (364 x 364)
```

## Notes

- The paper reports 442 artists and 14,559 images. The public [contempart dataset](https://github.com/georgeblck/contempart) contains 441 artists and 14,398 images (1 artist lost, 2 images with encoding issues). The pre-computed embeddings used here contain the full 442/14,559.
- See [results/inventory.md](results/inventory.md) for a complete mapping of every artifact to its file location.
