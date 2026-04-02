# contempart-eccv2020 (archive branch)

This branch preserves the initial reproduction that used **pre-computed embeddings** from the original 2020 analysis. No features are re-extracted from images.

For the full end-to-end pipeline (raw images to results), see the [main branch](https://github.com/georgeblck/contempart-eccv2020/tree/main).

## Paper

> Huckle, N., Garcia, N., & Nakashima, Y. (2020). *contempArt: A Multi-Modal Dataset of Contemporary Artworks and Socio-Demographic Data.* ECCV Workshop on Computer Vision for Fashion, Art and Design. [arXiv:2008.09558](https://arxiv.org/abs/2008.09558)

## What this branch does

Verifies the paper's results by re-running the analysis on the original pre-computed VGG FC7, Gram+SVD Texture, and Archetype embeddings from 2020. This was useful for confirming that the statistical pipeline is correct before building the full extraction pipeline on `main`.

## Data

Download the pre-computed artifacts from [Zenodo](https://doi.org/10.5281/zenodo.19367364) and extract into `data/`.

## See also

- [contempart](https://github.com/georgeblck/contempart) (dataset + metadata)
- [contempart-clip](https://github.com/georgeblck/contempart-clip) (re-analysis with CLIP and Stable Diffusion embeddings)
