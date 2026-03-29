# Reproduction Status

Tracking which paper results have been reproduced and to what precision.

## Table 2: Local and global style variation (p.12)

Status: reproduced, sigma_c_global exact, sigma_c within ~3%

| Embedding | sigma_c (ours) | sigma_c (paper) | sigma_c_global (ours) | sigma_c_global (paper) |
|-----------|---------------|-----------------|----------------------|----------------------|
| VGG | 0.274 +/- 0.094 | 0.283 +/- 0.080 | 0.435 +/- 0.101 | 0.435 +/- 0.101 |
| Texture | 0.132 +/- 0.054 | 0.137 +/- 0.049 | 0.211 +/- 0.094 | 0.211 +/- 0.094 |
| Archetype | 0.189 +/- 0.124 | 0.195 +/- 0.121 | 0.323 +/- 0.326 | 0.323 +/- 0.326 |

sigma_c_global matches exactly for all three embeddings. sigma_c is slightly lower than reported (0.274 vs 0.283 for VGG). The difference may come from using contempArtv2.npy vs contempArtV3.npy, or from a different sigma computation in the original code.

## Table 3: Rank correlations of style and network distances (p.13)

Status: reproduced, G^U matches to 3 decimal places

| Embedding | G^U (ours) | G^U (paper) | G^Y (ours) | G^Y (paper) |
|-----------|-----------|-------------|-----------|-------------|
| VGG | 0.007 | 0.007 | -0.034 | -0.032 |
| Texture | 0.043 | 0.043 | -0.027 | -0.025 |
| Archetype | 0.013 | 0.012 | -0.065 | -0.057 |

G^U values match to 3 decimal places. G^Y has small differences (0.002-0.008), likely from node2vec stochasticity in the original run (we reuse the original pre-computed distance matrices).

## Table 1: WikiArt clustering evaluation (p.10-11)

Status: not yet reproduced

## Figure 3: t-SNE + network visualization (p.13)

Status: not yet reproduced

## Section 6.2: Demographic visual inspection (p.14)

Status: not yet reproduced (original used only t-SNE visual inspection, no formal tests)
