"""Step 1: Extract VGG-19 FC7 features and Gram matrix texture descriptors.

Extracts two types of features per image:
  - FC7: 4,096-dim features from the second-to-last fully connected layer
  - Gram matrices: correlations of conv layer feature maps (style/texture)

The Gram matrices are saved raw. Step 2 applies SVD to reduce them.

Usage:
    uv run python -m src.step1_extract_vgg
    uv run python -m src.step1_extract_vgg --dataset wikiart
"""

from __future__ import annotations

import argparse
import typing
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm

from .paths import (
    CONTEMPART_IMAGES_DIR,
    EMBEDDINGS_DIR,
    MANIFEST_PATH,
    TEXTURE_EMB_PATH,
    VGG_EMB_PATH,
    WIKIART_IMAGES_DIR,
    WIKIART_META_PATH,
    WIKIART_TEXTURE_PATH,
    WIKIART_VGG_PATH,
)

# VGG-19 conv layers used for Gram matrices (matching paper p.7-8).
# The paper's Table 2 texture results use conv layers 2-5 (not 1-5).
# Original file naming: "conv_1234" = conv2_1 through conv5_1.
GRAM_LAYERS = ["conv2_1", "conv3_1", "conv4_1", "conv5_1"]
GRAM_LAYER_INDICES = [5, 10, 19, 28]  # indices in vgg19.features

PREPROCESS = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def extract_fc7_and_gram(
    image_paths: list[Path],
    batch_size: int = 32,
    device: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract VGG-19 FC7 features and Gram matrix texture descriptors.

    Returns:
        fc7: (N, 4096) FC7 features
        gram_svd: (N, 4096) SVD-reduced Gram matrix features
    """
    if device == "auto":
        device = get_device()

    vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
    vgg = vgg.to(device)
    vgg.eval()

    # Hook into conv layers for Gram matrices
    gram_features: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int) -> typing.Callable[..., None]:
        def hook(
            _module: torch.nn.Module,
            _input: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            gram_features[layer_idx] = output

        return hook

    hooks = []
    for idx in GRAM_LAYER_INDICES:
        h = vgg.features[idx].register_forward_hook(make_hook(idx))
        hooks.append(h)

    # Extract FC7 from classifier[3] (second FC Linear layer, pre-ReLU).
    # classifier layout: [0]=Linear, [1]=ReLU, [2]=Dropout,
    #                     [3]=Linear (FC7), [4]=ReLU, [5]=Dropout, [6]=Linear
    # The original extraction used the raw Linear output (with negatives).
    # Using post-ReLU (classifier[4] or [5]) clips negatives and changes
    # cosine distances, producing incorrect sigma_c values.
    fc7_features: list[torch.Tensor] = []

    def fc7_hook(
        _module: torch.nn.Module,
        _input: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        fc7_features.append(output.detach().cpu())

    fc7_h = vgg.classifier[3].register_forward_hook(fc7_hook)
    hooks.append(fc7_h)

    all_fc7: list[np.ndarray] = []
    all_gram_vectors: list[np.ndarray] = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="VGG-19"):
        batch_paths = image_paths[i : i + batch_size]
        images = []

        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                images.append(PREPROCESS(img))
            except Exception:
                images.append(torch.zeros(3, 224, 224))

        batch = torch.stack(images).to(device)
        fc7_features.clear()
        gram_features.clear()

        with torch.no_grad():
            vgg(batch)

        # FC7
        all_fc7.append(fc7_features[0].numpy())

        # Gram matrices per image
        for img_idx in range(len(batch_paths)):
            gram_parts = []
            for layer_idx in GRAM_LAYER_INDICES:
                feat = gram_features[layer_idx][img_idx]  # (C, H, W)
                c = feat.shape[0]
                feat_flat = feat.reshape(c, -1)
                # Center the features
                feat_flat = feat_flat - feat_flat.mean(dim=1, keepdim=True)
                # Gram matrix
                gram = torch.mm(feat_flat, feat_flat.t()) / feat_flat.shape[1]
                # Normalized upper triangle + diagonal
                gram_np = gram.cpu().numpy()
                upper = gram_np[np.triu_indices(c)]
                gram_parts.append(upper)
            all_gram_vectors.append(np.concatenate(gram_parts))

    # Clean up hooks
    for h in hooks:
        h.remove()

    fc7 = np.concatenate(all_fc7, axis=0)

    # SVD reduction of Gram vectors to 4096 dims
    gram_matrix = np.array(all_gram_vectors)
    print(f"  Gram matrix shape before SVD: {gram_matrix.shape}")
    from sklearn.decomposition import TruncatedSVD

    n_components = min(4096, gram_matrix.shape[0] - 1, gram_matrix.shape[1])
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    gram_svd = svd.fit_transform(gram_matrix)
    print(f"  Gram SVD shape: {gram_svd.shape}")

    return fc7, gram_svd


def get_contempart_paths() -> list[Path]:
    """Get image paths from the contempArt manifest."""
    manifest = pd.read_csv(MANIFEST_PATH, sep="\t")
    paths = []
    for _, row in manifest.iterrows():
        # Original path format: /home/.../visart2020/artist/file.jpg
        # We extract the last two segments: artist/file.jpg
        rel = "/".join(row["paths"].split("/")[-2:])
        paths.append(CONTEMPART_IMAGES_DIR / rel)
    return paths


def get_wikiart_paths() -> list[Path]:
    """Get image paths from the WikiArt manifest."""
    meta = pd.read_csv(WIKIART_META_PATH, sep="\t")
    paths = []
    for _, row in meta.iterrows():
        # serverPath: /home/.../wikiart/Style/filename.jpg
        rel = "/".join(row["serverPath"].split("/")[-2:])
        paths.append(WIKIART_IMAGES_DIR / rel)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["contempart", "wikiart", "both"],
        default="both",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    if args.dataset in ("contempart", "both"):
        print("Extracting contempArt features...")
        paths = get_contempart_paths()
        fc7, texture = extract_fc7_and_gram(paths, batch_size=args.batch_size)
        np.save(VGG_EMB_PATH, fc7)
        np.save(TEXTURE_EMB_PATH, texture)
        print(f"  FC7: {fc7.shape} -> {VGG_EMB_PATH}")
        print(f"  Texture: {texture.shape} -> {TEXTURE_EMB_PATH}")

    if args.dataset in ("wikiart", "both"):
        print("Extracting WikiArt features...")
        paths = get_wikiart_paths()
        fc7, texture = extract_fc7_and_gram(paths, batch_size=args.batch_size)
        np.save(WIKIART_VGG_PATH, fc7)
        np.save(WIKIART_TEXTURE_PATH, texture)
        print(f"  FC7: {fc7.shape} -> {WIKIART_VGG_PATH}")
        print(f"  Texture: {texture.shape} -> {WIKIART_TEXTURE_PATH}")

    print("Step 1 complete.")


if __name__ == "__main__":
    main()
