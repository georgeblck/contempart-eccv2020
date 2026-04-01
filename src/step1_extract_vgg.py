"""Step 1: Extract VGG-19 FC7 features and Gram matrix texture descriptors.

Extracts two types of features per image:
  - FC7: 4,096-dim features from the second-to-last fully connected layer
  - Texture: spatial mean + Gram matrix upper triangle per conv layer,
    concatenated and SVD-reduced to 4,096 dims

Paper reference (p.7-8):
  "We extract the Gram matrix ... from 4 convolutional layers of a
   pretrained VGG-19 network and apply SVD to reduce dimensionality."

Usage:
    uv run python -m src.step1_extract_vgg
    uv run python -m src.step1_extract_vgg --dataset wikiart
    uv run python -m src.step1_extract_vgg --limit 100  # subset for testing
"""

from __future__ import annotations

import argparse
import json
import tempfile
import typing
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
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

# Handle truncated JPEGs (matching original embedContemp.py)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# VGG-19 conv layers for Gram matrices.
# Paper Table 2 uses conv2_1 through conv5_1 (original "conv_1234" naming).
GRAM_LAYER_INDICES = [5, 10, 19, 28]  # conv2_1, conv3_1, conv4_1, conv5_1
GRAM_LAYER_CHANNELS = [128, 256, 512, 512]

# Total raw texture features per image: spatial means + Gram upper triangles
GRAM_DIM = sum(c + c * (c + 1) // 2 for c in GRAM_LAYER_CHANNELS)

# Image preprocessing matching original embedContemp.py:
#   Resize(512, LANCZOS) + CenterCrop(512) + ImageNet normalization
# The original used 512x512 crops, NOT the standard 224x224.
IMAGE_SIZE = 512
PREPROCESS = transforms.Compose(
    [
        transforms.Resize(
            IMAGE_SIZE, interpolation=transforms.InterpolationMode.LANCZOS
        ),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def get_device() -> str:
    """Select compute device. CUDA if available, otherwise CPU.

    MPS (Apple Silicon) is not used because AdaptiveAvgPool2d does not
    support non-divisible input sizes on the MPS backend, and 512x512
    images produce 16x16 feature maps that cannot be pooled to 7x7.
    """
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _gram_upper_triangle(feat_map: torch.Tensor) -> np.ndarray:
    """Compute Gram matrix upper triangle from a (C, H, W) feature map.

    Matches original gramMatrix() in utils.py:
      - NO centering (raw features)
      - Divide by spatial size (H*W), then by C*(C+1)
      - Extract upper triangle including diagonal
    """
    c = feat_map.shape[0]
    feat_flat = feat_map.reshape(c, -1)  # (C, H*W)
    norm = c * (c + 1)
    gram = feat_flat.matmul(feat_flat.t()) / feat_flat.shape[1] / norm
    idx = torch.triu_indices(c, c)
    return gram[idx[0], idx[1]].cpu().numpy()


def _spatial_mean(feat_map: torch.Tensor) -> np.ndarray:
    """Compute spatial mean of a (C, H, W) feature map.

    Matches original: torch.mean(item, dim=[2,3]) / (C*(C+1))
    Input here is (C, H, W) not (1, C, H, W) so we mean over dims [1,2].
    """
    c = feat_map.shape[0]
    norm = c * (c + 1)
    return (feat_map.mean(dim=[1, 2]) / norm).cpu().numpy()


def extract_fc7_and_gram(
    image_paths: list[Path],
    batch_size: int = 32,
    device: str = "auto",
    checkpoint_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract VGG-19 FC7 features and Gram matrix texture descriptors.

    Uses memory-mapped arrays for Gram features to avoid OOM on large
    datasets, and saves checkpoints so extraction can resume after a crash.

    Returns:
        fc7: (N, 4096) FC7 features (pre-ReLU)
        gram_svd: (N, 4096) SVD-reduced texture features
    """
    if device == "auto":
        device = get_device()

    n_images = len(image_paths)

    # Set up checkpoint and mmap paths
    if checkpoint_dir is None:
        checkpoint_dir = Path(tempfile.mkdtemp(prefix="vgg_extract_"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = checkpoint_dir / "checkpoint.json"
    fc7_mmap_path = checkpoint_dir / "fc7_raw.npy"
    gram_mmap_path = checkpoint_dir / "gram_raw.npy"

    # Check for existing checkpoint
    start_batch = 0
    if ckpt_file.exists():
        with open(ckpt_file) as f:
            ckpt = json.load(f)
        if ckpt.get("n_images") == n_images and ckpt.get("batch_size") == batch_size:
            start_batch = ckpt["completed_batches"]
            print(f"  Resuming from batch {start_batch} (checkpoint: {ckpt_file})")

    # Create or open memory-mapped arrays
    n_batches = (n_images + batch_size - 1) // batch_size
    if start_batch == 0:
        fc7_mmap = np.lib.format.open_memmap(
            str(fc7_mmap_path), mode="w+", dtype=np.float32, shape=(n_images, 4096)
        )
        gram_mmap = np.lib.format.open_memmap(
            str(gram_mmap_path), mode="w+", dtype=np.float32, shape=(n_images, GRAM_DIM)
        )
    else:
        fc7_mmap = np.lib.format.open_memmap(str(fc7_mmap_path), mode="r+")
        gram_mmap = np.lib.format.open_memmap(str(gram_mmap_path), mode="r+")

    # Nothing to do if fully checkpointed
    if start_batch >= n_batches:
        print("  All batches already completed, skipping to SVD.")
    else:
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
        # The original used the raw Linear output (preserving negative values).
        fc7_features: list[torch.Tensor] = []

        def fc7_hook(
            _module: torch.nn.Module,
            _input: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            # .clone() is critical: classifier[4] is ReLU(inplace=True) which
            # would overwrite the Linear output tensor, zeroing all negatives.
            fc7_features.append(output.detach().clone().cpu())

        fc7_h = vgg.classifier[3].register_forward_hook(fc7_hook)
        hooks.append(fc7_h)

        batch_ranges = list(range(0, n_images, batch_size))
        for batch_idx, i in enumerate(
            tqdm(batch_ranges, desc="VGG-19", initial=start_batch, total=n_batches)
        ):
            if batch_idx < start_batch:
                continue

            batch_paths = image_paths[i : i + batch_size]
            actual_batch = len(batch_paths)
            images = []

            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    images.append(PREPROCESS(img))
                except Exception:
                    images.append(torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE))

            batch = torch.stack(images).to(device)
            fc7_features.clear()
            gram_features.clear()

            with torch.no_grad():
                vgg(batch)

            # Write FC7 directly to mmap
            fc7_mmap[i : i + actual_batch] = fc7_features[0].numpy()

            # Write texture features directly to mmap
            for img_idx in range(actual_batch):
                means = []
                grams = []
                for layer_idx in GRAM_LAYER_INDICES:
                    feat = gram_features[layer_idx][img_idx]  # (C, H, W)
                    means.append(_spatial_mean(feat))
                    grams.append(_gram_upper_triangle(feat))
                gram_mmap[i + img_idx] = np.concatenate(means + grams)

            # Flush and save checkpoint every 50 batches
            if (batch_idx + 1) % 50 == 0 or batch_idx == n_batches - 1:
                fc7_mmap.flush()
                gram_mmap.flush()
                with open(ckpt_file, "w") as f:
                    json.dump(
                        {
                            "completed_batches": batch_idx + 1,
                            "n_images": n_images,
                            "batch_size": batch_size,
                        },
                        f,
                    )

        # Clean up hooks
        for h in hooks:
            h.remove()

    # Read FC7 from mmap into regular array
    fc7 = np.array(fc7_mmap)

    # SVD reduction of texture vectors to 4096 dims
    print(f"  Texture features before SVD: {gram_mmap.shape}")
    from sklearn.decomposition import TruncatedSVD

    n_components = min(4096, n_images - 1, GRAM_DIM)
    svd = TruncatedSVD(n_components=n_components)
    gram_svd = svd.fit_transform(gram_mmap)
    print(f"  Texture SVD shape: {gram_svd.shape}")

    # Clean up temp files
    del fc7_mmap, gram_mmap
    for p in [fc7_mmap_path, gram_mmap_path, ckpt_file]:
        p.unlink(missing_ok=True)
    if checkpoint_dir.exists() and not any(checkpoint_dir.iterdir()):
        checkpoint_dir.rmdir()

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
    parser.add_argument(
        "--limit", type=int, default=0, help="Process only first N images (0=all)"
    )
    args = parser.parse_args()

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    if args.dataset in ("contempart", "both"):
        print("Extracting contempArt features...")
        paths = get_contempart_paths()
        if args.limit > 0:
            paths = paths[: args.limit]
            print(f"  Limited to first {len(paths)} images")
        fc7, texture = extract_fc7_and_gram(
            paths,
            batch_size=args.batch_size,
            checkpoint_dir=EMBEDDINGS_DIR / ".checkpoint_contempart",
        )
        np.save(VGG_EMB_PATH, fc7)
        np.save(TEXTURE_EMB_PATH, texture)
        print(f"  FC7: {fc7.shape} -> {VGG_EMB_PATH}")
        print(f"  Texture: {texture.shape} -> {TEXTURE_EMB_PATH}")

    if args.dataset in ("wikiart", "both"):
        print("Extracting WikiArt features...")
        paths = get_wikiart_paths()
        if args.limit > 0:
            paths = paths[: args.limit]
            print(f"  Limited to first {len(paths)} images")
        fc7, texture = extract_fc7_and_gram(
            paths,
            batch_size=args.batch_size,
            checkpoint_dir=EMBEDDINGS_DIR / ".checkpoint_wikiart",
        )
        np.save(WIKIART_VGG_PATH, fc7)
        np.save(WIKIART_TEXTURE_PATH, texture)
        print(f"  FC7: {fc7.shape} -> {WIKIART_VGG_PATH}")
        print(f"  Texture: {texture.shape} -> {WIKIART_TEXTURE_PATH}")

    print("Step 1 complete.")


if __name__ == "__main__":
    main()
