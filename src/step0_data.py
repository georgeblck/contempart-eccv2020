"""
Step 0: Verify and prepare data for the contempArt ECCV 2020 reproduction.

Checks that all required files exist and match the paper's numbers.
Creates symlinks from the raw data directories into data/.

Paper numbers to verify:
  - 442 artists
  - 14,559 artworks
  - 15 art schools
  - G^U: 364 nodes, 5,614 edges
  - G^Y: 247,087 nodes, 745,144 edges

Usage:
    uv run python -m src.step0_data
"""

from pathlib import Path

import numpy as np
import pandas as pd


# Where the raw data lives (not committed to this repo)
CONTEMPART_DIR = Path("/Users/nikolaihuckle/Documents/projects/contempart")
ANALYSIS_DIR = Path("/Users/nikolaihuckle/Documents/projects/artAnalysis/visart2020")

# Where we expect data in this repo
DATA_DIR = Path("data")
EMBEDDINGS_DIR = Path("embeddings")


def check_file(path: Path, description: str) -> bool:
    if path.exists():
        print(f"  OK: {description} ({path})")
        return True
    print(f"  MISSING: {description} ({path})")
    return False


def main():
    ok = True
    print("=" * 60)
    print("Step 0: Data verification for contempArt ECCV 2020")
    print("=" * 60)

    # ---- Raw data in contempart/ ----
    print("\n--- Raw data (contempart/) ---")

    metadata_path = CONTEMPART_DIR / "finalData.csv"
    ok &= check_file(metadata_path, "Artist metadata (finalData.csv)")
    if metadata_path.exists():
        meta = pd.read_csv(metadata_path, sep=";")
        n_artists = len(meta)
        print(f"  Artists: {n_artists} (paper: 442)")
        if n_artists != 442:
            print(f"  WARNING: expected 442, got {n_artists}")
            ok = False

    imagedata_path = CONTEMPART_DIR / "imageData.csv"
    ok &= check_file(imagedata_path, "Image metadata (imageData.csv)")
    if imagedata_path.exists():
        imgdata = pd.read_csv(imagedata_path, sep=";")
        n_images_meta = len(imgdata)
        print(f"  Images in metadata: {n_images_meta} (paper: 14,559)")

    image_dir = CONTEMPART_DIR / "visart2020"
    ok &= check_file(image_dir, "Image directory (visart2020/)")
    if image_dir.exists():
        n_files = sum(1 for _ in image_dir.rglob("*") if _.is_file() and _.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
        n_folders = sum(1 for d in image_dir.iterdir() if d.is_dir() and not d.name.startswith("."))
        print(f"  Image files on disk: {n_files} (paper: 14,559)")
        print(f"  Artist folders: {n_folders}")
        if n_files < 14559:
            print(f"  WARNING: {14559 - n_files} images still missing (rsync incomplete?)")

    followship_path = CONTEMPART_DIR / "instagramFollowship.csv"
    ok &= check_file(followship_path, "Instagram follow graph (instagramFollowship.csv)")
    if followship_path.exists():
        edges = pd.read_csv(followship_path)
        n_edges = len(edges)
        n_accounts = len(set(edges["Target"]) | set(edges["Source"]))
        print(f"  Edges: {n_edges}")
        print(f"  Unique accounts: {n_accounts} (paper G^Y: 247,087)")

    follow_dir = CONTEMPART_DIR / "followDat"
    following_dir = CONTEMPART_DIR / "followingDat"
    ok &= check_file(follow_dir, "Follower CSVs (followDat/)")
    ok &= check_file(following_dir, "Following CSVs (followingDat/)")

    # ---- Pre-computed artifacts in artAnalysis/ ----
    print("\n--- Pre-computed artifacts (artAnalysis/) ---")

    manifest_path = ANALYSIS_DIR / "contempArtv2.csv"
    ok &= check_file(manifest_path, "Image manifest (contempArtv2.csv)")
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path, sep="\t")
        print(f"  Images in manifest: {len(manifest)} (paper: 14,559)")
        print(f"  Artists: {manifest['labelsCat'].nunique()}")

    for name, expected_shape in [
        ("rawVGG/contempArtv2.npy", (14559, 4096)),
        ("rawVGG/contempArtV3.npy", (14559, 4096)),
        ("styleSVD/contempArtV3_conv_01234.npy", None),
        ("rawVGG/rand1992_1000.npy", (1000, 4096)),
        ("styleSVD/rand1992_1000_conv_01234.npy", None),
    ]:
        path = ANALYSIS_DIR / name
        if check_file(path, name):
            arr = np.load(path)
            shape_ok = "OK" if expected_shape is None or arr.shape == expected_shape else f"MISMATCH (expected {expected_shape})"
            print(f"    Shape: {arr.shape} {shape_ok}")

    finaldata_path = ANALYSIS_DIR / "nodeResults/finalData.csv"
    ok &= check_file(finaldata_path, "Artist metadata (nodeResults/finalData.csv)")
    if finaldata_path.exists():
        fd = pd.read_csv(finaldata_path, sep=";")
        print(f"  Artists: {len(fd)}")
        n_schools = fd["university"].nunique() if "university" in fd.columns else "unknown column"
        print(f"  Schools: {n_schools} (paper: 15)")

    # node2vec
    for name in ["smallGraphs/smallGraph.csv", "bigGraphs/bigWithn2v.csv"]:
        path = ANALYSIS_DIR / "nodeResults" / name
        if check_file(path, f"nodeResults/{name}"):
            df = pd.read_csv(path)
            print(f"    Rows: {len(df)}")

    # Distance matrices
    print("\n--- Distance matrices ---")
    dist_dir = ANALYSIS_DIR / "distances"
    n_dist_files = sum(1 for _ in dist_dir.glob("*.npy"))
    print(f"  Total .npy files: {n_dist_files}")

    for name, expected_shape in [
        ("small_n2v_cos.npy", (364, 364)),
        ("big_n2v_cos.npy", (364, 364)),
        ("insta_cos_50_True.npy", (364, 364)),
        ("all_cos_50_True.npy", (442, 442)),
    ]:
        path = dist_dir / name
        if check_file(path, name):
            arr = np.load(path)
            shape_ok = "OK" if arr.shape == expected_shape else f"MISMATCH (expected {expected_shape})"
            print(f"    Shape: {arr.shape} {shape_ok}")

    # Archetype pickles
    n_pickles = sum(1 for _ in (ANALYSIS_DIR / "archeTypes").glob("*.pickle"))
    print(f"\n  Archetype pickles: {n_pickles}")

    # WikiArt sample
    wikiart_path = ANALYSIS_DIR / "rand1992_1000.csv"
    if wikiart_path.exists():
        wk = pd.read_csv(wikiart_path, sep="\t")
        print(f"  WikiArt sample: {len(wk)} images, {wk['Style1'].nunique() if 'Style1' in wk.columns else '?'} styles")

    # ---- Summary ----
    print("\n" + "=" * 60)
    if ok:
        print("All critical files present. Ready for reproduction.")
    else:
        print("Some files are missing. Check above for details.")
    print("=" * 60)


if __name__ == "__main__":
    main()
