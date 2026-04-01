"""Step 3: Archetypal analysis on texture embeddings.

Computes archetype decomposition of the SVD-reduced Gram matrix features.
Each image is represented as a convex mixture of M archetypes, yielding
a 2M-dimensional embedding (alpha + beta weights concatenated).

Uses py_pcha as a drop-in replacement for the original SPAMS library.

Paper reference (p.8-9):
  "Wynen et al. uses the previously described Gramian texture descriptor
   and archetypal analysis to compute and visualise a set of art archetypes."

Usage:
    uv run python -m src.step3_archetype
    uv run python -m src.step3_archetype --n-archetypes 36
"""

from __future__ import annotations

import argparse

import numpy as _np_compat
import numpy as np

# py_pcha uses np.mat which was removed in NumPy 2.0
if not hasattr(_np_compat, "mat"):
    _np_compat.mat = _np_compat.asmatrix  # type: ignore[attr-defined]

from py_pcha import PCHA

from .paths import (
    ARCHETYPE_PATH,
    EMBEDDINGS_DIR,
    TEXTURE_EMB_PATH,
    WIKIART_ARCHETYPE_PATH,
    WIKIART_TEXTURE_PATH,
)


def compute_archetypes(
    embeddings: np.ndarray,
    n_archetypes: int,
) -> np.ndarray:
    """Compute archetypal analysis and return concatenated mixture weights.

    Returns an (N, 2*M) matrix where M = n_archetypes.
    First M columns are alpha (archetype-to-image weights),
    last M columns are beta (image-to-archetype weights).
    """
    # PCHA expects (features, samples) layout
    data = embeddings.T.astype(np.float64)

    # Run PCHA
    _XC, S, C, _SSE, _varexpl = PCHA(data, noc=n_archetypes)

    # S = archetype-to-image weights (M, N) -> alpha
    # C = image-to-archetype weights (N, M) -> beta
    alpha = S.T  # (N, M)
    beta = C  # (N, M)

    result = np.hstack([alpha, beta])
    print(f"  Archetypes: M={n_archetypes}, output shape: {result.shape}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-archetypes-contempart", type=int, default=36)
    parser.add_argument("--n-archetypes-wikiart", type=int, default=35)
    args = parser.parse_args()

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    if TEXTURE_EMB_PATH.exists():
        print("Computing contempArt archetypes...")
        texture = np.load(TEXTURE_EMB_PATH)
        result = compute_archetypes(texture, args.n_archetypes_contempart)
        np.save(ARCHETYPE_PATH, result)
        print(f"  Saved: {ARCHETYPE_PATH}")
    else:
        print(f"  Skipping contempArt (no {TEXTURE_EMB_PATH})")

    if WIKIART_TEXTURE_PATH.exists():
        print("Computing WikiArt archetypes...")
        texture = np.load(WIKIART_TEXTURE_PATH)
        result = compute_archetypes(texture, args.n_archetypes_wikiart)
        np.save(WIKIART_ARCHETYPE_PATH, result)
        print(f"  Saved: {WIKIART_ARCHETYPE_PATH}")
    else:
        print(f"  Skipping WikiArt (no {WIKIART_TEXTURE_PATH})")

    print("Step 3 complete.")


if __name__ == "__main__":
    main()
