"""Embeddings helper for VidQuery using SentenceTransformers."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any
import numpy as np

# Allow downloading model weights in CI or when online access is available
if os.environ.get("CI") == "true" or os.environ.get("HF_HUB_OFFLINE") == "0":
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
else:
    # Prefer local cached weights to avoid network timeouts in sandboxed environments
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


@lru_cache(maxsize=1)
def get_embedder() -> Any:
    """Lazily load sentence-transformers 'all-MiniLM-L6-v2' once and cache it."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


def embed_texts(texts: list[str]) -> np.ndarray:
    """Compute normalized sentence embeddings for a list of texts.

    Args:
        texts: List of strings to encode.

    Returns:
        np.ndarray: 2D array of shape (len(texts), embedding_dim) with unit-normalized vectors.
    """
    if not texts:
        return np.empty((0, 384), dtype=np.float32)

    model = get_embedder()
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(embeddings, dtype=np.float32)
