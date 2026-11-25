"""Two-Stage Hierarchical Video Retriever (Algorithm 1 from IEEE paper).

Mitigates the 'Search Space Expansion Effect' in long educational lectures (>35 min):
- Stage 1: Macro-Chapter Localization (T_macro = 300s, Top-M = 2)
- Stage 2: Micro-Chunk Grounding within shortlisted macro-windows
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable
import numpy as np


@dataclass
class MacroBlock:
    """Represents a coarse-grained temporal macro-window in a video."""
    block_idx: int
    start_sec: float
    end_sec: float
    text: str
    chunk_indices: list[int] = field(default_factory=list)
    chunks: list[dict[str, Any]] = field(default_factory=list)
    score: float = 0.0


def build_macro_blocks(
    chunks: list[dict[str, Any]],
    t_macro: float = 300.0,
) -> list[MacroBlock]:
    """Partition micro-chunks into non-overlapping temporal macro-windows.

    Args:
        chunks: List of micro-chunk dictionaries with 'start', 'end', and 'text'.
        t_macro: Duration of each macro-window in seconds (default: 300.0 = 5 minutes).

    Returns:
        List of MacroBlock instances ordered chronologically.
    """
    if not chunks:
        return []

    # Determine max duration from chunks
    max_time = max(float(c.get("end", 0.0)) for c in chunks)
    n_blocks = max(1, math.ceil(max_time / t_macro))

    blocks: list[MacroBlock] = [
        MacroBlock(
            block_idx=j,
            start_sec=round(j * t_macro, 2),
            end_sec=round((j + 1) * t_macro, 2),
            text="",
            chunk_indices=[],
            chunks=[],
        )
        for j in range(n_blocks)
    ]

    for idx, c in enumerate(chunks):
        start = float(c.get("start", 0.0))
        end = float(c.get("end", start))
        midpoint = (start + end) / 2.0
        block_idx = int(midpoint // t_macro)
        block_idx = min(block_idx, n_blocks - 1)

        blocks[block_idx].chunk_indices.append(idx)
        blocks[block_idx].chunks.append(c)

    # Filter out empty blocks if any, and build composite text
    populated_blocks: list[MacroBlock] = []
    for b in blocks:
        if b.chunks:
            # Join constituent micro-chunk texts with spaces
            b.text = " ".join(c.get("text", "").strip() for c in b.chunks if c.get("text", "").strip())
            populated_blocks.append(b)

    return populated_blocks


def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two 1D or 2D vectors."""
    v1 = np.asarray(vec1, dtype=np.float32).ravel()
    v2 = np.asarray(vec2, dtype=np.float32).ravel()
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-9 or norm2 < 1e-9:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def search_hierarchical(
    query: str,
    chunks: list[dict[str, Any]],
    t_macro: float = 300.0,
    top_m_macro: int = 2,
    k: int = 4,
    embed_fn: Callable[[list[str]], Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Execute Two-Stage Hierarchical Video Retrieval over a list of chunks.

    Stage 1:
        Partitions chunks into macro-blocks of duration t_macro (default 300s).
        Computes cosine similarity between query and each macro-block text.
        Selects top-M macro-blocks (default M=2, restricting search to ~10 minutes).

    Stage 2:
        Restricts candidate pool to micro-chunks inside selected macro-blocks.
        Computes fine-grained cosine similarity and ranks micro-chunks to return top-k.

    Args:
        query: User search question or query text.
        chunks: List of candidate micro-chunk dictionaries.
        t_macro: Macro-window duration in seconds (default: 300.0).
        top_m_macro: Number of top macro-blocks to select in Stage 1 (default: 2).
        k: Number of micro-chunks to return in Stage 2 (default: 4).
        embed_fn: Function taking list[str] and returning embeddings ndarray.

    Returns:
        tuple (top_micro_chunks, selected_macro_blocks_metadata)
    """
    if not chunks:
        return [], []

    if embed_fn is None:
        from embeddings import embed_texts
        embed_fn = embed_texts

    # 1. Partition into macro-blocks
    macro_blocks = build_macro_blocks(chunks, t_macro=t_macro)
    if not macro_blocks:
        return [], []

    # If all chunks fall into a single macro-block or video <= t_macro, fallback cleanly
    if len(macro_blocks) <= top_m_macro:
        selected_blocks = macro_blocks
    else:
        # 2. Stage 1: Embed query and macro-block texts
        macro_texts = [b.text for b in macro_blocks]
        q_emb = embed_fn([query])[0]
        macro_embs = embed_fn(macro_texts)

        for b, m_emb in zip(macro_blocks, macro_embs):
            b.score = _cosine_similarity(q_emb, m_emb)

        # Sort macro-blocks by similarity descending and pick top_m_macro
        sorted_blocks = sorted(macro_blocks, key=lambda b: b.score, reverse=True)
        selected_blocks = sorted_blocks[:top_m_macro]

    selected_macro_metadata = [
        {
            "block_idx": b.block_idx,
            "start_sec": b.start_sec,
            "end_sec": b.end_sec,
            "score": round(float(b.score), 4),
            "n_chunks": len(b.chunks),
        }
        for b in selected_blocks
    ]

    # 3. Stage 2: Gather candidate micro-chunks from selected macro-blocks
    candidate_micro: list[dict[str, Any]] = []
    for b in selected_blocks:
        for c in b.chunks:
            c_copy = dict(c)
            c_copy["macro_block_idx"] = b.block_idx
            c_copy["macro_start_sec"] = b.start_sec
            c_copy["macro_end_sec"] = b.end_sec
            c_copy["macro_score"] = round(float(b.score), 4)
            candidate_micro.append(c_copy)

    if not candidate_micro:
        return [], selected_macro_metadata

    # 4. Stage 2 Ranking: Score candidate micro-chunks
    # Check if micro-chunks already have pre-computed scores against this query
    needs_embedding = any("score" not in c or c["score"] == 0.0 for c in candidate_micro)
    if needs_embedding:
        q_emb = embed_fn([query])[0]
        micro_texts = [c["text"] for c in candidate_micro]
        micro_embs = embed_fn(micro_texts)
        for c, emb in zip(candidate_micro, micro_embs):
            c["score"] = round(_cosine_similarity(q_emb, emb), 4)
    else:
        # Scores already present
        for c in candidate_micro:
            c["score"] = round(float(c["score"]), 4)

    # Sort micro-chunks by score descending
    candidate_micro.sort(key=lambda c: c["score"], reverse=True)
    top_micro = candidate_micro[:k]

    return top_micro, selected_macro_metadata
