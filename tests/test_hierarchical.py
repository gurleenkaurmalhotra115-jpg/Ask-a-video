"""Automated unit test suite for Two-Stage Hierarchical Video Retrieval (Algorithm 1)."""

from __future__ import annotations

from typing import Any
import numpy as np
import pytest

from hierarchical_retriever import MacroBlock, build_macro_blocks, search_hierarchical
from mcp_server import mcp, search_video_moments


def test_build_macro_blocks_basic():
    """Verify temporal macro-window partitioning and text concatenation."""
    chunks = [
        {"chunk_id": 1, "start": 10.0, "end": 45.0, "text": "Introduction to gradient descent."},
        {"chunk_id": 2, "start": 50.0, "end": 120.0, "text": "Cost function formulation."},
        {"chunk_id": 3, "start": 320.0, "end": 400.0, "text": "Batch gradient descent update rule."},
        {"chunk_id": 4, "start": 650.0, "end": 720.0, "text": "Summary of learning rate tuning."},
    ]
    blocks = build_macro_blocks(chunks, t_macro=300.0)
    # Total duration is 720s -> 3 macro blocks (0-300s, 300-600s, 600-900s)
    assert len(blocks) == 3

    # Block 0 (0-300s)
    assert blocks[0].block_idx == 0
    assert blocks[0].start_sec == 0.0
    assert blocks[0].end_sec == 300.0
    assert len(blocks[0].chunks) == 2
    assert "gradient descent" in blocks[0].text
    assert "Cost function" in blocks[0].text

    # Block 1 (300-600s)
    assert blocks[1].block_idx == 1
    assert blocks[1].start_sec == 300.0
    assert blocks[1].end_sec == 600.0
    assert len(blocks[1].chunks) == 1
    assert "Batch gradient descent update rule" in blocks[1].text

    # Block 2 (600-900s)
    assert blocks[2].block_idx == 2
    assert blocks[2].start_sec == 600.0
    assert blocks[2].end_sec == 900.0
    assert len(blocks[2].chunks) == 1
    assert "Summary of learning rate" in blocks[2].text


def test_build_macro_blocks_empty():
    """Empty chunk list returns empty block list."""
    assert build_macro_blocks([], t_macro=300.0) == []


def test_build_macro_blocks_short_single_window():
    """Video with duration <= t_macro returns single macro block."""
    chunks = [
        {"chunk_id": 1, "start": 0.0, "end": 30.0, "text": "Welcome to the lecture."},
        {"chunk_id": 2, "start": 35.0, "end": 90.0, "text": "Today we cover attention."},
    ]
    blocks = build_macro_blocks(chunks, t_macro=300.0)
    assert len(blocks) == 1
    assert blocks[0].block_idx == 0
    assert len(blocks[0].chunks) == 2


def test_search_hierarchical_two_stage_filtering():
    """Verify that Stage 1 filters out distant distractor macro-blocks."""
    # 4 chunks in 4 separate 300s blocks
    chunks = [
        {"chunk_id": 1, "start": 50.0, "end": 100.0, "text": "Distractor mention of gradient in intro."},
        {"chunk_id": 2, "start": 350.0, "end": 400.0, "text": "Brief mention of cost functions."},
        {"chunk_id": 3, "start": 650.0, "end": 750.0, "text": "Target mathematical proof of batch gradient descent update rule with parameter theta."},
        {"chunk_id": 4, "start": 950.0, "end": 1000.0, "text": "Distractor recap of previous lecture."},
    ]

    # Mock embed_fn: query matches chunk 3 strongly
    def mock_embed(texts: list[str]) -> np.ndarray:
        vecs = []
        for t in texts:
            vec = np.zeros(16, dtype=np.float32)
            if "theta" in t or "proof" in t:
                vec[0] = 1.0
            elif "gradient" in t:
                vec[1] = 0.5
            else:
                vec[2] = 0.2
            vecs.append(vec / (np.linalg.norm(vec) + 1e-9))
        return np.array(vecs)

    top_micro, selected_macro = search_hierarchical(
        query="batch gradient descent parameter update rule theta",
        chunks=chunks,
        t_macro=300.0,
        top_m_macro=1,  # Select only top 1 macro-block
        k=2,
        embed_fn=mock_embed,
    )

    assert len(selected_macro) == 1
    assert selected_macro[0]["block_idx"] == 2  # Block 2 covers 600-900s
    assert len(top_micro) == 1
    assert top_micro[0]["chunk_id"] == 3
    assert top_micro[0]["macro_block_idx"] == 2
    assert "Target mathematical proof" in top_micro[0]["text"]


def test_search_hierarchical_short_video_fallback():
    """Video shorter than top_m_macro * t_macro keeps all blocks without crash."""
    chunks = [
        {"chunk_id": 1, "start": 0.0, "end": 60.0, "text": "Part 1"},
        {"chunk_id": 2, "start": 70.0, "end": 140.0, "text": "Part 2"},
    ]
    top_micro, selected_macro = search_hierarchical(
        query="Part 1",
        chunks=chunks,
        t_macro=300.0,
        top_m_macro=2,
        k=2,
        embed_fn=lambda texts: np.tile(np.array([1.0, 0.0], dtype=np.float32), (len(texts), 1)),
    )
    assert len(top_micro) == 2
    assert len(selected_macro) == 1


def test_fastapi_search_hierarchical_integration(monkeypatch):
    """Test FastAPI search_video endpoint with use_hierarchical=True."""
    # Monkeypatch get_qdrant_client and search_hierarchical in api
    def fake_search_hierarchical(*args, **kwargs):
        return [
            {
                "chunk_id": 42,
                "text": "Hierarchical chunk result",
                "start": 612.0,
                "end": 645.0,
                "score": 0.88,
                "macro_block_idx": 2,
            }
        ], [{"block_idx": 2, "start_sec": 600.0, "end_sec": 900.0, "score": 0.91, "n_chunks": 5}]

    import api
    from api import SearchRequest
    monkeypatch.setattr(api, "get_qdrant_client", lambda path="qdrant_data": object())
    monkeypatch.setattr(api, "search_hierarchical", fake_search_hierarchical)

    req = SearchRequest(
        video_id="test_video_123",
        query="gradient descent update rule",
        strategy="fixed_token",
        use_hierarchical=True,
        t_macro=300.0,
        top_m_macro=2,
        k=4,
    )
    data = api.search_video(req)
    assert data["hierarchical"] is True
    assert len(data["selected_macro_blocks"]) == 1
    assert data["results"][0]["chunk_id"] == 42
    assert data["results"][0]["score"] == 0.88


def test_mcp_search_video_moments_hierarchical(monkeypatch):
    """Test MCP tool search_video_moments with hierarchical=True."""
    def fake_qdrant_search_hierarchical(*args, **kwargs):
        return [
            {
                "chunk_id": 99,
                "start": 350.0,
                "end": 390.0,
                "text": "MCP hierarchical moment.",
                "score": 0.79,
                "macro_block_idx": 1,
                "macro_start_sec": 300.0,
                "macro_end_sec": 600.0,
            }
        ], []

    import mcp_server
    monkeypatch.setattr(mcp_server, "get_qdrant_client", lambda: object())
    monkeypatch.setattr(mcp_server, "qdrant_search_hierarchical", fake_qdrant_search_hierarchical)

    moments = search_video_moments(
        video_id="demo_lecture",
        query="backpropagation derivation",
        k=2,
        strategy="fixed_token",
        hierarchical=True,
    )
    assert len(moments) == 1
    assert moments[0]["chunk_id"] == 99
    assert moments[0]["macro_block_idx"] == 1
    assert moments[0]["macro_window"] == (300.0, 600.0)
