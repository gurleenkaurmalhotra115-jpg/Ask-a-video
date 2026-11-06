"""Unit tests for Qdrant vector store operations."""

from __future__ import annotations

import numpy as np
import pytest
from qdrant_client import QdrantClient

from vectorstore import delete_video, index_chunks, list_indexed, search


def fake_embedder(texts: list[str]) -> np.ndarray:
    """Deterministic mock embedder: hashes words into a normalized 384-dim vector."""
    vectors = []
    for text in texts:
        vec = np.zeros(384, dtype=np.float32)
        words = text.lower().split()
        for w in words:
            idx = abs(hash(w)) % 384
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        else:
            vec[0] = 1.0
        vectors.append(vec)
    return np.asarray(vectors, dtype=np.float32)


@pytest.fixture
def memory_client() -> QdrantClient:
    """Return an isolated in-memory QdrantClient instance."""
    return QdrantClient(":memory:")


@pytest.fixture
def sample_chunks() -> list[dict[str, any]]:
    """Return a small set of test chunks."""
    return [
        {
            "chunk_id": 0,
            "strategy": "fixed_token",
            "text": "Machine learning models learn patterns from training data.",
            "start": 0.0,
            "end": 10.0,
            "n_words": 8,
        },
        {
            "chunk_id": 1,
            "strategy": "fixed_token",
            "text": "Neural networks utilize backpropagation to update weights.",
            "start": 10.0,
            "end": 20.0,
            "n_words": 7,
        },
        {
            "chunk_id": 2,
            "strategy": "fixed_token",
            "text": "Transformers rely on attention mechanisms to model context.",
            "start": 20.0,
            "end": 30.0,
            "n_words": 8,
        },
    ]


def test_index_and_filtered_search(
    memory_client: QdrantClient, sample_chunks: list[dict[str, any]]
) -> None:
    """Test that search only returns chunks matching both video_id and strategy."""
    # Index video1 with fixed_token
    index_chunks(memory_client, "vid_1", "fixed_token", sample_chunks, embed_fn=fake_embedder)

    # Index video1 with sentence (different strategy)
    alt_strategy_chunks = [
        {
            "chunk_id": 0,
            "strategy": "sentence",
            "text": "Transformers rely on attention mechanisms to model context.",
            "start": 0.0,
            "end": 15.0,
            "n_words": 8,
        }
    ]
    index_chunks(memory_client, "vid_1", "sentence", alt_strategy_chunks, embed_fn=fake_embedder)

    # Index video2 with fixed_token (different video)
    alt_video_chunks = [
        {
            "chunk_id": 0,
            "strategy": "fixed_token",
            "text": "Completely unrelated content about cooking and recipes.",
            "start": 0.0,
            "end": 15.0,
            "n_words": 7,
        }
    ]
    index_chunks(memory_client, "vid_2", "fixed_token", alt_video_chunks, embed_fn=fake_embedder)

    # Search specifically for vid_1 + fixed_token
    results = search(
        memory_client,
        query="neural networks backpropagation",
        video_id="vid_1",
        strategy="fixed_token",
        k=4,
        embed_fn=fake_embedder,
    )

    assert len(results) > 0
    # Top result should be chunk_id 1 which matches neural networks backpropagation
    assert results[0]["chunk_id"] == 1
    assert "Neural networks" in results[0]["text"]
    assert results[0]["start"] == 10.0
    assert results[0]["end"] == 20.0

    # Verify no chunks from vid_2 or sentence strategy leaked in
    assert len(results) <= len(sample_chunks)


def test_reindexing_prevents_duplicates(
    memory_client: QdrantClient, sample_chunks: list[dict[str, any]]
) -> None:
    """Test that indexing the same video and strategy twice overwrites and does not duplicate."""
    # First index
    index_chunks(memory_client, "vid_dup", "fixed_token", sample_chunks, embed_fn=fake_embedder)
    indexed_1 = list_indexed(memory_client)
    entry_1 = [e for e in indexed_1 if e["video_id"] == "vid_dup" and e["strategy"] == "fixed_token"]
    assert len(entry_1) == 1
    assert entry_1[0]["n_chunks"] == 3

    # Re-index with same chunks
    index_chunks(memory_client, "vid_dup", "fixed_token", sample_chunks, embed_fn=fake_embedder)
    indexed_2 = list_indexed(memory_client)
    entry_2 = [e for e in indexed_2 if e["video_id"] == "vid_dup" and e["strategy"] == "fixed_token"]
    assert len(entry_2) == 1
    assert entry_2[0]["n_chunks"] == 3


def test_results_sorted_by_score(
    memory_client: QdrantClient, sample_chunks: list[dict[str, any]]
) -> None:
    """Test that search results are strictly ordered by score descending."""
    index_chunks(memory_client, "vid_sort", "fixed_token", sample_chunks, embed_fn=fake_embedder)

    results = search(
        memory_client,
        query="machine learning training patterns",
        video_id="vid_sort",
        strategy="fixed_token",
        k=3,
        embed_fn=fake_embedder,
    )

    assert len(results) == 3
    for i in range(len(results) - 1):
        assert results[i]["score"] >= results[i + 1]["score"]


def test_delete_video_removes_points(
    memory_client: QdrantClient, sample_chunks: list[dict[str, any]]
) -> None:
    """Test that delete_video removes all indexed points for the specified video."""
    index_chunks(memory_client, "vid_del_1", "fixed_token", sample_chunks, embed_fn=fake_embedder)
    index_chunks(memory_client, "vid_del_2", "fixed_token", sample_chunks, embed_fn=fake_embedder)

    assert any(e["video_id"] == "vid_del_1" for e in list_indexed(memory_client))
    assert any(e["video_id"] == "vid_del_2" for e in list_indexed(memory_client))

    delete_video(memory_client, "vid_del_1")

    indexed_after = list_indexed(memory_client)
    assert not any(e["video_id"] == "vid_del_1" for e in indexed_after)
    assert any(e["video_id"] == "vid_del_2" for e in indexed_after)

    # Search on deleted video should yield 0 results
    results = search(
        memory_client,
        query="learning",
        video_id="vid_del_1",
        strategy="fixed_token",
        k=4,
        embed_fn=fake_embedder,
    )
    assert len(results) == 0


def test_transcript_source_isolation(
    memory_client: QdrantClient, sample_chunks: list[dict[str, any]]
) -> None:
    """Test that indexing with transcript_source separates captions and whisper results."""
    whisper_chunk = [
        {
            "chunk_id": 0,
            "strategy": "fixed_token",
            "text": "This is raw ASR audio output from local Whisper model.",
            "start": 5.0,
            "end": 15.0,
            "n_words": 9,
        }
    ]
    # Index both sources for same video and strategy
    index_chunks(
        memory_client, "vid_multi_src", "fixed_token", sample_chunks,
        transcript_source="captions", embed_fn=fake_embedder
    )
    index_chunks(
        memory_client, "vid_multi_src", "fixed_token", whisper_chunk,
        transcript_source="whisper", embed_fn=fake_embedder
    )

    cap_res = search(
        memory_client, "learning", "vid_multi_src", "fixed_token",
        transcript_source="captions", embed_fn=fake_embedder
    )
    wh_res = search(
        memory_client, "audio", "vid_multi_src", "fixed_token",
        transcript_source="whisper", embed_fn=fake_embedder
    )

    assert len(cap_res) > 0
    assert all(r["transcript_source"] == "captions" for r in cap_res)
    assert len(wh_res) == 1
    assert wh_res[0]["transcript_source"] == "whisper"
    assert "Whisper model" in wh_res[0]["text"]

