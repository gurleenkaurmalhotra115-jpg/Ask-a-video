"""Local Qdrant vector store management for video transcript chunks."""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable
import uuid

from qdrant_client import QdrantClient, models

COLLECTION = "video_chunks"
VECTOR_SIZE = 384
POINT_NAMESPACE = uuid.NAMESPACE_DNS


def get_client(path: str = "qdrant_data") -> QdrantClient:
    """Return a local disk-persisted QdrantClient instance."""
    return QdrantClient(path=path)


def ensure_collection(client: QdrantClient) -> None:
    """Create the video_chunks collection if it does not already exist."""
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )


def index_chunks(
    client: QdrantClient,
    video_id: str,
    strategy: str,
    chunks: list[dict[str, Any]],
    transcript_source: str = "captions",
    embed_fn: Callable[[list[str]], Any] | None = None,
) -> int:
    """Index chunks for a given video, strategy, and transcript source into Qdrant.

    Deletes any previously indexed points for (video_id, strategy, transcript_source) to avoid duplicates,
    computes embeddings in a single batch, and upserts points with payload metadata.

    Args:
        client: QdrantClient instance.
        video_id: Video identifier.
        strategy: Chunking strategy name.
        chunks: List of chunk dictionaries.
        transcript_source: Source of transcript ('captions' or 'whisper').
        embed_fn: Optional embedding function taking list[str] and returning vectors.

    Returns:
        int: Number of chunks indexed.
    """
    ensure_collection(client)

    if embed_fn is None:
        from embeddings import embed_texts

        embed_fn = embed_texts

    # Delete existing points for this (video_id, strategy, transcript_source) tuple
    delete_filter = [
        models.FieldCondition(
            key="video_id",
            match=models.MatchValue(value=video_id),
        ),
        models.FieldCondition(
            key="strategy",
            match=models.MatchValue(value=strategy),
        ),
    ]
    if transcript_source:
        delete_filter.append(
            models.FieldCondition(
                key="transcript_source",
                match=models.MatchValue(value=transcript_source),
            )
        )

    client.delete(
        collection_name=COLLECTION,
        points_selector=models.FilterSelector(
            filter=models.Filter(must=delete_filter)
        ),
    )

    if not chunks:
        return 0

    texts = [c["text"] for c in chunks]
    raw_vectors = embed_fn(texts)
    vectors = [
        vec.tolist() if hasattr(vec, "tolist") else list(vec)
        for vec in raw_vectors
    ]

    points = [
        models.PointStruct(
            id=str(uuid.uuid5(POINT_NAMESPACE, f"{video_id}|{transcript_source}|{strategy}|{c['chunk_id']}")),
            vector=vectors[i],
            payload={
                "text": c["text"],
                "start": float(c["start"]),
                "end": float(c["end"]),
                "video_id": video_id,
                "strategy": strategy,
                "transcript_source": transcript_source,
                "chunk_id": int(c["chunk_id"]),
                "n_words": int(c["n_words"]),
            },
        )
        for i, c in enumerate(chunks)
    ]

    client.upsert(collection_name=COLLECTION, points=points)
    return len(points)


def search(
    client: QdrantClient,
    query: str,
    video_id: str,
    strategy: str,
    transcript_source: str | None = None,
    k: int = 4,
    embed_fn: Callable[[list[str]], Any] | None = None,
) -> list[dict[str, Any]]:
    """Retrieve top-k chunks matching the query for a specific video and strategy.

    Args:
        client: QdrantClient instance.
        query: User search question or query text.
        video_id: Video identifier filter.
        strategy: Chunking strategy filter.
        transcript_source: Optional transcript source filter ('captions' or 'whisper').
        k: Maximum number of results to return.
        embed_fn: Optional embedding function taking list[str] and returning vectors.

    Returns:
        List of dicts: {score, text, start, end, chunk_id}, sorted by score descending.
    """
    if not client.collection_exists(COLLECTION):
        return []

    if embed_fn is None:
        from embeddings import embed_texts

        embed_fn = embed_texts

    query_vectors = embed_fn([query])
    query_vector = query_vectors[0]
    if hasattr(query_vector, "tolist"):
        query_vector = query_vector.tolist()
    else:
        query_vector = list(query_vector)

    must_filters = [
        models.FieldCondition(
            key="video_id",
            match=models.MatchValue(value=video_id),
        ),
        models.FieldCondition(
            key="strategy",
            match=models.MatchValue(value=strategy),
        ),
    ]
    if transcript_source:
        must_filters.append(
            models.FieldCondition(
                key="transcript_source",
                match=models.MatchValue(value=transcript_source),
            )
        )

    search_filter = models.Filter(must=must_filters)

    if hasattr(client, "query_points"):
        res = client.query_points(
            collection_name=COLLECTION,
            query=query_vector,
            query_filter=search_filter,
            limit=k,
        )
        points = res.points
    else:
        points = client.search(
            collection_name=COLLECTION,
            query_vector=query_vector,
            query_filter=search_filter,
            limit=k,
        )

    results = []
    for pt in points:
        payload = pt.payload or {}
        results.append({
            "score": float(pt.score),
            "text": str(payload.get("text", "")),
            "start": float(payload.get("start", 0.0)),
            "end": float(payload.get("end", 0.0)),
            "chunk_id": int(payload.get("chunk_id", 0)),
            "transcript_source": str(payload.get("transcript_source", "captions")),
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def list_indexed(client: QdrantClient) -> list[dict[str, Any]]:
    """Return summary of all indexed videos and strategies with chunk counts.

    Returns:
        List of dicts: {"video_id": str, "strategy": str, "transcript_source": str, "n_chunks": int}
    """
    if not client.collection_exists(COLLECTION):
        return []

    records: list[tuple[str, str, str]] = []
    offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=COLLECTION,
            limit=250,
            with_payload=["video_id", "strategy", "transcript_source"],
            with_vectors=False,
            offset=offset,
        )
        for pt in points:
            if pt.payload:
                vid = pt.payload.get("video_id")
                strat = pt.payload.get("strategy")
                src = pt.payload.get("transcript_source", "captions")
                if vid and strat:
                    records.append((str(vid), str(strat), str(src)))

        if next_offset is None:
            break
        offset = next_offset

    counts = Counter(records)
    return [
        {"video_id": vid, "strategy": strat, "transcript_source": src, "n_chunks": count}
        for (vid, strat, src), count in counts.items()
    ]


def delete_video(client: QdrantClient, video_id: str) -> None:
    """Delete all indexed chunks associated with a specific video ID."""
    if not client.collection_exists(COLLECTION):
        return

    client.delete(
        collection_name=COLLECTION,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="video_id",
                        match=models.MatchValue(value=video_id),
                    ),
                ]
            )
        ),
    )
