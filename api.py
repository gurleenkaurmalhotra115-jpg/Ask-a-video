"""FastAPI REST Backend for Ask-a-video (VidQuery).

Provides production REST endpoints for indexing lectures, semantic search,
confidence-gated & multi-agent-verified Q&A, and empirical benchmark reporting,
with optional Redis caching and PostgreSQL audit persistence.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from chunking import DEFAULT_PARAMS, STRATEGIES, chunk_transcript
from llm import LLMError, answer_question, log_qa
from transcript import TranscriptError, get_transcript
from vectorstore import get_client as get_qdrant_client, index_chunks, search

logger = logging.getLogger(__name__)

app = FastAPI(
    title="VidQuery (Ask-a-video) API",
    description="Timestamp-grounded Video RAG API with Retrieval-Confidence Gate, Multi-Model Fallback, and Citation Verification.",
    version="1.1.0",
)


class IndexRequest(BaseModel):
    url_or_id: str = Field(..., description="YouTube URL or 11-character video ID.")
    strategy: str = Field(default="fixed_token", description="Chunking strategy name.")
    allow_whisper_fallback: bool = Field(default=False, description="Allow Whisper ASR fallback if captions unavailable.")


class SearchRequest(BaseModel):
    video_id: str = Field(..., description="11-character YouTube video ID.")
    query: str = Field(..., min_length=1, description="Search query.")
    strategy: str = Field(default="fixed_token", description="Chunking strategy.")
    k: int = Field(default=4, ge=1, le=20, description="Number of chunks to retrieve.")


class AskRequest(BaseModel):
    video_id: str = Field(..., description="11-character YouTube video ID.")
    question: str = Field(..., min_length=1, description="User question.")
    strategy: str = Field(default="fixed_token", description="Chunking strategy.")
    k: int = Field(default=4, ge=1, le=20, description="Number of chunks to retrieve.")
    min_confidence: float | None = Field(default=None, ge=0.0, le=1.0, description="Optional confidence gate threshold override.")
    verify_grounded: bool = Field(default=False, description="Enable second-pass citation verification agent.")
    use_cache: bool = Field(default=True, description="Enable disk/Redis caching.")


def _get_redis_client() -> Any | None:
    """Connect to Redis if REDIS_URL is configured and redis package is installed."""
    redis_url = os.getenv("REDIS_URL", "").strip()
    if not redis_url:
        return None
    try:
        import redis
        client = redis.from_url(redis_url, decode_responses=True, socket_connect_timeout=1.5)
        client.ping()
        return client
    except Exception as exc:
        logger.debug("Redis unavailable (%s); continuing without Redis cache.", exc)
        return None


def _persist_qa_postgres(
    video_id: str,
    strategy: str,
    k: int,
    question: str,
    qa_result: dict[str, Any],
) -> bool:
    """Optionally persist Q&A audit row to PostgreSQL when DATABASE_URL is configured."""
    db_url = os.getenv("DATABASE_URL", "").strip()
    if not db_url:
        return False
    try:
        import psycopg
        with psycopg.connect(db_url, connect_timeout=2) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS qa_audit_logs (
                        id SERIAL PRIMARY KEY,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        video_id TEXT NOT NULL,
                        strategy TEXT NOT NULL,
                        k INTEGER NOT NULL,
                        question TEXT NOT NULL,
                        grounded BOOLEAN NOT NULL,
                        gated_out BOOLEAN NOT NULL,
                        model TEXT NOT NULL,
                        latency_sec DOUBLE PRECISION NOT NULL,
                        payload JSONB NOT NULL
                    );
                    """
                )
                cur.execute(
                    """
                    INSERT INTO qa_audit_logs (video_id, strategy, k, question, grounded, gated_out, model, latency_sec, payload)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
                    """,
                    (
                        video_id,
                        strategy,
                        k,
                        question,
                        bool(qa_result.get("grounded", False)),
                        bool(qa_result.get("gated_out", False)),
                        str(qa_result.get("model", "")),
                        float(qa_result.get("latency_sec", 0.0)),
                        json.dumps(qa_result),
                    ),
                )
            conn.commit()
        return True
    except Exception as exc:
        logger.warning("Postgres audit persistence skipped (%s); JSONL log remains active.", exc)
        return False


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness and dependency readiness endpoint."""
    redis_ok = _get_redis_client() is not None
    db_configured = bool(os.getenv("DATABASE_URL", "").strip())
    return {
        "status": "ok",
        "service": "vidquery-api",
        "qdrant_path": os.getenv("QDRANT_PATH", "qdrant_data"),
        "redis_connected": redis_ok,
        "postgres_configured": db_configured,
    }


@app.post("/videos/index")
def index_video(req: IndexRequest) -> dict[str, Any]:
    """Ingest transcript, chunk with the requested strategy, and index into Qdrant."""
    if req.strategy not in STRATEGIES:
        raise HTTPException(status_code=400, detail=f"Unknown strategy '{req.strategy}'. Choose from {list(STRATEGIES)}.")
    try:
        tx = get_transcript(req.url_or_id, allow_whisper_fallback=req.allow_whisper_fallback)
    except TranscriptError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    chunks = chunk_transcript(tx["segments"], req.strategy, **DEFAULT_PARAMS[req.strategy])
    qdrant = get_qdrant_client(os.getenv("QDRANT_PATH", "qdrant_data"))
    indexed_count = index_chunks(
        qdrant,
        video_id=tx["video_id"],
        strategy=req.strategy,
        chunks=chunks,
        transcript_source=tx["source"],
    )
    return {
        "video_id": tx["video_id"],
        "strategy": req.strategy,
        "transcript_source": tx["source"],
        "chunks_indexed": indexed_count,
    }


@app.post("/videos/search")
def search_video(req: SearchRequest) -> dict[str, Any]:
    """Retrieve top-k timestamped chunks for a video and strategy."""
    if req.strategy not in STRATEGIES:
        raise HTTPException(status_code=400, detail=f"Unknown strategy '{req.strategy}'. Choose from {list(STRATEGIES)}.")
    qdrant = get_qdrant_client(os.getenv("QDRANT_PATH", "qdrant_data"))
    results = search(
        client=qdrant,
        query=req.query,
        video_id=req.video_id,
        strategy=req.strategy,
        k=req.k,
    )
    return {
        "video_id": req.video_id,
        "strategy": req.strategy,
        "k": req.k,
        "results": results,
    }


@app.post("/videos/ask")
def ask_video(req: AskRequest) -> dict[str, Any]:
    """Retrieve chunks, apply retrieval-confidence gate, generate grounded answer, and optionally verify."""
    if req.strategy not in STRATEGIES:
        raise HTTPException(status_code=400, detail=f"Unknown strategy '{req.strategy}'. Choose from {list(STRATEGIES)}.")

    cache_key = hashlib.sha256(
        f"{req.video_id}|{req.strategy}|{req.k}|{req.min_confidence}|{req.verify_grounded}|{req.question}".encode("utf-8")
    ).hexdigest()

    redis_client = _get_redis_client() if req.use_cache else None
    if redis_client is not None:
        try:
            cached = redis_client.get(f"vidquery:qa:{cache_key}")
            if cached:
                payload = json.loads(cached)
                payload["redis_cache_hit"] = True
                return payload
        except Exception:
            pass

    qdrant = get_qdrant_client(os.getenv("QDRANT_PATH", "qdrant_data"))
    retrieved = search(
        client=qdrant,
        query=req.question,
        video_id=req.video_id,
        strategy=req.strategy,
        k=req.k,
    )
    if not retrieved:
        raise HTTPException(status_code=404, detail=f"No indexed chunks found for video_id='{req.video_id}' and strategy='{req.strategy}'.")

    try:
        qa_res = answer_question(
            question=req.question,
            chunks=retrieved,
            use_cache=req.use_cache,
            min_confidence=req.min_confidence,
            verify_grounded=req.verify_grounded,
        )
    except LLMError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    log_qa(
        video_id=req.video_id,
        strategy=req.strategy,
        k=req.k,
        question=req.question,
        retrieved_chunks=retrieved,
        llm_result=qa_res,
    )
    pg_persisted = _persist_qa_postgres(req.video_id, req.strategy, req.k, req.question, qa_res)

    response_payload = {
        "video_id": req.video_id,
        "strategy": req.strategy,
        "k": req.k,
        "redis_cache_hit": False,
        "postgres_persisted": pg_persisted,
        **qa_res,
    }

    if redis_client is not None:
        try:
            redis_client.setex(f"vidquery:qa:{cache_key}", 3600, json.dumps(response_payload))
        except Exception:
            pass

    return response_payload


@app.get("/benchmarks")
def get_benchmarks() -> dict[str, Any]:
    """Return empirical retrieval and generation benchmark summaries from results/."""
    out: dict[str, Any] = {}
    retrieval_csv = "results/retrieval_summary.csv"
    generation_csv = "results/generation_eval_summary.csv"
    if os.path.exists(retrieval_csv):
        out["retrieval_summary"] = pd.read_csv(retrieval_csv).to_dict(orient="records")
    if os.path.exists(generation_csv):
        out["generation_summary"] = pd.read_csv(generation_csv).to_dict(orient="records")
    return out
