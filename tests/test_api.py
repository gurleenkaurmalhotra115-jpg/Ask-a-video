"""Unit tests for the FastAPI REST backend endpoints (api.py)."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

import api
from api import AskRequest, SearchRequest


def test_health_endpoint() -> None:
    """Test /health returns service readiness payload."""
    data = api.health()
    assert data["status"] == "ok"
    assert data["service"] == "vidquery-api"


def test_benchmarks_endpoint() -> None:
    """Test /benchmarks exposes retrieval and generation CSV metrics."""
    data = api.get_benchmarks()
    assert "retrieval_summary" in data
    assert len(data["retrieval_summary"]) >= 1


def test_search_rejects_unknown_strategy() -> None:
    """Test /videos/search validates chunking strategy."""
    req = SearchRequest(video_id="dQw4w9WgXcQ", query="test", strategy="invalid_strat")
    with pytest.raises(HTTPException) as exc_info:
        api.search_video(req)
    assert exc_info.value.status_code == 400


def test_ask_endpoint_confidence_gate_and_verification(monkeypatch) -> None:
    """Test /videos/ask integrates search, confidence gate, verification, and logging."""
    fake_chunks = [
        {
            "chunk_id": 1,
            "text": "Backpropagation computes gradients via the chain rule.",
            "start": 10.0,
            "end": 25.0,
            "score": 0.81,
        }
    ]
    monkeypatch.setattr(api, "get_qdrant_client", lambda path="qdrant_data": object())
    monkeypatch.setattr(api, "search", lambda **kwargs: fake_chunks)
    monkeypatch.setattr(api, "log_qa", lambda **kwargs: None)
    monkeypatch.setattr(
        api,
        "answer_question",
        lambda **kwargs: {
            "answer": "Backpropagation uses the chain rule [1].",
            "grounded": True,
            "cited": [1],
            "sources": [{"n": 1, "start": 10.0, "end": 25.0, "text": fake_chunks[0]["text"]}],
            "model": "gemini-3.8-flash",
            "latency_sec": 0.2,
            "fallback_used": False,
            "attempts": 1,
            "cache_hit": False,
            "gated_out": False,
            "confidence": {"passed": True, "top_score": 0.81, "level": "high"},
            "verification": {"verified": True, "unsupported_claims": [], "confidence": "high"},
        },
    )

    req = AskRequest(
        video_id="demo_vid_01",
        question="How does backprop work?",
        strategy="fixed_token",
        verify_grounded=True,
    )
    body = api.ask_video(req)
    assert body["grounded"] is True
    assert body["verification"]["verified"] is True
    assert body["postgres_persisted"] is False
