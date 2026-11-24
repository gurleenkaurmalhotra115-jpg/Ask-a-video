"""Unit tests for LLM grounded answering, citation mapping, retries, fallback, and caching."""

from __future__ import annotations

import os
import tempfile
import pytest
from google.genai.errors import APIError

from llm import (
    LLMError,
    answer_question,
    assess_retrieval_confidence,
    build_context,
    get_client,
    get_model_chain,
)


@pytest.fixture
def sample_chunks() -> list[dict[str, any]]:
    """Fixture providing standard retrieved test chunks with timestamps."""
    return [
        {
            "chunk_id": 101,
            "text": "Supervised learning algorithms require labeled datasets for training.",
            "start": 15.0,
            "end": 35.0,
            "score": 0.88,
        },
        {
            "chunk_id": 102,
            "text": "Reinforcement learning learns via rewards and penalties in an environment.",
            "start": 40.0,
            "end": 65.0,
            "score": 0.75,
        },
        {
            "chunk_id": 103,
            "text": "Unsupervised learning discovers hidden patterns in unlabeled data.",
            "start": 70.0,
            "end": 95.0,
            "score": 0.71,
        },
    ]


def test_build_context_format(sample_chunks: list[dict[str, any]]) -> None:
    """Test that context blocks follow [n] (mm:ss-mm:ss) text format."""
    context = build_context(sample_chunks)
    lines = context.split("\n\n")
    assert len(lines) == 3
    assert lines[0].startswith("[1] (00:15-00:35) Supervised learning")
    assert lines[1].startswith("[2] (00:40-01:05) Reinforcement learning")
    assert lines[2].startswith("[3] (01:10-01:35) Unsupervised learning")


def test_citation_parsing_and_mapping(sample_chunks: list[dict[str, any]]) -> None:
    """Test parsing of multiple citations like [1][3] into cited list and sources."""
    fake_response = (
        "Supervised learning relies on labeled data [1], whereas unsupervised learning "
        "finds patterns without labels [3]."
    )
    result = answer_question(
        question="What is the difference?",
        chunks=sample_chunks,
        generate_fn=lambda prompt: fake_response,
        use_cache=False,
    )

    assert result["grounded"] is True
    assert result["cited"] == [1, 3]
    assert len(result["sources"]) == 2
    assert result["sources"][0]["n"] == 1
    assert result["sources"][0]["start"] == 15.0
    assert result["sources"][0]["end"] == 35.0
    assert result["sources"][1]["n"] == 3
    assert result["sources"][1]["start"] == 70.0
    assert result["sources"][1]["end"] == 95.0


def test_out_of_range_citation_ignored(sample_chunks: list[dict[str, any]]) -> None:
    """Test that invented or out-of-range citations like [99] are filtered out."""
    fake_response = "Supervised learning uses labels [1], and an invented source [99]."
    result = answer_question(
        question="How does it learn?",
        chunks=sample_chunks,
        generate_fn=lambda prompt: fake_response,
        use_cache=False,
    )

    assert result["grounded"] is True
    assert result["cited"] == [1]
    assert len(result["sources"]) == 1
    assert result["sources"][0]["n"] == 1


def test_not_in_video_response(sample_chunks: list[dict[str, any]]) -> None:
    """Test that NOT_IN_VIDEO yields grounded=False and empty sources."""
    result = answer_question(
        question="Who won the 2022 World Cup?",
        chunks=sample_chunks,
        generate_fn=lambda prompt: "NOT_IN_VIDEO",
        use_cache=False,
    )

    assert result["grounded"] is False
    assert result["cited"] == []
    assert result["sources"] == []


def test_sources_timestamps_come_from_metadata(sample_chunks: list[dict[str, any]]) -> None:
    """Test that sources timestamps always come strictly from chunk metadata, not LLM text."""
    fake_response = "At 99:99 we see that supervised learning uses labels [1]."
    result = answer_question(
        question="Tell me about labels",
        chunks=sample_chunks,
        generate_fn=lambda prompt: fake_response,
        use_cache=False,
    )

    assert result["grounded"] is True
    assert len(result["sources"]) == 1
    assert result["sources"][0]["start"] == 15.0
    assert result["sources"][0]["end"] == 35.0


def test_fallback_to_top_chunk_when_no_citation(sample_chunks: list[dict[str, any]]) -> None:
    """Test that a grounded answer without explicit citations falls back to chunk 1."""
    fake_response = "Supervised learning requires labeled data."
    result = answer_question(
        question="How does supervised learning work?",
        chunks=sample_chunks,
        generate_fn=lambda prompt: fake_response,
        use_cache=False,
    )

    assert result["grounded"] is True
    assert result["cited"] == [1]
    assert len(result["sources"]) == 1
    assert result["sources"][0]["n"] == 1
    assert result["sources"][0]["start"] == 15.0


def test_missing_api_key_raises_llm_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that missing GEMINI_API_KEY raises a clear user-readable LLMError."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("llm.load_dotenv", lambda: None)

    with pytest.raises(LLMError) as exc_info:
        get_client()

    assert "Add GEMINI_API_KEY to your .env file" in str(exc_info.value)


# -------------------- Robustness and Retry Tests --------------------


def test_retry_on_503_twice_then_success(sample_chunks: list[dict[str, any]]) -> None:
    """Test that two 503 failures are retried and success on 3rd attempt succeeds with attempts == 3."""
    calls = 0

    def fake_generator(prompt: str, model: str = "") -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise APIError(503, {"error": {"message": "Service Unavailable", "status": "UNAVAILABLE"}})
        return "Model recovered and answered [1]."

    res = answer_question(
        question="What happened?",
        chunks=sample_chunks,
        generate_fn=fake_generator,
        use_cache=False,
        sleep_fn=lambda s: None,
    )

    assert calls == 3
    assert res["attempts"] == 3
    assert res["fallback_used"] is False
    assert "Model recovered" in res["answer"]


def test_503_always_triggers_fallback_model(
    sample_chunks: list[dict[str, any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that exhausting main model retries on 503 triggers the fallback model."""
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.8-flash")
    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash")

    def fake_generator(prompt: str, model: str = "gemini-3.8-flash") -> str:
        if model == "gemini-3.8-flash":
            raise APIError(503, {"error": {"message": "Primary overloaded", "status": "UNAVAILABLE"}})
        return f"Grounded response from fallback {model} [1]."

    res = answer_question(
        question="Explain concept",
        chunks=sample_chunks,
        generate_fn=fake_generator,
        use_cache=False,
        sleep_fn=lambda s: None,
    )

    assert res["fallback_used"] is True
    assert res["model"] == "gemini-2.5-flash"
    assert "fallback gemini-2.5-flash" in res["answer"]


def test_503_always_without_fallback_raises_friendly_error(
    sample_chunks: list[dict[str, any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that repeated 503 errors without fallback raise friendly overloaded message."""
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.8-flash")
    monkeypatch.delenv("GEMINI_FALLBACK_MODEL", raising=False)

    def fake_generator(prompt: str, model: str = "") -> str:
        raise APIError(503, {"error": {"message": "High demand", "status": "UNAVAILABLE"}})

    with pytest.raises(LLMError) as exc_info:
        answer_question(
            question="Explain concept",
            chunks=sample_chunks,
            generate_fn=fake_generator,
            use_cache=False,
            sleep_fn=lambda s: None,
        )

    msg = str(exc_info.value)
    assert "overloaded" in msg.lower()
    assert "tried 4 times" in msg


def test_404_is_not_retried(
    sample_chunks: list[dict[str, any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that a 404 model not found error is not retried (attempts == 1)."""
    calls = 0

    def fake_generator(prompt: str, model: str = "") -> str:
        nonlocal calls
        calls += 1
        raise APIError(404, {"error": {"message": "models/invalid not found", "status": "NOT_FOUND"}})

    with pytest.raises(LLMError) as exc_info:
        answer_question(
            question="Check model",
            chunks=sample_chunks,
            generate_fn=fake_generator,
            use_cache=False,
            sleep_fn=lambda s: None,
        )

    assert calls == 1
    assert "not found" in str(exc_info.value).lower()


def test_disk_cache_hit_avoids_generator_call(sample_chunks: list[dict[str, any]]) -> None:
    """Test that identical second call returns cached result without invoking generator."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_cache_file = tmp.name

    try:
        calls = 0

        def fake_generator(prompt: str, model: str = "") -> str:
            nonlocal calls
            calls += 1
            return "First generated answer [1]."

        # First call (cache miss)
        res1 = answer_question(
            question="What is the answer?",
            chunks=sample_chunks,
            generate_fn=fake_generator,
            use_cache=True,
            cache_path=tmp_cache_file,
        )
        assert res1["cache_hit"] is False
        assert calls == 1
        assert res1["answer"] == "First generated answer [1]."

        # Second call with same question and chunks (cache hit)
        res2 = answer_question(
            question="What is the answer?",
            chunks=sample_chunks,
            generate_fn=fake_generator,
            use_cache=True,
            cache_path=tmp_cache_file,
        )
        assert res2["cache_hit"] is True
        assert calls == 1  # Generator was NOT invoked a second time!
        assert res2["answer"] == "First generated answer [1]."
    finally:
        if os.path.exists(tmp_cache_file):
            os.remove(tmp_cache_file)


# -------------------- Multi-Model Fallback & Confidence Gate Tests --------------------


def test_get_model_chain_deduplicates_and_preserves_legacy_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test ordered multi-model chain parsing and deduplication."""
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.8-flash")
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "gemini-3.8-flash, gemini-2.5-flash, gemini-2.5-flash-lite, gemini-2.5-flash")
    monkeypatch.delenv("GEMINI_FALLBACK_MODEL", raising=False)
    assert get_model_chain() == ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.5-flash-lite"]

    monkeypatch.delenv("GEMINI_FALLBACK_MODELS", raising=False)
    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash")
    assert get_model_chain() == ["gemini-3.8-flash", "gemini-2.5-flash"]


def test_multi_model_fallback_chain_cascades(
    sample_chunks: list[dict[str, any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test cascading through multiple fallback models when earlier models fail transiently."""
    monkeypatch.setenv("GEMINI_MODEL", "primary-m")
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "fb-1, fb-2")
    visited_models: list[str] = []

    def fake_generator(prompt: str, model: str = "") -> str:
        visited_models.append(model)
        if model == "primary-m":
            raise APIError(503, {"error": {"message": "Overloaded", "status": "UNAVAILABLE"}})
        if model == "fb-1":
            raise APIError(429, {"error": {"message": "Quota exceeded", "status": "RESOURCE_EXHAUSTED"}})
        return f"Answered by {model} [1]."

    res = answer_question(
        question="Cascade test",
        chunks=sample_chunks,
        generate_fn=fake_generator,
        use_cache=False,
        sleep_fn=lambda s: None,
    )

    assert res["fallback_used"] is True
    assert res["model"] == "fb-2"
    assert res["grounded"] is True
    assert "primary-m" in visited_models
    assert "fb-1" in visited_models
    assert "fb-2" in visited_models


def test_assess_retrieval_confidence_levels() -> None:
    """Test confidence level classification across score regimes."""
    assert assess_retrieval_confidence([])["level"] == "none"
    assert assess_retrieval_confidence([])["passed"] is False

    no_scores = [{"text": "Chunk without score", "start": 0.0, "end": 5.0}]
    res_unknown = assess_retrieval_confidence(no_scores)
    assert res_unknown["level"] == "unknown"
    assert res_unknown["passed"] is True

    low = assess_retrieval_confidence([{"text": "x", "start": 0.0, "end": 5.0, "score": 0.14}], min_confidence=0.22)
    assert low["level"] == "low"
    assert low["passed"] is False

    med = assess_retrieval_confidence([{"text": "x", "start": 0.0, "end": 5.0, "score": 0.28}], min_confidence=0.22)
    assert med["level"] == "medium"
    assert med["passed"] is True

    high = assess_retrieval_confidence([{"text": "x", "start": 0.0, "end": 5.0, "score": 0.52}], min_confidence=0.22)
    assert high["level"] == "high"
    assert high["passed"] is True


def test_confidence_gate_blocks_low_similarity_without_llm_call() -> None:
    """Test that weak retrieval scores short-circuit before calling the LLM."""
    weak_chunks = [
        {"chunk_id": 1, "text": "Unrelated snippet", "start": 0.0, "end": 10.0, "score": 0.13},
        {"chunk_id": 2, "text": "Another weak match", "start": 10.0, "end": 20.0, "score": 0.09},
    ]
    calls = 0

    def fake_generator(prompt: str, model: str = "") -> str:
        nonlocal calls
        calls += 1
        return "Should not be reached [1]."

    res = answer_question(
        question="Out of domain question",
        chunks=weak_chunks,
        generate_fn=fake_generator,
        use_cache=False,
        min_confidence=0.22,
    )

    assert calls == 0
    assert res["gated_out"] is True
    assert res["grounded"] is False
    assert res["answer"] == "NOT_IN_VIDEO"
    assert res["model"] == "retrieval-confidence-gate"
    assert res["confidence"]["level"] == "low"


def test_confidence_gate_bypassed_when_threshold_zero() -> None:
    """Test that min_confidence=0.0 allows LLM execution even on low-score chunks."""
    weak_chunks = [{"chunk_id": 1, "text": "Marginal snippet", "start": 0.0, "end": 10.0, "score": 0.13}]
    calls = 0

    def fake_generator(prompt: str, model: str = "") -> str:
        nonlocal calls
        calls += 1
        return "Marginal answer [1]."

    res = answer_question(
        question="Test bypass",
        chunks=weak_chunks,
        generate_fn=fake_generator,
        use_cache=False,
        min_confidence=0.0,
    )

    assert calls == 1
    assert res["gated_out"] is False
    assert res["grounded"] is True


def test_multi_agent_citation_verification(sample_chunks: list[dict[str, any]]) -> None:
    """Test second-pass verification agent attaches verification verdict."""
    res_ok = answer_question(
        question="What is supervised learning?",
        chunks=sample_chunks,
        generate_fn=lambda p: "Supervised learning uses labeled datasets [1].",
        verify_grounded=True,
        verify_fn=lambda vp: '{"verified": true, "unsupported_claims": [], "confidence": "high"}',
        use_cache=False,
    )
    assert res_ok["grounded"] is True
    assert res_ok["verification"]["verified"] is True
    assert res_ok["verification"]["confidence"] == "high"

    res_flagged = answer_question(
        question="What is supervised learning?",
        chunks=sample_chunks,
        generate_fn=lambda p: "Supervised learning requires quantum GPUs [1].",
        verify_grounded=True,
        verify_fn=lambda vp: '{"verified": false, "unsupported_claims": ["requires quantum GPUs"], "confidence": "low"}',
        use_cache=False,
    )
    assert res_flagged["verification"]["verified"] is False
    assert "requires quantum GPUs" in res_flagged["verification"]["unsupported_claims"]
