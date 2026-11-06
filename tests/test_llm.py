"""Unit tests for LLM grounded answering, citation mapping, retries, fallback, and caching."""

from __future__ import annotations

import os
import tempfile
import pytest
from google.genai.errors import APIError

from llm import LLMError, answer_question, build_context, get_client


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
