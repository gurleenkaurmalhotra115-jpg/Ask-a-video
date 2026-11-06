"""Unit tests for pluggable transcript chunking strategies."""

from __future__ import annotations

import pytest

from chunking import chunk_transcript, split_sentences, STRATEGIES


def create_synthetic_transcript(
    n_segments: int = 100, seg_duration: float = 3.0, with_punctuation: bool = True
) -> list[dict[str, any]]:
    """Build a synthetic transcript with n segments of equal duration."""
    phrases = [
        "In this video we explore how neural networks learn representations from data",
        "Gradient descent iteratively updates model weights to minimize the loss function",
        "Attention mechanisms allow models to focus dynamically on relevant tokens in the sequence",
        "Transformers utilize self attention to model long range dependencies effectively",
        "Cosine similarity measures the angle between high dimensional embedding vectors",
        "Chunking text into coherent units improves retrieval performance in question answering systems",
    ]
    segments = []
    for i in range(n_segments):
        start = round(i * seg_duration, 2)
        base_phrase = phrases[i % len(phrases)]
        if with_punctuation:
            # Add punctuation at regular intervals
            if (i + 1) % 4 == 0:
                text = f"{base_phrase}."
            elif (i + 1) % 7 == 0:
                text = f"Can this approach generalize well to unseen data?"
            else:
                text = base_phrase
        else:
            text = base_phrase.lower().replace(".", "").replace("?", "")

        segments.append({
            "start": start,
            "duration": seg_duration,
            "text": text,
        })
    return segments


@pytest.fixture
def synthetic_transcript() -> list[dict[str, any]]:
    """Return a standard synthetic transcript with punctuation."""
    return create_synthetic_transcript(n_segments=100, seg_duration=3.0, with_punctuation=True)


@pytest.fixture
def no_punct_transcript() -> list[dict[str, any]]:
    """Return a synthetic transcript without any punctuation."""
    return create_synthetic_transcript(n_segments=100, seg_duration=3.0, with_punctuation=False)


@pytest.mark.parametrize("strategy", ["fixed_token", "sentence", "semantic", "time_window"])
def test_all_strategies_properties(
    strategy: str, synthetic_transcript: list[dict[str, any]]
) -> None:
    """Verify that all strategies produce valid chunks covering all segments in time order."""
    chunks = chunk_transcript(synthetic_transcript, strategy)

    # 1. Non-empty output
    assert len(chunks) > 0, f"Strategy {strategy} returned no chunks."

    # 2. start < end for all chunks
    for chunk in chunks:
        assert chunk["start"] < chunk["end"], (
            f"Chunk {chunk['chunk_id']} has start {chunk['start']} >= end {chunk['end']} in strategy {strategy}"
        )
        assert chunk["n_words"] > 0
        assert len(chunk["text"].strip()) > 0

    # 3. Chunks must be sorted by start time
    for i in range(len(chunks) - 1):
        assert chunks[i]["start"] <= chunks[i + 1]["start"], (
            f"Chunks not sorted by start in strategy {strategy}: {chunks[i]['start']} > {chunks[i+1]['start']}"
        )

    # 4. Full coverage of all segments:
    # First chunk starts at or before the first segment, and last chunk ends at or after the last segment.
    # Every segment's start time must be >= first chunk start and <= last chunk start.
    first_seg_start = synthetic_transcript[0]["start"]
    last_seg_end = round(
        synthetic_transcript[-1]["start"] + synthetic_transcript[-1]["duration"], 2
    )
    assert chunks[0]["start"] <= first_seg_start + 1e-2
    assert chunks[-1]["end"] >= last_seg_end - 1e-2

    # Every individual segment must be covered by at least one chunk
    for seg in synthetic_transcript:
        s_start = seg["start"]
        s_end = round(seg["start"] + seg["duration"], 2)
        covered = any(
            c["start"] <= s_start + 1e-2 and c["end"] >= s_end - 1e-2 for c in chunks
        )
        assert covered, f"Segment [{s_start}, {s_end}] is not covered by any chunk in {strategy}."


def test_time_window_length_limit(synthetic_transcript: list[dict[str, any]]) -> None:
    """Assert that no chunk in time_window is longer than window_sec + one segment."""
    window_sec = 40.0
    overlap_sec = 8.0
    one_segment_duration = 3.0
    chunks = chunk_transcript(
        synthetic_transcript, "time_window", window_sec=window_sec, overlap_sec=overlap_sec
    )

    max_allowed = window_sec + one_segment_duration + 0.05
    for c in chunks:
        duration = c["end"] - c["start"]
        assert duration <= max_allowed, (
            f"Chunk duration {duration} exceeds window_sec ({window_sec}) + one segment ({one_segment_duration})"
        )


def test_unknown_strategy_raises_value_error(synthetic_transcript: list[dict[str, any]]) -> None:
    """Assert that an unknown strategy name raises ValueError."""
    with pytest.raises(ValueError) as excinfo:
        chunk_transcript(synthetic_transcript, "non_existent_strategy")
    assert "Unknown strategy" in str(excinfo.value)


def test_no_punctuation_fallback_for_sentence(
    no_punct_transcript: list[dict[str, any]]
) -> None:
    """Test that the sentence strategy falls back gracefully when transcripts lack punctuation."""
    # Ensure split_sentences triggered fallback
    sents = split_sentences(no_punct_transcript)
    assert len(sents) > 0

    chunks = chunk_transcript(no_punct_transcript, "sentence", sentences_per_chunk=5, overlap_sentences=1)

    assert len(chunks) > 0
    for c in chunks:
        assert c["start"] < c["end"]

    # Verify coverage across all unpunctuated segments
    first_seg_start = no_punct_transcript[0]["start"]
    last_seg_end = round(
        no_punct_transcript[-1]["start"] + no_punct_transcript[-1]["duration"], 2
    )
    assert chunks[0]["start"] <= first_seg_start + 1e-2
    assert chunks[-1]["end"] >= last_seg_end - 1e-2

    for seg in no_punct_transcript:
        s_start = seg["start"]
        s_end = round(seg["start"] + seg["duration"], 2)
        covered = any(
            c["start"] <= s_start + 1e-2 and c["end"] >= s_end - 1e-2 for c in chunks
        )
        assert covered, f"Unpunctuated segment [{s_start}, {s_end}] not covered by sentence fallback."
