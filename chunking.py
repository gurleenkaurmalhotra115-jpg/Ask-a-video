"""Pluggable chunking strategies for video transcripts."""

from __future__ import annotations

import re
from typing import Any, Callable

import numpy as np


def words_with_times(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split each segment into words, assigning each word its parent segment's start and end times.

    Args:
        segments: List of transcript segments {"start", "duration", "text"}.

    Returns:
        List of dicts: {"word": str, "start": float, "end": float, "seg_idx": int}
    """
    words = []
    for idx, seg in enumerate(segments):
        seg_start = float(seg["start"])
        seg_end = float(seg["start"] + seg["duration"])
        text = seg.get("text", "")
        for w in text.split():
            words.append({
                "word": w,
                "start": seg_start,
                "end": seg_end,
                "seg_idx": idx,
            })
    return words


def split_sentences(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split transcript segments into sentences with character offset mapping to timestamps.

    If the transcript lacks punctuation (< 1 sentence boundary per 40 words,
    common in auto-generated captions), falls back to grouping ~15 words per sentence.

    Args:
        segments: List of transcript segments.

    Returns:
        List of dicts: {"text": str, "start": float, "end": float, "n_words": int, "seg_start_idx": int, "seg_end_idx": int}
    """
    if not segments:
        return []

    words = words_with_times(segments)
    if not words:
        return []

    total_words = len(words)
    full_text = " ".join(seg.get("text", "") for seg in segments)
    punct_count = len(re.findall(r"[.!?](?:\s+|$)", full_text))

    # Punctuation fallback: fewer than 1 sentence boundary per 40 words
    if total_words > 0 and (punct_count / total_words) < (1.0 / 40.0):
        fallback_sents = []
        words_per_sent = 15
        for i in range(0, total_words, words_per_sent):
            group = words[i : i + words_per_sent]
            s_text = " ".join(w["word"] for w in group)
            fallback_sents.append({
                "text": s_text,
                "start": group[0]["start"],
                "end": group[-1]["end"],
                "n_words": len(group),
                "seg_start_idx": group[0]["seg_idx"],
                "seg_end_idx": group[-1]["seg_idx"],
            })
        return fallback_sents

    # Standard sentence splitting using character offsets mapped to segments
    joined_parts = []
    char_to_seg = []
    for idx, seg in enumerate(segments):
        raw_text = seg.get("text", "").strip()
        if not raw_text:
            continue
        if joined_parts:
            joined_parts.append(" ")
            char_to_seg.append(idx)
        joined_parts.append(raw_text)
        char_to_seg.extend([idx] * len(raw_text))

    if not joined_parts:
        return []

    joined_text = "".join(joined_parts)
    sentences = []

    for match in re.finditer(r"[^.!?]+(?:[.!?]+|$)", joined_text):
        m_str = match.group()
        s_text = m_str.strip()
        if not s_text:
            continue
        start_pos = match.start() + (len(m_str) - len(m_str.lstrip()))
        end_pos = match.end() - (len(m_str) - len(m_str.rstrip()))

        first_seg_idx = char_to_seg[start_pos]
        last_seg_idx = char_to_seg[min(end_pos - 1, len(char_to_seg) - 1)]

        start_time = float(segments[first_seg_idx]["start"])
        end_time = float(segments[last_seg_idx]["start"] + segments[last_seg_idx]["duration"])

        sentences.append({
            "text": s_text,
            "start": round(start_time, 2),
            "end": round(end_time, 2),
            "n_words": len(s_text.split()),
            "seg_start_idx": first_seg_idx,
            "seg_end_idx": last_seg_idx,
        })

    if not sentences:
        s_text = " ".join(w["word"] for w in words)
        return [{
            "text": s_text,
            "start": words[0]["start"],
            "end": words[-1]["end"],
            "n_words": len(words),
            "seg_start_idx": words[0]["seg_idx"],
            "seg_end_idx": words[-1]["seg_idx"],
        }]

    return sentences


def fixed_token(
    segments: list[dict[str, Any]],
    chunk_words: int = 120,
    overlap_words: int = 20,
) -> list[dict[str, Any]]:
    """Chunk transcript using fixed-size word windows with overlap.

    Word count is used as the token proxy for fast, deterministic chunking without
    requiring a dedicated subword tokenizer.

    Args:
        segments: Transcript segments.
        chunk_words: Number of words per chunk window.
        overlap_words: Number of words shared between adjacent chunks.

    Returns:
        List of chunk dicts: {"chunk_id", "strategy", "text", "start", "end", "n_words"}.
    """
    if not segments:
        return []

    words = words_with_times(segments)
    if not words:
        return []

    if overlap_words >= chunk_words:
        overlap_words = max(0, chunk_words - 1)

    step = max(1, chunk_words - overlap_words)
    chunks = []
    total_words = len(words)

    i = 0
    chunk_id = 0
    while i < total_words:
        chunk_w = words[i : i + chunk_words]
        if not chunk_w:
            break

        start = chunk_w[0]["start"]
        end = chunk_w[-1]["end"]
        text = " ".join(w["word"] for w in chunk_w)

        chunks.append({
            "chunk_id": chunk_id,
            "strategy": "fixed_token",
            "text": text,
            "start": round(start, 2),
            "end": round(end, 2),
            "n_words": len(chunk_w),
        })
        chunk_id += 1

        if i + chunk_words >= total_words:
            break
        i += step

    return chunks


def sentence(
    segments: list[dict[str, Any]],
    sentences_per_chunk: int = 5,
    overlap_sentences: int = 1,
) -> list[dict[str, Any]]:
    """Chunk transcript by grouping complete sentences with overlap.

    Args:
        segments: Transcript segments.
        sentences_per_chunk: Number of complete sentences in each chunk.
        overlap_sentences: Number of sentences overlapping between adjacent chunks.

    Returns:
        List of chunk dicts: {"chunk_id", "strategy", "text", "start", "end", "n_words"}.
    """
    if not segments:
        return []

    sents = split_sentences(segments)
    if not sents:
        return []

    if overlap_sentences >= sentences_per_chunk:
        overlap_sentences = max(0, sentences_per_chunk - 1)

    step = max(1, sentences_per_chunk - overlap_sentences)
    chunks = []
    total_sents = len(sents)

    i = 0
    chunk_id = 0
    while i < total_sents:
        group = sents[i : i + sentences_per_chunk]
        if not group:
            break

        start = group[0]["start"]
        end = group[-1]["end"]
        text = " ".join(s["text"] for s in group)
        n_words = sum(s["n_words"] for s in group)

        chunks.append({
            "chunk_id": chunk_id,
            "strategy": "sentence",
            "text": text,
            "start": round(start, 2),
            "end": round(end, 2),
            "n_words": n_words,
        })
        chunk_id += 1

        if i + sentences_per_chunk >= total_sents:
            break
        i += step

    return chunks


def semantic(
    segments: list[dict[str, Any]],
    percentile: int = 25,
    min_words: int = 40,
    max_words: int = 200,
) -> list[dict[str, Any]]:
    """Chunk transcript semantically based on cosine similarity dips between sentences.

    Embeds sentences using SentenceTransformer, computes pairwise cosine similarities
    smoothed over a window of 2, and places split boundaries where similarity drops
    below the specified percentile, bounded by min_words and max_words.

    Args:
        segments: Transcript segments.
        percentile: Percentile threshold (0-100) below which a boundary is triggered.
        min_words: Minimum words before a chunk can split on a similarity dip.
        max_words: Maximum words after which a chunk must split regardless of similarity.

    Returns:
        List of chunk dicts: {"chunk_id", "strategy", "text", "start", "end", "n_words"}.
    """
    from embeddings import embed_texts

    if not segments:
        return []

    sents = split_sentences(segments)
    if not sents:
        return []

    if len(sents) == 1:
        s = sents[0]
        return [{
            "chunk_id": 0,
            "strategy": "semantic",
            "text": s["text"],
            "start": round(s["start"], 2),
            "end": round(s["end"], 2),
            "n_words": s["n_words"],
        }]

    texts = [s["text"] for s in sents]
    embeddings = embed_texts(texts)

    raw_sims = [
        float(np.dot(embeddings[i], embeddings[i + 1]))
        for i in range(len(sents) - 1)
    ]

    smoothed_sims = []
    for i in range(len(raw_sims)):
        if i == 0:
            smoothed_sims.append(raw_sims[0])
        else:
            smoothed_sims.append((raw_sims[i - 1] + raw_sims[i]) / 2.0)

    threshold = float(np.percentile(smoothed_sims, percentile))

    chunks = []
    chunk_id = 0
    curr_group = [sents[0]]
    curr_words = sents[0]["n_words"]

    for i in range(len(sents) - 1):
        next_sent = sents[i + 1]
        sim = smoothed_sims[i]

        split_needed = False
        if curr_words + next_sent["n_words"] > max_words:
            split_needed = True
        elif curr_words >= min_words and sim <= threshold:
            split_needed = True

        if split_needed:
            c_text = " ".join(s["text"] for s in curr_group)
            chunks.append({
                "chunk_id": chunk_id,
                "strategy": "semantic",
                "text": c_text,
                "start": round(curr_group[0]["start"], 2),
                "end": round(curr_group[-1]["end"], 2),
                "n_words": curr_words,
            })
            chunk_id += 1
            curr_group = [next_sent]
            curr_words = next_sent["n_words"]
        else:
            curr_group.append(next_sent)
            curr_words += next_sent["n_words"]

    if curr_group:
        c_text = " ".join(s["text"] for s in curr_group)
        chunks.append({
            "chunk_id": chunk_id,
            "strategy": "semantic",
            "text": c_text,
            "start": round(curr_group[0]["start"], 2),
            "end": round(curr_group[-1]["end"], 2),
            "n_words": curr_words,
        })

    return chunks


def time_window(
    segments: list[dict[str, Any]],
    window_sec: float = 40.0,
    overlap_sec: float = 8.0,
) -> list[dict[str, Any]]:
    """Chunk transcript by pure time windows with overlap, ignoring text boundaries.

    Args:
        segments: Transcript segments.
        window_sec: Target duration in seconds of each time window.
        overlap_sec: Target overlap in seconds between consecutive windows.

    Returns:
        List of chunk dicts: {"chunk_id", "strategy", "text", "start", "end", "n_words"}.
    """
    if not segments:
        return []

    if overlap_sec >= window_sec:
        overlap_sec = max(0.0, window_sec - 1.0)

    n_segs = len(segments)
    chunks = []
    chunk_id = 0
    i = 0

    while i < n_segs:
        chunk_segs = [segments[i]]
        first_seg_start = float(segments[i]["start"])
        target_end = first_seg_start + window_sec

        j = i
        while j < n_segs - 1:
            current_end = float(segments[j]["start"] + segments[j]["duration"])
            if current_end >= target_end:
                break
            j += 1
            chunk_segs.append(segments[j])

        start_time = float(chunk_segs[0]["start"])
        last_seg = chunk_segs[-1]
        end_time = float(last_seg["start"] + last_seg["duration"])
        text = " ".join(seg.get("text", "").strip() for seg in chunk_segs)
        text = " ".join(text.split())

        chunks.append({
            "chunk_id": chunk_id,
            "strategy": "time_window",
            "text": text,
            "start": round(start_time, 2),
            "end": round(end_time, 2),
            "n_words": len(text.split()),
        })
        chunk_id += 1

        if j >= n_segs - 1:
            break

        overlap_target = end_time - overlap_sec
        next_i = i + 1
        for k in range(i + 1, j + 1):
            seg_k_end = float(segments[k]["start"] + segments[k]["duration"])
            if seg_k_end > overlap_target:
                next_i = k
                break
        else:
            next_i = j

        i = max(i + 1, next_i)

    return chunks


STRATEGIES: dict[str, Callable[..., list[dict[str, Any]]]] = {
    "fixed_token": fixed_token,
    "sentence": sentence,
    "semantic": semantic,
    "time_window": time_window,
}

DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "fixed_token": {"chunk_words": 120, "overlap_words": 20},
    "sentence": {"sentences_per_chunk": 5, "overlap_sentences": 1},
    "semantic": {"percentile": 25, "min_words": 40, "max_words": 200},
    "time_window": {"window_sec": 40, "overlap_sec": 8},
}


def chunk_transcript(
    segments: list[dict[str, Any]],
    strategy: str,
    **params: Any,
) -> list[dict[str, Any]]:
    """Partition transcript segments into chunks using a designated strategy.

    Args:
        segments: List of transcript segment dictionaries {"start", "duration", "text"}.
        strategy: Strategy name: 'fixed_token', 'sentence', 'semantic', or 'time_window'.
        **params: Hyperparameters forwarded to the specific strategy function.

    Returns:
        List of chunk dicts: {"chunk_id", "strategy", "text", "start", "end", "n_words"}.

    Raises:
        ValueError: If strategy is not recognized.
    """
    if strategy not in STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Supported strategies: {list(STRATEGIES.keys())}"
        )

    strategy_fn = STRATEGIES[strategy]
    return strategy_fn(segments, **params)


def chunk_stats(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute summary statistics for a list of chunks.

    Args:
        chunks: List of chunk dictionaries.

    Returns:
        dict: {"n_chunks": int, "avg_words": float, "avg_duration_sec": float, "min_duration": float, "max_duration": float}
    """
    if not chunks:
        return {
            "n_chunks": 0,
            "avg_words": 0.0,
            "avg_duration_sec": 0.0,
            "min_duration": 0.0,
            "max_duration": 0.0,
        }

    durations = [round(c["end"] - c["start"], 2) for c in chunks]
    words = [c["n_words"] for c in chunks]
    return {
        "n_chunks": len(chunks),
        "avg_words": round(float(np.mean(words)), 1),
        "avg_duration_sec": round(float(np.mean(durations)), 1),
        "min_duration": round(float(min(durations)), 2),
        "max_duration": round(float(max(durations)), 2),
    }
