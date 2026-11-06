"""Common evaluation helpers for dataset loading, validation, and caching."""

from __future__ import annotations

import io
import json
import logging
import os
import warnings
from typing import Any

import pandas as pd

from transcript import get_transcript

logger = logging.getLogger(__name__)

TRANSCRIPT_CACHE_DIR = os.path.join("data", "transcripts")
MAX_GOLD_SPAN_DURATION_SEC = 90.0
REQUIRED_CORE_COLUMNS = {"video_id", "question"}


def _normalize_curly_quotes(text: str) -> str:
    """Replace typographical curly quotes with standard ASCII quotes."""
    return (
        text.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
    )


def _parse_bool(val: Any, row_idx: int, field_name: str = "answerable") -> bool:
    """Parse a value into a boolean with strict validation."""
    if isinstance(val, (bool,)):
        return bool(val)
    if pd.isna(val):
        raise ValueError(f"Row {row_idx}: '{field_name}' cannot be empty/NaN.")
    val_str = str(val).strip().lower()
    if val_str in {"true", "1", "yes", "t"}:
        return True
    if val_str in {"false", "0", "no", "f"}:
        return False
    raise ValueError(f"Row {row_idx}: Invalid boolean value for '{field_name}': '{val}'")


def load_questions(path: str) -> pd.DataFrame:
    """Load and validate an evaluation questions CSV file.

    Supports both:
    1. Schema A: video_id, tier, question, gold_start_sec, gold_end_sec, gold_answer_key, is_negative
    2. Schema B: video_id, question, answer_start_sec, answer_end_sec, type, answerable, alt_start_sec, alt_end_sec

    Enforces:
    - Non-empty video_id and question text
    - Numeric timestamps with end > start
    - Strict guardrail: (end - start) <= 90.0 seconds to prevent length-confounding IoU bias
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Questions CSV file not found: {path}")

    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        content = f.read()

    # Tolerate curly quotes
    content = _normalize_curly_quotes(content)

    # Check for literal 'csv' or markdown codeblock marker on the first line
    lines = content.splitlines()
    if lines:
        first_line_clean = lines[0].strip().lower().replace("`", "")
        if first_line_clean in {"csv", "text"}:
            warnings.warn(
                f"Skipping leading literal '{lines[0].strip()}' line in questions CSV.",
                UserWarning,
            )
            content = "\n".join(lines[1:])

    df = pd.read_csv(io.StringIO(content))

    missing_core = REQUIRED_CORE_COLUMNS - set(df.columns)
    if missing_core:
        raise ValueError(f"Questions CSV is missing core columns: {sorted(missing_core)}")

    cleaned_rows = []
    has_alt_cols = {"alt_start_sec", "alt_end_sec"}.issubset(df.columns)

    for i, row in df.iterrows():
        row_num = i + 1  # 1-indexed for clear error reporting

        v_id = str(row["video_id"]).strip()
        q_text = str(row["question"]).strip()
        q_type = str(row.get("type", "factual")).strip()
        tier = str(row.get("tier", "unknown")).strip()
        answer_key = str(row.get("gold_answer_key", "")).strip()

        # Determine answerability from is_negative or answerable
        if "is_negative" in df.columns:
            is_neg = _parse_bool(row["is_negative"], row_num, field_name="is_negative")
            ans_bool = not is_neg
        elif "answerable" in df.columns:
            ans_bool = _parse_bool(row["answerable"], row_num, field_name="answerable")
            is_neg = not ans_bool
        else:
            raise ValueError(f"Row {row_num}: Must contain either 'is_negative' or 'answerable' column.")

        # Determine start/end timestamps from gold_* or answer_*
        raw_start = row.get("gold_start_sec") if "gold_start_sec" in df.columns else row.get("answer_start_sec")
        raw_end = row.get("gold_end_sec") if "gold_end_sec" in df.columns else row.get("answer_end_sec")

        row_dict = {
            "video_id": v_id,
            "tier": tier,
            "question": q_text,
            "type": q_type,
            "answerable": ans_bool,
            "is_negative": is_neg,
            "gold_answer_key": answer_key,
            "answer_start_sec": None,
            "answer_end_sec": None,
            "gold_start_sec": None,
            "gold_end_sec": None,
            "alt_start_sec": None,
            "alt_end_sec": None,
        }

        if ans_bool:
            if pd.isna(raw_start) or pd.isna(raw_end) or str(raw_start).strip() == "" or str(raw_end).strip() == "":
                raise ValueError(f"Row {row_num}: Answerable question is missing start/end timestamp.")

            try:
                start_sec = float(raw_start)
                end_sec = float(raw_end)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Row {row_num}: Timestamps must be numeric: {exc}") from exc

            if end_sec <= start_sec:
                raise ValueError(
                    f"Row {row_num}: end_sec ({end_sec}) must be greater than start_sec ({start_sec})."
                )

            span_duration = end_sec - start_sec
            if span_duration > MAX_GOLD_SPAN_DURATION_SEC:
                raise ValueError(
                    f"Row {row_num}: Gold span duration ({span_duration:.1f}s) exceeds the strict "
                    f"{MAX_GOLD_SPAN_DURATION_SEC}s maximum guardrail ([{start_sec}, {end_sec}]). "
                    "Tighten the span to isolate chunking logic from length-confounded IoU bias."
                )

            row_dict["answer_start_sec"] = start_sec
            row_dict["answer_end_sec"] = end_sec
            row_dict["gold_start_sec"] = start_sec
            row_dict["gold_end_sec"] = end_sec

            # Check optional alt span
            if has_alt_cols:
                raw_alt_start = row.get("alt_start_sec")
                raw_alt_end = row.get("alt_end_sec")
                if pd.notna(raw_alt_start) and pd.notna(raw_alt_end) and str(raw_alt_start).strip() != "" and str(raw_alt_end).strip() != "":
                    try:
                        alt_start = float(raw_alt_start)
                        alt_end = float(raw_alt_end)
                    except (ValueError, TypeError) as exc:
                        raise ValueError(f"Row {row_num}: Alt timestamps must be numeric: {exc}") from exc

                    if alt_end <= alt_start:
                        raise ValueError(
                            f"Row {row_num}: alt_end_sec ({alt_end}) must be greater than alt_start_sec ({alt_start})."
                        )

                    alt_span_dur = alt_end - alt_start
                    if alt_span_dur > MAX_GOLD_SPAN_DURATION_SEC:
                        raise ValueError(
                            f"Row {row_num}: Alt span duration ({alt_span_dur:.1f}s) exceeds "
                            f"{MAX_GOLD_SPAN_DURATION_SEC}s maximum guardrail."
                        )

                    row_dict["alt_start_sec"] = alt_start
                    row_dict["alt_end_sec"] = alt_end

        cleaned_rows.append(row_dict)

    return pd.DataFrame(cleaned_rows)

    return pd.DataFrame(cleaned_rows)


def get_transcript_cached(video_id: str, transcript_source: str = "captions") -> list[dict[str, Any]]:
    """Retrieve video transcript for specified source, caching to data/transcripts/<source>/<video_id>.json.

    Args:
        video_id: 11-character YouTube video ID.
        transcript_source: 'captions' or 'whisper'.

    Returns:
        list[dict]: List of transcript segment dicts {"start", "duration", "text"}.
    """
    sub_dir = os.path.join(TRANSCRIPT_CACHE_DIR, transcript_source)
    os.makedirs(sub_dir, exist_ok=True)
    cache_path = os.path.join(sub_dir, f"{video_id}.json")
    legacy_path = os.path.join(TRANSCRIPT_CACHE_DIR, f"{video_id}.json")

    # If asking for captions and legacy file exists, support it
    target_read_path = cache_path
    if transcript_source == "captions" and not os.path.exists(cache_path) and os.path.exists(legacy_path):
        target_read_path = legacy_path

    if os.path.exists(target_read_path) and os.path.getsize(target_read_path) > 0:
        try:
            with open(target_read_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    return data
        except Exception as exc:
            logger.warning("Failed to load cached transcript from %s: %s", target_read_path, exc)

    if transcript_source == "whisper":
        from transcript import transcribe_with_whisper

        segments = transcribe_with_whisper(video_id)
    else:
        from transcript import fetch_captions

        segments = fetch_captions(video_id)

    # Cache to disk
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.warning("Failed to write transcript cache to %s: %s", cache_path, exc)

    return segments
