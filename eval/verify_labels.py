"""CLI tool to verify ground-truth answer timestamp spans against video transcripts."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

from eval.common import get_transcript_cached, load_questions
from transcript import fmt_time


def get_span_text(segments: list[dict[str, Any]], span_start: float, span_end: float) -> str:
    """Extract and concatenate text of all transcript segments overlapping [span_start, span_end]."""
    overlapping = []
    for seg in segments:
        s = float(seg["start"])
        d = float(seg.get("duration", 0.0))
        e = s + d
        if s < span_end and e > span_start:
            overlapping.append(seg.get("text", "").strip())
    return " ".join(t for t in overlapping if t)


def verify_labels(questions_path: str = "eval/questions.csv", out_path: str = "eval/label_check.md") -> int:
    """Verify labels in questions CSV, printing to stdout and writing to markdown report.

    Returns:
        int: Total number of warnings encountered.
    """
    df = load_questions(questions_path)
    answerable_df = df[df["answerable"] == True].copy()  # noqa: E712

    lines_out: list[str] = [
        "# Label Verification Report",
        "",
        f"**Source File:** `{questions_path}`",
        f"**Total Answerable Questions:** {len(answerable_df)}",
        "",
        "---",
        "",
    ]

    total_warnings = 0
    total_rows = len(answerable_df)

    print(f"\nVerifying labels for {total_rows} answerable questions from {questions_path}...\n" + "=" * 70)

    for idx, row in answerable_df.iterrows():
        row_num = idx + 1
        v_id = row["video_id"]
        q_text = row["question"]
        q_type = row.get("type", "unknown")
        start_sec = float(row["answer_start_sec"])
        end_sec = float(row["answer_end_sec"])
        duration = end_sec - start_sec

        try:
            segments = get_transcript_cached(v_id)
        except Exception as exc:
            err_msg = f"Failed to load transcript for video {v_id}: {exc}"
            print(f"Row #{row_num}: ERROR - {err_msg}")
            lines_out.append(f"### Row #{row_num}: {q_text}\n- **ERROR:** {err_msg}\n")
            total_warnings += 1
            continue

        span_text = get_span_text(segments, start_sec, end_sec)

        # Check warnings on main span
        row_warnings = []
        if duration > 90.0:
            row_warnings.append(f"Duration {duration:.1f}s > 90s (too wide: narrow to the core answer)")
        elif duration < 15.0:
            row_warnings.append(f"Duration {duration:.1f}s < 15s (too short)")

        # Alt span check if present
        alt_text = None
        alt_warnings = []
        alt_start = row.get("alt_start_sec")
        alt_end = row.get("alt_end_sec")
        has_alt = alt_start is not None and alt_end is not None and not (alt_start != alt_start)
        if has_alt:
            a_start = float(alt_start)
            a_end = float(alt_end)
            a_dur = a_end - a_start
            alt_text = get_span_text(segments, a_start, a_end)
            if a_dur > 90.0:
                alt_warnings.append(f"Alt duration {a_dur:.1f}s > 90s (too wide: narrow to the core answer)")
            elif a_dur < 15.0:
                alt_warnings.append(f"Alt duration {a_dur:.1f}s < 15s (too short)")

        total_warnings += len(row_warnings) + len(alt_warnings)

        # Console print
        print(f"Row #{row_num} [{q_type}] ({v_id}): {q_text}")
        print(f"  Main Span: {fmt_time(start_sec)} - {fmt_time(end_sec)} ({duration:.1f}s)")
        print(f"  Transcript: \"{span_text[:120]}{'...' if len(span_text) > 120 else ''}\"")
        for w in row_warnings:
            print(f"  ⚠️  WARNING: {w}")

        if has_alt:
            print(f"  Alt Span:  {fmt_time(alt_start)} - {fmt_time(alt_end)} ({(alt_end - alt_start):.1f}s)")
            print(f"  Alt Text:   \"{alt_text[:120] if alt_text else ''}{'...' if alt_text and len(alt_text) > 120 else ''}\"")
            for w in alt_warnings:
                print(f"  ⚠️  WARNING (Alt): {w}")
        print("-" * 70)

        # Markdown report entry
        lines_out.append(f"### Row #{row_num}: {q_text}")
        lines_out.append(f"- **Video ID:** `{v_id}` | **Type:** `{q_type}`")
        lines_out.append(f"- **Main Span:** `{fmt_time(start_sec)}` - `{fmt_time(end_sec)}` ({duration:.1f}s)")
        lines_out.append(f"- **Transcript Text:**\n  > {span_text}")
        for w in row_warnings:
            lines_out.append(f"- ⚠️ **WARNING:** {w}")

        if has_alt:
            lines_out.append(f"- **Alt Span:** `{fmt_time(alt_start)}` - `{fmt_time(alt_end)}` ({(alt_end - alt_start):.1f}s)")
            lines_out.append(f"- **Alt Transcript Text:**\n  > {alt_text}")
            for w in alt_warnings:
                lines_out.append(f"- ⚠️ **WARNING (Alt):** {w}")
        lines_out.append("")

    summary_msg = f"Summary: {total_rows} answerable rows evaluated, {total_warnings} warnings."
    print("\n" + "=" * 70)
    print(summary_msg)

    lines_out.append("---")
    lines_out.append(f"## Summary\n\n- **Total Rows:** {total_rows}\n- **Total Warnings:** {total_warnings}\n")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_out))

    print(f"Report written to {out_path}\n")
    return total_warnings


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify ground-truth question timestamp spans against transcripts.")
    parser.add_argument("--questions", default="eval/questions.csv", help="Path to questions CSV file.")
    parser.add_argument("--out", default="eval/label_check.md", help="Output markdown report path.")
    args = parser.parse_args()

    try:
        verify_labels(args.questions, args.out)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
