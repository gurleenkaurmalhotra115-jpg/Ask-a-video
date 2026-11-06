"""CLI retrieval-only evaluation of chunking strategies against ground-truth spans."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import sys
from typing import Any

import numpy as np
import pandas as pd
from tabulate import tabulate

from chunking import DEFAULT_PARAMS, chunk_transcript
from eval.common import get_transcript_cached, load_questions
from eval.metrics import compute_bootstrap_ci, compute_retrieval_metrics
from vectorstore import get_client, index_chunks, list_indexed, search

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def ensure_video_indexed(
    client: Any,
    video_id: str,
    segments: list[dict[str, Any]],
    transcript_source: str = "captions",
    force_reindex: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Ensure all 4 chunking strategies are indexed in Qdrant for the given video and transcript source.

    Returns:
        dict: Mapping of strategy name -> list of chunk dicts.
    """
    indexed = list_indexed(client)
    indexed_map = {
        item["strategy"]: item.get("n_chunks", item.get("count", 0))
        for item in indexed
        if item["video_id"] == video_id and item.get("transcript_source", "captions") == transcript_source
    }

    chunks_by_strategy: dict[str, list[dict[str, Any]]] = {}

    for strat, params in DEFAULT_PARAMS.items():
        chunks = chunk_transcript(segments, strat, **params)
        chunks_by_strategy[strat] = chunks

        if force_reindex or indexed_map.get(strat, 0) == 0:
            print(f"Indexing {len(chunks)} chunks for video {video_id} [{transcript_source}] with strategy '{strat}'...")
            index_chunks(client, video_id, strat, chunks, transcript_source=transcript_source)

    return chunks_by_strategy


def run_evaluation(
    questions_path: str = "eval/questions.csv",
    k: int = 4,
    out_dir: str = "results",
    db_path: str = "qdrant_data",
    sources: list[str] | None = None,
    reindex: bool = False,
) -> None:
    """Run retrieval-only benchmark across all questions, strategies, and transcript sources."""
    if sources is None:
        sources = ["captions"]

    os.makedirs(out_dir, exist_ok=True)
    df = load_questions(questions_path)

    run_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    print("\n" + "=" * 80)
    print("VIDQUERY RETRIEVAL EVALUATION (RESEARCH BENCHMARK)")
    print(f"Date:               {run_time}")
    print(f"Embedding Model:    {EMBEDDING_MODEL_NAME}")
    print(f"Questions File:     {questions_path} ({len(df)} total questions)")
    print(f"Top-k Retrieval:    {k}")
    print(f"Transcript Sources: {', '.join(sources)}")
    print(f"Qdrant Storage:     {db_path}")
    print("Chunk Params:")
    for s_name, s_params in DEFAULT_PARAMS.items():
        print(f"  - {s_name}: {s_params}")
    print("=" * 80 + "\n")

    client = get_client(path=db_path)
    unique_videos = df["video_id"].unique()

    # Pre-index all videos per transcript source
    video_segments: dict[tuple[str, str], list[dict[str, Any]]] = {}
    video_chunks: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = {}

    for src in sources:
        for v_id in unique_videos:
            print(f"Fetching / loading [{src}] transcript for video: {v_id}")
            try:
                segs = get_transcript_cached(v_id, transcript_source=src)
                video_segments[(v_id, src)] = segs
                chunks_by_strat = ensure_video_indexed(client, v_id, segs, transcript_source=src, force_reindex=reindex)
                video_chunks[(v_id, src)] = chunks_by_strat
            except Exception as exc:
                print(f"Warning: Failed to fetch/index [{src}] transcript for {v_id}: {exc}")

    # Separate answerable and unanswerable questions
    answerable_df = df[df["answerable"] == True].copy()  # noqa: E712
    negative_df = df[df["answerable"] == False].copy()  # noqa: E712

    per_question_rows: list[dict[str, Any]] = []
    strategies = list(DEFAULT_PARAMS.keys())

    print(f"\nEvaluating {len(answerable_df)} answerable questions across {len(strategies)} strategies and {len(sources)} transcript sources...")

    for src in sources:
        for idx, row in answerable_df.iterrows():
            v_id = row["video_id"]
            if (v_id, src) not in video_segments:
                continue

            q_text = row["question"]
            q_type = row.get("type", "unknown")

            gold_spans = [(float(row["answer_start_sec"]), float(row["answer_end_sec"]))]
            alt_start = row.get("alt_start_sec")
            alt_end = row.get("alt_end_sec")
            if alt_start is not None and alt_end is not None and not pd.isna(alt_start) and not pd.isna(alt_end):
                gold_spans.append((float(alt_start), float(alt_end)))

            for strat in strategies:
                retrieved = search(
                    client=client,
                    query=q_text,
                    video_id=v_id,
                    strategy=strat,
                    transcript_source=src,
                    k=k,
                )

                metrics = compute_retrieval_metrics(retrieved, gold_spans)

                top1 = retrieved[0] if retrieved else {}
                chunk_ids = [c.get("chunk_id") for c in retrieved]

                per_question_rows.append({
                    "video_id": v_id,
                    "question": q_text,
                    "type": q_type,
                    "transcript_source": src,
                    "strategy": strat,
                    "hit_at_1": metrics["hit_at_1"],
                    "hit_at_k": metrics["hit_at_k"],
                    "reciprocal_rank": metrics["reciprocal_rank"],
                    "iou_top1": metrics["iou_top1"],
                    "start_error_top1": metrics["start_error_top1"],
                    "within10_top1": metrics["within10_top1"],
                    "r1_iou_03": metrics["r1_iou_03"],
                    "r1_iou_05": metrics["r1_iou_05"],
                    "r1_iou_07": metrics["r1_iou_07"],
                    "rk_iou_03": metrics["rk_iou_03"],
                    "rk_iou_05": metrics["rk_iou_05"],
                    "rk_iou_07": metrics["rk_iou_07"],
                    "top1_score": metrics["top1_score"],
                    "top1_chunk_id": top1.get("chunk_id"),
                    "top1_start": top1.get("start"),
                    "top1_end": top1.get("end"),
                    "retrieved_chunk_ids": json.dumps(chunk_ids),
                })

    pq_df = pd.DataFrame(per_question_rows)
    pq_path = os.path.join(out_dir, "retrieval_per_question.csv")
    pq_df.to_csv(pq_path, index=False)
    print(f"Detailed question results saved to: {pq_path}")

    # Evaluate unanswerable (negative) questions for top-1 score separation
    neg_scores: dict[tuple[str, str], list[float]] = {}
    pos_scores: dict[tuple[str, str], list[float]] = {}

    for src in sources:
        for strat in strategies:
            pair = (src, strat)
            pos_scores[pair] = pq_df[(pq_df["transcript_source"] == src) & (pq_df["strategy"] == strat)]["top1_score"].tolist()
            neg_scores[pair] = []

    if len(negative_df) > 0:
        for idx, row in negative_df.iterrows():
            v_id = row["video_id"]
            q_text = row["question"]
            for src in sources:
                if (v_id, src) not in video_segments:
                    continue
                for strat in strategies:
                    neg_res = search(
                        client=client,
                        query=q_text,
                        video_id=v_id,
                        strategy=strat,
                        transcript_source=src,
                        k=1,
                    )
                    score = float(neg_res[0].get("score", 0.0)) if neg_res else 0.0
                    neg_scores[(src, strat)].append(score)

    # Strategy Summary Metrics with Bootstrap 95% Confidence Intervals
    summary_rows: list[dict[str, Any]] = []
    display_rows: list[list[Any]] = []

    for src in sources:
        for strat in strategies:
            strat_df = pq_df[(pq_df["transcript_source"] == src) & (pq_df["strategy"] == strat)]
            n_q = len(strat_df)
            if n_q == 0:
                continue

            hit1_mean, hit1_lo, hit1_hi = compute_bootstrap_ci(strat_df["hit_at_1"].tolist())
            hitk_mean, hitk_lo, hitk_hi = compute_bootstrap_ci(strat_df["hit_at_k"].tolist())
            mrr_mean, mrr_lo, mrr_hi = compute_bootstrap_ci(strat_df["reciprocal_rank"].tolist())
            miou_mean, miou_lo, miou_hi = compute_bootstrap_ci(strat_df["iou_top1"].tolist())

            r1_03_mean, r1_03_lo, r1_03_hi = compute_bootstrap_ci(strat_df["r1_iou_03"].tolist())
            r1_05_mean, r1_05_lo, r1_05_hi = compute_bootstrap_ci(strat_df["r1_iou_05"].tolist())
            r1_07_mean, r1_07_lo, r1_07_hi = compute_bootstrap_ci(strat_df["r1_iou_07"].tolist())

            within10 = float(strat_df["within10_top1"].mean() * 100.0)
            med_start_err = float(np.median(strat_df["start_error_top1"].abs()))

            # Chunk statistics
            all_durations = []
            total_chunks = 0
            for v_id in unique_videos:
                c_list = video_chunks.get((v_id, src), {}).get(strat, [])
                total_chunks += len(c_list)
                for c in c_list:
                    all_durations.append(c["end"] - c["start"])
            avg_dur = float(np.mean(all_durations)) if all_durations else 0.0

            summary_rows.append({
                "transcript_source": src,
                "strategy": strat,
                "n_questions": n_q,
                "Hit@1": round(hit1_mean, 4),
                "Hit@1_95CI": f"[{hit1_lo:.2f}, {hit1_hi:.2f}]",
                "Hit@k": round(hitk_mean, 4),
                "Hit@k_95CI": f"[{hitk_lo:.2f}, {hitk_hi:.2f}]",
                "MRR": round(mrr_mean, 4),
                "MRR_95CI": f"[{mrr_lo:.2f}, {mrr_hi:.2f}]",
                "mean_IoU@1": round(miou_mean, 4),
                "mean_IoU@1_95CI": f"[{miou_lo:.2f}, {miou_hi:.2f}]",
                "R@1_IoU_0.3": round(r1_03_mean, 4),
                "R@1_IoU_0.3_95CI": f"[{r1_03_lo:.2f}, {r1_03_hi:.2f}]",
                "R@1_IoU_0.5": round(r1_05_mean, 4),
                "R@1_IoU_0.5_95CI": f"[{r1_05_lo:.2f}, {r1_05_hi:.2f}]",
                "R@1_IoU_0.7": round(r1_07_mean, 4),
                "R@1_IoU_0.7_95CI": f"[{r1_07_lo:.2f}, {r1_07_hi:.2f}]",
                "Within10@1 (%)": round(within10, 2),
                "median_abs_start_error": round(med_start_err, 2),
                "avg_chunk_duration_s": round(avg_dur, 2),
                "n_chunks": total_chunks,
            })

            display_rows.append([
                src,
                strat,
                n_q,
                f"{hit1_mean * 100:.1f}% [{hit1_lo * 100:.0f}, {hit1_hi * 100:.0f}]",
                f"{hitk_mean * 100:.1f}%",
                f"{mrr_mean:.3f}",
                f"{miou_mean:.3f} [{miou_lo:.2f}, {miou_hi:.2f}]",
                f"{r1_03_mean * 100:.1f}%",
                f"{r1_05_mean * 100:.1f}%",
                f"{r1_07_mean * 100:.1f}%",
                f"{within10:.1f}%",
                f"{med_start_err:.1f}s",
                f"{avg_dur:.1f}s",
                total_chunks,
            ])

    sum_df = pd.DataFrame(summary_rows)
    sum_path = os.path.join(out_dir, "retrieval_summary.csv")
    sum_df.to_csv(sum_path, index=False)
    print(f"Summary metrics saved to: {sum_path}\n")

    # Print Main Research Summary Table
    headers = [
        "Source",
        "Strategy",
        "Q#",
        "Hit@1 [95% CI]",
        f"Hit@{k}",
        "MRR",
        "Mean IoU@1 [95% CI]",
        "R@1@0.3",
        "R@1@0.5",
        "R@1@0.7",
        "Within 10s",
        "Med |Start|",
        "Avg Dur",
        "Chunks",
    ]
    print("### RETRIEVAL PERFORMANCE (MEDVIDQA STANDARDS & BOOTSTRAP CIs)")
    print(tabulate(display_rows, headers=headers, tablefmt="github"))
    print()

    # Breakdown by Question Type
    type_display_rows: list[list[Any]] = []
    types_found = sorted(pq_df["type"].unique())

    for q_type in types_found:
        type_df = pq_df[pq_df["type"] == q_type]
        for src in sources:
            for strat in strategies:
                st_df = type_df[(type_df["transcript_source"] == src) & (type_df["strategy"] == strat)]
                if len(st_df) == 0:
                    continue
                h1 = st_df["hit_at_1"].mean()
                hk = st_df["hit_at_k"].mean()
                m = st_df["reciprocal_rank"].mean()
                iou_val = st_df["iou_top1"].mean()
                r1_03 = st_df["r1_iou_03"].mean()
                type_display_rows.append([
                    src,
                    strat,
                    q_type,
                    len(st_df),
                    f"{h1 * 100:.1f}%",
                    f"{hk * 100:.1f}%",
                    f"{m:.3f}",
                    f"{iou_val:.3f}",
                    f"{r1_03 * 100:.1f}%",
                ])

    if type_display_rows:
        type_headers = ["Source", "Strategy", "Question Type", "Count", "Hit@1", f"Hit@{k}", "MRR", "Mean IoU@1", "R@1@0.3"]
        print("### PERFORMANCE SPLIT BY QUESTION TYPE")
        print(tabulate(type_display_rows, headers=type_headers, tablefmt="github"))
        print()

    # Negative vs Positive Score Separation
    if len(negative_df) > 0:
        neg_table_rows = []
        for src in sources:
            for strat in strategies:
                pair = (src, strat)
                p_mean = float(np.mean(pos_scores[pair])) if pos_scores[pair] else 0.0
                n_mean = float(np.mean(neg_scores[pair])) if neg_scores[pair] else 0.0
                delta = p_mean - n_mean
                neg_table_rows.append([
                    src,
                    strat,
                    f"{p_mean:.4f}",
                    f"{n_mean:.4f}",
                    f"{delta:+.4f}",
                ])

        neg_headers = ["Source", "Strategy", "Mean Top-1 Pos", f"Mean Top-1 Neg (n={len(negative_df)})", "Separation Delta"]
        print("### NEGATIVE VS POSITIVE SCORE SEPARATION")
        print(tabulate(neg_table_rows, headers=neg_headers, tablefmt="github"))
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate video transcript chunking strategies on retrieval benchmarks.")
    parser.add_argument("--questions", default="eval/questions.csv", help="Path to questions CSV file.")
    parser.add_argument("--k", type=int, default=4, help="Top-k retrieval budget.")
    parser.add_argument("--out", default="results", help="Directory where evaluation results are saved.")
    parser.add_argument("--db", default="qdrant_data", help="Path to Qdrant storage folder.")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["captions"],
        choices=["captions", "whisper"],
        help="Transcript sources to evaluate ('captions', 'whisper', or both).",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force re-indexing chunks into Qdrant before evaluation.",
    )
    args = parser.parse_args()

    try:
        run_evaluation(
            questions_path=args.questions,
            k=args.k,
            out_dir=args.out,
            db_path=args.db,
            sources=args.sources,
            reindex=args.reindex,
        )
    except Exception as exc:
        print(f"Evaluation error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
