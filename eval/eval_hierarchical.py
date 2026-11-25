"""Two-Stage Hierarchical Video Retriever (Algorithm 1) Evaluation Benchmark.

Evaluates the 'Search Space Expansion Effect' mitigation on long educational
lectures (>35 min, N = 100 queries) against MedVidQA temporal ground-truth spans.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

# Ensure project root is in sys.path
_pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

import numpy as np
import pandas as pd
from tabulate import tabulate

from chunking import DEFAULT_PARAMS, chunk_transcript
from eval.common import get_transcript_cached, load_questions
from eval.metrics import compute_bootstrap_ci, compute_retrieval_metrics
from embeddings import embed_texts
from hierarchical_retriever import build_macro_blocks, search_hierarchical

LONG_VIDEOS = [
    "zjkBMFhNj_g",
    "8SF_h3xF3cE",
    "jGwO_UgTS7I",
    "II4giR4vOOo",
    "J7DzL2_Na80",
    "PaCmpygFfXo",
    "5p248yoa3oE",
    "YLczBemZpaE",
    "MJ1vWb1rGwM",
]


def run_hierarchical_eval(
    questions_path: str = "eval/questions.csv",
    t_macro: float = 300.0,
    top_m_macro: int = 2,
    k_micro: int = 4,
    out_dir: str = "results",
) -> dict[str, Any]:
    """Run comparative benchmark between Flat Single-Stage and Two-Stage Hierarchical retrieval."""
    os.makedirs(out_dir, exist_ok=True)
    df = load_questions(questions_path)

    # Filter to long video answerable questions
    df_long = df[df["video_id"].isin(LONG_VIDEOS)].copy()
    if "answerable" in df_long.columns:
        df_long = df_long[df_long["answerable"] == True].copy()
    elif "type" in df_long.columns:
        df_long = df_long[df_long["type"] == "answerable"].copy()

    total_queries = len(df_long)
    print("=" * 80, flush=True)
    print("VIDQUERY TWO-STAGE HIERARCHICAL VIDEO RETRIEVAL BENCHMARK", flush=True)
    print(f"Target Corpus: Long Lectures (>35 min, {len(LONG_VIDEOS)} videos, {total_queries} queries)", flush=True)
    print(f"Parameters:    T_macro = {t_macro}s (5m), Top-M Macro = {top_m_macro}, Top-k Micro = {k_micro}", flush=True)
    print("=" * 80, flush=True)

    # 1. Pre-chunk and pre-embed all long videos
    video_chunks: dict[str, list[dict[str, Any]]] = {}
    video_chunk_embs: dict[str, np.ndarray] = {}
    video_macro_blocks: dict[str, list[Any]] = {}
    video_macro_embs: dict[str, np.ndarray] = {}

    for vid in LONG_VIDEOS:
        print(f"Pre-indexing & embedding long video: {vid} ...", flush=True)
        segs = get_transcript_cached(vid, transcript_source="captions")
        chunks = chunk_transcript(segs, "fixed_token", **DEFAULT_PARAMS["fixed_token"])
        video_chunks[vid] = chunks

        # Compute micro embeddings
        texts = [c["text"] for c in chunks]
        embs = np.asarray(embed_texts(texts), dtype=np.float32)
        # L2-normalize
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms < 1e-9] = 1.0
        video_chunk_embs[vid] = embs / norms

        # Macro blocks
        m_blocks = build_macro_blocks(chunks, t_macro=t_macro)
        video_macro_blocks[vid] = m_blocks
        m_texts = [b.text for b in m_blocks]
        if m_texts:
            m_embs = np.asarray(embed_texts(m_texts), dtype=np.float32)
            m_norms = np.linalg.norm(m_embs, axis=1, keepdims=True)
            m_norms[m_norms < 1e-9] = 1.0
            video_macro_embs[vid] = m_embs / m_norms

    # 2. Embed queries
    queries = df_long["question"].tolist()
    q_embs_raw = np.asarray(embed_texts(queries), dtype=np.float32)
    q_norms = np.linalg.norm(q_embs_raw, axis=1, keepdims=True)
    q_norms[q_norms < 1e-9] = 1.0
    q_embs = q_embs_raw / q_norms

    results_flat: list[dict[str, Any]] = []
    results_hier: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []

    for idx, (_, row) in enumerate(df_long.iterrows()):
        vid = row["video_id"]
        q_vec = q_embs[idx]
        gold_spans = [(float(row["answer_start_sec"]), float(row["answer_end_sec"]))]
        if "alt_start_sec" in row and pd.notna(row["alt_start_sec"]) and pd.notna(row["alt_end_sec"]):
            gold_spans.append((float(row["alt_start_sec"]), float(row["alt_end_sec"])))

        chunks = video_chunks[vid]
        c_embs = video_chunk_embs[vid]

        # Flat Dense Retrieval
        flat_sims = np.dot(c_embs, q_vec)
        flat_top_indices = np.argsort(flat_sims)[::-1][:k_micro]
        flat_retrieved = []
        for i in flat_top_indices:
            c_copy = dict(chunks[i])
            c_copy["score"] = float(flat_sims[i])
            flat_retrieved.append(c_copy)

        m_flat = compute_retrieval_metrics(flat_retrieved, gold_spans)
        results_flat.append(m_flat)

        # Hierarchical Two-Stage Retrieval
        m_blocks = video_macro_blocks[vid]
        if len(m_blocks) <= top_m_macro:
            hier_retrieved = flat_retrieved
        else:
            m_embs = video_macro_embs[vid]
            macro_sims = np.dot(m_embs, q_vec)
            top_m_indices = set(np.argsort(macro_sims)[::-1][:top_m_macro])

            # Gather micro candidate indices from selected macro blocks
            cand_indices = []
            for m_idx in top_m_indices:
                cand_indices.extend(m_blocks[m_idx].chunk_indices)

            cand_indices = list(set(cand_indices))
            cand_sims = flat_sims[cand_indices]
            hier_top_sub = np.argsort(cand_sims)[::-1][:k_micro]
            hier_retrieved = []
            for sub_i in hier_top_sub:
                real_idx = cand_indices[sub_i]
                c_copy = dict(chunks[real_idx])
                c_copy["score"] = float(cand_sims[sub_i])
                hier_retrieved.append(c_copy)

        m_hier = compute_retrieval_metrics(hier_retrieved, gold_spans)
        results_hier.append(m_hier)

        comparison_rows.append({
            "video_id": vid,
            "question": row["question"],
            "flat_hit_at_1": m_flat["hit_at_1"],
            "flat_hit_at_k": m_flat["hit_at_k"],
            "flat_error_sec": m_flat["start_error_top1"],
            "hier_hit_at_1": m_hier["hit_at_1"],
            "hier_hit_at_k": m_hier["hit_at_k"],
            "hier_error_sec": m_hier["start_error_top1"],
        })

    # Summary Statistics
    flat_hit1 = float(np.mean([r["hit_at_1"] for r in results_flat]))
    hier_hit1 = float(np.mean([r["hit_at_1"] for r in results_hier]))
    flat_hitk = float(np.mean([r["hit_at_k"] for r in results_flat]))
    hier_hitk = float(np.mean([r["hit_at_k"] for r in results_hier]))

    flat_errors = [r["start_error_top1"] for r in results_flat if r["start_error_top1"] is not None]
    hier_errors = [r["start_error_top1"] for r in results_hier if r["start_error_top1"] is not None]

    flat_med_err = float(np.median(flat_errors)) if flat_errors else 0.0
    hier_med_err = float(np.median(hier_errors)) if hier_errors else 0.0

    _, ci_lower, ci_upper = compute_bootstrap_ci([r["hit_at_1"] for r in results_hier])
    error_reduction_pct = ((flat_med_err - hier_med_err) / max(1.0, flat_med_err)) * 100.0

    table_data = [
        ["Single-Stage Flat (fixed_token)", f"{flat_hit1*100:.1f}%", f"{flat_hitk*100:.1f}%", f"{flat_med_err:.1f}s", "Baseline (0.0%)"],
        ["Two-Stage Hierarchical (T=300s, M=2)", f"{hier_hit1*100:.1f}%", f"{hier_hitk*100:.1f}%", f"{hier_med_err:.1f}s", f"-{error_reduction_pct:.1f}% error"],
    ]

    print("\n" + tabulate(
        table_data,
        headers=["Retrieval Method", "Hit@1", f"Hit@{k_micro}", "Median Boundary Error", "Error Reduction"],
        tablefmt="github",
    ))
    print(f"\nBootstrap 95% Confidence Interval for Hierarchical Hit@1: [{ci_lower*100:.1f}%, {ci_upper*100:.1f}%]")

    # Save summary CSV
    summary_df = pd.DataFrame([
        {
            "method": "single_stage_flat",
            "hit_at_1": round(flat_hit1, 4),
            "hit_at_k": round(flat_hitk, 4),
            "median_boundary_error_sec": round(flat_med_err, 2),
            "ci_lower": None,
            "ci_upper": None,
        },
        {
            "method": "two_stage_hierarchical",
            "hit_at_1": round(hier_hit1, 4),
            "hit_at_k": round(hier_hitk, 4),
            "median_boundary_error_sec": round(hier_med_err, 2),
            "ci_lower": round(ci_lower, 4),
            "ci_upper": round(ci_upper, 4),
        },
    ])
    summary_path = os.path.join(out_dir, "hierarchical_eval_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary metrics saved to: {summary_path}")

    per_q_path = os.path.join(out_dir, "hierarchical_eval_per_question.csv")
    pd.DataFrame(comparison_rows).to_csv(per_q_path, index=False)
    print(f"Per-question results saved to: {per_q_path}")

    return {
        "flat_hit1": flat_hit1,
        "hier_hit1": hier_hit1,
        "flat_med_err": flat_med_err,
        "hier_med_err": hier_med_err,
        "hit1_ci": hit1_ci,
    }


if __name__ == "__main__":
    run_hierarchical_eval()
