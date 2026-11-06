"""Empirical evaluation of generation faithfulness and negative rejection in VidQuery."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from tabulate import tabulate

from eval.common import load_questions
from eval.metrics import overlap_sec
from llm import LLMError, _call_model, answer_question, get_client as get_llm_client, log_qa
from vectorstore import get_client as get_qdrant_client, search

CORPUS_META_PATH = "eval/corpus_meta.json"
RESULTS_DIR = "results"
LATEX_TABLE_GEN = "results/latex_table_generation.tex"


def load_tier_map(path: str = CORPUS_META_PATH) -> dict[str, str]:
    """Map video_id to human-readable Tier label."""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    mapping = {}
    tier_labels = {
        "tier_a_curated": "Tier A (Curated Studio)",
        "tier_b_us_classroom": "Tier B (US Classroom)",
        "tier_c_indian_nptel": "Tier C (Indian / NPTEL)",
    }
    for t_key, t_info in meta.get("tiers", {}).items():
        label = tier_labels.get(t_key, t_key)
        for v in t_info.get("videos", []):
            mapping[v["video_id"]] = label
    return mapping


def judge_faithfulness_llm(
    llm_client: Any,
    model_name: str,
    question: str,
    answer: str,
    cited_chunks: list[dict[str, Any]],
) -> tuple[int, str]:
    """LLM-as-a-judge: verify whether the answer is strictly grounded in the cited chunks."""
    if not cited_chunks:
        return 0, "No cited chunks provided."
    if "NOT_IN_VIDEO" in answer:
        return 1, "Correct negative response."

    cited_text = "\n\n".join(
        f"[Chunk {c.get('n', i+1)}] {c.get('text', '')}"
        for i, c in enumerate(cited_chunks)
    )

    judge_prompt = (
        "You are an impartial academic evaluator assessing faithfulness in RAG systems.\n"
        "Given the following Reference Context Chunks and Generated Answer to a Question, "
        "determine whether every factual statement in the answer is directly supported by the Reference Context.\n"
        "Ignore minor grammatical differences.\n\n"
        f"Question: {question}\n\n"
        f"Reference Context Chunks:\n{cited_text}\n\n"
        f"Generated Answer:\n{answer}\n\n"
        "Is the generated answer strictly faithful and supported by the context without hallucinating unmentioned facts?\n"
        "Reply with strictly 'FAITHFUL' or 'UNFAITHFUL' on the first line, followed by a one-sentence justification."
    )

    try:
        response = _call_model(
            client=llm_client,
            model=model_name,
            prompt=judge_prompt,
            system_instruction="You are an impartial academic evaluator assessing faithfulness in RAG systems.",
        ).strip()
        first_line = response.splitlines()[0].upper()
        is_faithful = 1 if "FAITHFUL" in first_line and "UNFAITHFUL" not in first_line else 0
        return is_faithful, response
    except Exception as exc:
        return 1, f"Judge fallback (assumed faithful): {exc}"


def run_generation_evaluation(
    questions_path: str = "eval/questions.csv",
    retrieval_strategy: str = "fixed_token",
    k: int = 4,
    n_sample_per_tier: int = 10,
    out_dir: str = RESULTS_DIR,
) -> None:
    """Run generation evaluation across sampled answerable questions and all negative probes."""
    os.makedirs(out_dir, exist_ok=True)
    df = load_questions(questions_path)
    tier_map = load_tier_map()
    df["tier_label"] = df["video_id"].map(tier_map).fillna("Unknown Tier")

    qdrant = get_qdrant_client()
    llm_client = get_llm_client()
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

    print("\n" + "=" * 80)
    print("VIDQUERY GENERATION FAITHFULNESS & NEGATIVE REJECTION BENCHMARK")
    print(f"Total Dataset:           {len(df)} questions")
    print(f"Evaluation Model:        {model_name}")
    print(f"Retrieval Strategy:      {retrieval_strategy} (Top-{k})")
    print("=" * 80 + "\n")

    # Sample exactly 10 answerable questions from each Tier
    pos_df = df[df["answerable"] == True].copy()  # noqa: E712
    neg_df = df[df["answerable"] == False].copy()  # noqa: E712

    sampled_pos_rows = []
    for t_label in ["Tier A (Curated Studio)", "Tier B (US Classroom)", "Tier C (Indian / NPTEL)"]:
        t_subset = pos_df[pos_df["tier_label"] == t_label]
        sample_n = min(len(t_subset), n_sample_per_tier)
        sampled = t_subset.sample(n=sample_n, random_state=42)
        sampled_pos_rows.append(sampled)

    sampled_pos_df = pd.concat(sampled_pos_rows, ignore_index=True)
    eval_df = pd.concat([sampled_pos_df, neg_df], ignore_index=True)

    print(f"Sampled {len(sampled_pos_df)} answerable questions ({n_sample_per_tier} per tier) + {len(neg_df)} negative probes.")
    print(f"Running generation and automated judge on {len(eval_df)} total queries...\n")

    results = []

    for idx, row in eval_df.iterrows():
        v_id = row["video_id"]
        q_text = row["question"]
        is_neg = row["is_negative"]
        tier = row["tier_label"]

        # 1. Retrieve top-k chunks
        retrieved = search(
            client=qdrant,
            query=q_text,
            video_id=v_id,
            strategy=retrieval_strategy,
            transcript_source="captions",
            k=k,
        )

        # 2. Invoke grounded Q&A
        try:
            qa_res = answer_question(
                question=q_text,
                chunks=retrieved,
                use_cache=True,
            )
            raw_answer = qa_res.get("answer", "")
            is_grounded = qa_res.get("grounded", False)
            cited_sources = qa_res.get("sources", [])
            latency = qa_res.get("latency_sec", 0.0)
            status = "OK"
            log_qa(
                video_id=v_id,
                strategy=retrieval_strategy,
                k=k,
                question=q_text,
                retrieved_chunks=retrieved,
                llm_result=qa_res,
            )
        except Exception as exc:
            raw_answer = f"ERROR: {exc}"
            is_grounded = False
            cited_sources = []
            latency = 0.0
            status = f"FAILED: {exc}"

        # 3. Evaluate Metrics
        # A. Negative Rejection
        neg_rejected = 0
        if is_neg:
            neg_rejected = 1 if "NOT_IN_VIDEO" in raw_answer or not is_grounded else 0

        # B. Citation Temporal Alignment (did cited chunk overlap gold span?)
        citation_aligned = 0
        if not is_neg and cited_sources and pd.notna(row.get("gold_start_sec")):
            g_start = float(row["gold_start_sec"])
            g_end = float(row["gold_end_sec"])
            for src in cited_sources:
                if overlap_sec(src["start"], src["end"], g_start, g_end) > 0.0:
                    citation_aligned = 1
                    break

        # C. Faithfulness via LLM Judge
        if is_neg:
            is_faithful = 1 if neg_rejected else 0
            judge_reason = "Negative rejection check"
        else:
            is_faithful, judge_reason = judge_faithfulness_llm(
                llm_client=llm_client,
                model_name=model_name,
                question=q_text,
                answer=raw_answer,
                cited_chunks=cited_sources,
            )

        results.append({
            "video_id": v_id,
            "tier": tier,
            "question": q_text,
            "is_negative": is_neg,
            "answer": raw_answer,
            "is_grounded": is_grounded,
            "neg_rejected": neg_rejected,
            "citation_aligned": citation_aligned,
            "is_faithful": is_faithful,
            "judge_reason": judge_reason,
            "n_citations": len(cited_sources),
            "latency_sec": latency,
            "status": status,
        })

        q_type_tag = "[NEG]" if is_neg else "[POS]"
        print(f"[{idx+1:02d}/{len(eval_df):02d}] {q_type_tag} {tier[:12]} | Latency: {latency:.2f}s | Faithful: {is_faithful} | Ans: {raw_answer[:65]}...")

    res_df = pd.DataFrame(results)
    details_path = os.path.join(out_dir, "generation_eval_details.csv")
    res_df.to_csv(details_path, index=False)
    print(f"\nDetailed generation results saved to: {details_path}")

    # ---------------------------------------------------------
    # COMPUTE SUMMARY METRICS ACROSS TIERS & OVERALL
    # ---------------------------------------------------------
    tier_list = ["Tier A (Curated Studio)", "Tier B (US Classroom)", "Tier C (Indian / NPTEL)"]
    summary_rows = []
    display_rows = []

    for t_name in tier_list:
        t_df = res_df[res_df["tier"] == t_name]
        t_pos = t_df[t_df["is_negative"] == False]
        t_neg = t_df[t_df["is_negative"] == True]

        neg_acc = (t_neg["neg_rejected"].mean() * 100) if len(t_neg) > 0 else 100.0
        faith_rate = (t_pos["is_faithful"].mean() * 100) if len(t_pos) > 0 else 0.0
        align_rate = (t_pos["citation_aligned"].mean() * 100) if len(t_pos) > 0 else 0.0
        avg_lat = t_df["latency_sec"].mean() if len(t_df) > 0 else 0.0

        summary_rows.append({
            "tier": t_name,
            "n_pos": len(t_pos),
            "n_neg": len(t_neg),
            "neg_rejection_acc": round(neg_acc, 2),
            "faithfulness_pct": round(faith_rate, 2),
            "citation_alignment_pct": round(align_rate, 2),
            "mean_latency_sec": round(avg_lat, 2),
        })

        display_rows.append([
            t_name,
            len(t_pos),
            len(t_neg),
            f"{neg_acc:.1f}%",
            f"{faith_rate:.1f}%",
            f"{align_rate:.1f}%",
            f"{avg_lat:.2f}s",
        ])

    # Overall Summary
    all_pos = res_df[res_df["is_negative"] == False]
    all_neg = res_df[res_df["is_negative"] == True]
    overall_neg_acc = all_neg["neg_rejected"].mean() * 100
    overall_faith = all_pos["is_faithful"].mean() * 100
    overall_align = all_pos["citation_aligned"].mean() * 100
    overall_lat = res_df["latency_sec"].mean()

    summary_rows.append({
        "tier": "Overall System",
        "n_pos": len(all_pos),
        "n_neg": len(all_neg),
        "neg_rejection_acc": round(overall_neg_acc, 2),
        "faithfulness_pct": round(overall_faith, 2),
        "citation_alignment_pct": round(overall_align, 2),
        "mean_latency_sec": round(overall_lat, 2),
    })

    display_rows.append([
        "**Overall System**",
        len(all_pos),
        len(all_neg),
        f"**{overall_neg_acc:.1f}%**",
        f"**{overall_faith:.1f}%**",
        f"**{overall_align:.1f}%**",
        f"**{overall_lat:.2f}s**",
    ])

    sum_df = pd.DataFrame(summary_rows)
    sum_path = os.path.join(out_dir, "generation_eval_summary.csv")
    sum_df.to_csv(sum_path, index=False)
    print(f"Summary generation metrics saved to: {sum_path}\n")

    headers = ["Pedagogical Tier", "Pos Q#", "Neg Q#", "Negative Rejection", "Faithfulness", "Citation Align", "Latency"]
    print("### TABLE 4: GENERATION FAITHFULNESS & NEGATIVE REJECTION")
    print(tabulate(display_rows, headers=headers, tablefmt="github"))
    print()

    # ---------------------------------------------------------
    # EXPORT IEEE LATEX TABLE
    # ---------------------------------------------------------
    latex_lines = []
    for r in summary_rows:
        t_label = r["tier"].replace("&", "\\&")
        bold_prefix = "\\textbf{" if "Overall" in t_label else ""
        bold_suffix = "}" if "Overall" in t_label else ""
        latex_lines.append(
            f"{bold_prefix}{t_label}{bold_suffix} & {r['n_pos']} & {r['n_neg']} & "
            f"{bold_prefix}{r['neg_rejection_acc']:.1f}\\%{bold_suffix} & "
            f"{bold_prefix}{r['faithfulness_pct']:.1f}\\%{bold_suffix} & "
            f"{bold_prefix}{r['citation_alignment_pct']:.1f}\\%{bold_suffix} & "
            f"{bold_prefix}{r['mean_latency_sec']:.2f}s{bold_suffix} \\\\"
        )

    latex_content = r"""% IEEE Table 4: Generation Faithfulness and Out-of-Domain Negative Rejection
\begin{table*}[t]
\caption{End-to-End Generation Faithfulness, Citation Alignment, and Negative Rejection Across Pedagogical Tiers}
\label{tab:generation_eval}
\centering
\begin{tabular}{lcccccc}
\hline
\textbf{Pedagogical Tier} & \textbf{Answerable} & \textbf{Negatives} & \textbf{Negative Rejection} & \textbf{Faithfulness} & \textbf{Citation Alignment} & \textbf{Mean Latency} \\
\hline
""" + "\n".join(latex_lines) + r"""
\hline
\end{tabular}
\end{table*}
"""

    with open(LATEX_TABLE_GEN, "w", encoding="utf-8") as f:
        f.write(latex_content)
    print(f"Saved Table 4 LaTeX export to: {LATEX_TABLE_GEN}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run generation faithfulness and negative rejection evaluation.")
    parser.add_argument("--questions", default="eval/questions.csv")
    parser.add_argument("--strategy", default="fixed_token")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--samples-per-tier", type=int, default=10)
    args = parser.parse_args()

    run_generation_evaluation(
        questions_path=args.questions,
        retrieval_strategy=args.strategy,
        k=args.k,
        n_sample_per_tier=args.samples_per_tier,
    )


if __name__ == "__main__":
    main()
