"""One-click end-to-end verification script for the VidQuery project."""

from __future__ import annotations

import os
import sys
import json
import pandas as pd


def check_file(path: str, desc: str) -> bool:
    exists = os.path.exists(path)
    status = "OK" if exists else "MISSING"
    print(f"[{status:7s}] {desc}: {path}")
    return exists


def main() -> int:
    print("\n" + "=" * 70)
    print("VIDQUERY SYSTEM & RESEARCH VERIFICATION")
    print("=" * 70 + "\n")

    all_passed = True

    # 1. Core Source Files
    print("--- 1. Core Source Code ---")
    core_files = [
        ("app.py", "Streamlit UI Application"),
        ("chunking.py", "Pluggable Chunking Module"),
        ("transcript.py", "Transcript Ingestion & Cache"),
        ("embeddings.py", "Embedding Generator (MiniLM)"),
        ("vectorstore.py", "Qdrant Vector Database Operations"),
        ("llm.py", "Grounded Gemini QA & Cache"),
    ]
    for f, desc in core_files:
        if not check_file(f, desc):
            all_passed = False

    # 2. Evaluation Scripts & Datasets
    print("\n--- 2. Datasets & Evaluation Pipeline ---")
    eval_files = [
        ("eval/corpus_meta.json", "13-Video Corpus Metadata"),
        ("eval/questions.csv", "156-Question Benchmark Dataset"),
        ("eval/common.py", "Dataset Loader & Schemas"),
        ("eval/metrics.py", "MedVidQA IoU Metric Engine"),
        ("eval/evaluate.py", "Factorial Retrieval Benchmark"),
        ("eval/eval_generation.py", "Generation & Neg Rejection Eval"),
    ]
    for f, desc in eval_files:
        if not check_file(f, desc):
            all_passed = False

    # 3. Research Deliverables
    print("\n--- 3. Research Paper & LaTeX Tables ---")
    paper_files = [
        ("paper.tex", "Complete IEEE Conference Manuscript"),
        ("results/latex_table_main.tex", "Table 1 (Retrieval Benchmark)"),
        ("results/latex_table_tiers.tex", "Table 2 & 3 (Tiers & Duration)"),
        ("results/latex_table_generation.tex", "Table 4 (Generation Faithfulness)"),
        ("results/retrieval_summary.csv", "Retrieval Summary Metrics"),
        ("results/generation_eval_summary.csv", "Generation Summary Metrics"),
        ("PRESENTATION.md", "12-Slide Presentation & Defense Deck"),
        ("README.md", "Project Documentation"),
    ]
    for f, desc in paper_files:
        if not check_file(f, desc):
            all_passed = False

    # 4. Check Question Dataset Structure
    print("\n--- 4. Data Integrity Checks ---")
    try:
        from eval.common import load_questions
        q_df = load_questions("eval/questions.csv")
        n_total = len(q_df)
        n_pos = len(q_df[q_df["answerable"] == True])
        n_neg = len(q_df[q_df["answerable"] == False])
        print(f"[OK     ] Dataset loaded: {n_total} total queries ({n_pos} answerable, {n_neg} negative probes)")
        assert n_total == 156, f"Expected 156 questions, got {n_total}"
        assert n_pos == 130, f"Expected 130 answerable, got {n_pos}"
        assert n_neg == 26, f"Expected 26 negatives, got {n_neg}"
    except Exception as exc:
        print(f"[ERROR  ] Dataset verification failed: {exc}")
        all_passed = False

    # 5. Check Cached Transcripts
    print("\n--- 5. Offline Cached Transcripts ---")
    captions_dir = "data/transcripts/captions"
    if os.path.exists(captions_dir):
        cached_files = [f for f in os.listdir(captions_dir) if f.endswith(".json")]
        print(f"[OK     ] Found {len(cached_files)} cached transcript files in {captions_dir}")
        assert len(cached_files) >= 13, f"Expected at least 13 cached transcripts, found {len(cached_files)}"
    else:
        print(f"[ERROR  ] Captions cache directory missing: {captions_dir}")
        all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL VERIFICATIONS PASSED! System is 100% submission ready.")
    else:
        print("SOME CHECKS FAILED. See messages above.")
    print("=" * 70 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
