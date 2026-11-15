# Ask-a-Video: Supplementary Material & Reproduction Guide

This repository contains the evaluation artifacts, gold benchmarks, results, and reproduction scripts for the paper:
**"Evaluating Chunking Strategies and Temporal Grounding in Long-Form Educational Video RAG Across Acoustic and Pedagogical Tiers"**

---

## 1. Directory Structure

```
.
├── paper.pdf                           # Complete 6-page camera-ready manuscript
├── eval/
│   ├── questions.csv                   # 156 human-annotated queries (130 answerable + 26 negative probes)
│   ├── corpus_meta.json                # Video metadata across Tiers A, B, and C
│   ├── eval_mcnemar.py                 # Paired McNemar test on discordant pairs (p = 0.508)
│   ├── eval_hierarchical.py            # Two-Stage Hierarchical Video Retrieval (Algorithm 2)
│   ├── profile_indexing.py             # Preprocessing & indexing throughput profiler (Table IV)
│   ├── eval_generation.py              # End-to-end generation & automated judge evaluation
│   ├── evaluate.py                     # Main retrieval benchmark evaluator
│   ├── metrics.py                      # MedVidQA IoU and boundary error calculations
│   └── common.py                       # Data loading and guardrail validators
├── results/
│   ├── retrieval_summary.csv           # Table I: Overall retrieval performance benchmark
│   ├── tier_breakdown.csv              # Table II: Pedagogical & acoustic tier breakdown
│   ├── duration_breakdown.csv          # Table III: Search space expansion breakdown
│   ├── generation_eval_summary.csv     # Table V: Generation & attribution evaluation summary
│   ├── generation_eval_details.csv     # Per-query generation outputs and judge rationales
│   └── retrieval_per_question.csv      # Raw per-query retrieval predictions and metrics
└── README_SUPPLEMENTARY.md             # This document
```

---

## 2. Environment Setup

```bash
python -m venv venv
# Linux / macOS
source venv/bin/activate
# Windows
.\venv\Scripts\activate

pip install pandas numpy scipy tabulate
```

---

## 3. Step-by-Step Reproduction Instructions

### A. Reproducing Table I (Paired McNemar Significance Test)
To verify that semantic boundary chunking yields no statistically significant retrieval gain over simple fixed-token partitioning ($p = 0.508$ on 9 discordant query pairs):
```bash
python eval/eval_mcnemar.py
```
*Expected Output:*
```
Total Evaluated Queries: 130
Both Hit:                5
Fixed-Token Only:        6
Semantic Only:           3
Both Miss:               116
Discordant Pairs (b+c):  9 (b=6, c=3)
Two-Sided Exact p-value: 0.5078
Conclusion: No statistically significant difference (p > 0.05).
```

### B. Reproducing Table III & Section VI (Hierarchical Retrieval Recovery)
To evaluate the Two-Stage Hierarchical Video Retriever (Algorithm 2) on the 100 long-lecture queries ($>35$ min):
```bash
python eval/eval_hierarchical.py
```
*Expected Output:*
```
Single-Stage Flat (fixed_token):      Hit@1:  6.0% ( 6/100), Med Error: 1012.6s
Two-Stage Hierarchical (fixed_token): Hit@1: 18.0% (18/100), Med Error:  264.1s
Bootstrap 95% Confidence Interval for Hierarchical Hit@1: [11.0%, 25.0%]
Boundary Error Reduction: 73.9% (from 1012.6s to 264.1s)
```

### C. Reproducing Table IV (Computational Indexing Latency)
To profile the 160x chunking overhead and 4.4x total indexing penalty:
```bash
python eval/profile_indexing.py
```

### D. Reproducing Table V (Generation & Attribution Evaluation)
To verify the 100% adversarial negative probe rejection and answerable rejection breakdown across sampled queries (seed 42):
```bash
python -c "
import pandas as pd
df = pd.read_csv('results/generation_eval_details.csv')
pos = df[df['is_negative'] == False]
for tier, g in pos.groupby('tier'):
    ans = (g['answer'] != 'NOT_IN_VIDEO').sum()
    rej = (g['answer'] == 'NOT_IN_VIDEO').sum()
    print(f'{tier}: Answered = {ans}, Rejected = {rej}')
"
```
*Expected Output:*
```
Tier A (Curated Studio): Answered = 7, Rejected = 3
Tier B (US Classroom): Answered = 7, Rejected = 3
Tier C (Indian / NPTEL): Answered = 0, Rejected = 10
```

---

## 4. Anonymity & Open-Source Commitment
All author identifiers, institution references, and repository usernames have been omitted to adhere to double-blind peer-review guidelines. Full source code, weights, and interactive demonstration packages will be unmasked and publicly released upon formal paper acceptance.
