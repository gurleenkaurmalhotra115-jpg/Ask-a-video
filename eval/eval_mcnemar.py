"""Paired McNemar test on discordant query pairs between fixed_token and semantic chunking."""

from __future__ import annotations
import pandas as pd
from scipy.stats import binomtest

def run_mcnemar_test(csv_path: str = "results/retrieval_per_question.csv") -> None:
    df = pd.read_csv(csv_path)
    
    # Filter to answerable queries
    fixed = df[df["strategy"] == "fixed_token"].set_index(["video_id", "question"])
    semantic = df[df["strategy"] == "semantic"].set_index(["video_id", "question"])
    
    common_idx = fixed.index.intersection(semantic.index)
    f_hits = fixed.loc[common_idx, "hit_at_1"].astype(int)
    s_hits = semantic.loc[common_idx, "hit_at_1"].astype(int)
    
    # Contingency table
    both_hit = ((f_hits == 1) & (s_hits == 1)).sum()
    fixed_only = ((f_hits == 1) & (s_hits == 0)).sum()
    semantic_only = ((f_hits == 0) & (s_hits == 1)).sum()
    both_miss = ((f_hits == 0) & (s_hits == 0)).sum()
    
    total = len(common_idx)
    discordant = fixed_only + semantic_only
    
    # Exact binomial test (two-sided)
    res = binomtest(min(fixed_only, semantic_only), n=discordant, p=0.5, alternative="two-sided")
    
    print("=" * 60)
    print("Paired McNemar Test: fixed_token vs. semantic (Hit@1)")
    print("=" * 60)
    print(f"Total Evaluated Queries: {total}")
    print(f"Both Hit:                {both_hit}")
    print(f"Fixed-Token Only:        {fixed_only}")
    print(f"Semantic Only:           {semantic_only}")
    print(f"Both Miss:               {both_miss}")
    print(f"Discordant Pairs (b+c):  {discordant} (b={fixed_only}, c={semantic_only})")
    print(f"Two-Sided Exact p-value: {res.pvalue:.4f}")
    print("=" * 60)
    if res.pvalue > 0.05:
        print("Conclusion: No statistically significant difference (p > 0.05).")
    else:
        print("Conclusion: Statistically significant difference (p <= 0.05).")

if __name__ == "__main__":
    run_mcnemar_test()
