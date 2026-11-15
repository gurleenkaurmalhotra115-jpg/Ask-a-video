"""Build an anonymous Supplementary Material ZIP bundle for conference submission."""

import os
import zipfile

OUTPUT_ZIP = "Ask_a_Video_Supplementary_Material.zip"

FILES_TO_INCLUDE = [
    "paper.pdf",
    "README_SUPPLEMENTARY.md",
    "eval/questions.csv",
    "eval/corpus_meta.json",
    "eval/eval_mcnemar.py",
    "eval/eval_hierarchical.py",
    "eval/profile_indexing.py",
    "eval/eval_generation.py",
    "eval/evaluate.py",
    "eval/metrics.py",
    "eval/common.py",
    "results/retrieval_summary.csv",
    "results/tier_breakdown.csv",
    "results/duration_breakdown.csv",
    "results/generation_eval_summary.csv",
    "results/generation_eval_details.csv",
    "results/retrieval_per_question.csv",
]

def build_zip():
    print(f"Creating anonymous supplementary bundle: {OUTPUT_ZIP} ...")
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in FILES_TO_INCLUDE:
            if os.path.exists(f):
                zf.write(f, arcname=f)
                print(f"  + Added: {f}")
            else:
                print(f"  ! Warning: {f} not found!")

    size_mb = os.path.getsize(OUTPUT_ZIP) / (1024 * 1024)
    print(f"\n[SUCCESS] Successfully packaged {len(FILES_TO_INCLUDE)} files into {OUTPUT_ZIP} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    build_zip()
