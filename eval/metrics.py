"""Pure evaluation metrics for temporal chunk retrieval against ground-truth spans."""

from __future__ import annotations

from typing import Any
import numpy as np


def overlap_sec(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Compute the temporal overlap in seconds between two intervals [a_start, a_end] and [b_start, b_end].

    Returns 0.0 if the intervals do not overlap or if either interval is degenerate.
    """
    inter_start = max(float(a_start), float(b_start))
    inter_end = min(float(a_end), float(b_end))
    return max(0.0, inter_end - inter_start)


def iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Compute the Intersection over Union (IoU) of two temporal spans a=(start, end) and b=(start, end).

    Returns 0.0 if union is non-positive or if intervals do not overlap.
    """
    a_start, a_end = float(a[0]), float(a[1])
    b_start, b_end = float(b[0]), float(b[1])

    intersection = overlap_sec(a_start, a_end, b_start, b_end)
    dur_a = max(0.0, a_end - a_start)
    dur_b = max(0.0, b_end - b_start)
    union = dur_a + dur_b - intersection

    if union <= 0.0:
        return 0.0
    return intersection / union


def compute_retrieval_metrics(
    ranked_chunks: list[dict[str, Any]],
    gold_spans: list[tuple[float, float]],
) -> dict[str, Any]:
    """Compute ranked retrieval metrics for a single question against gold spans.

    Args:
        ranked_chunks: Retrieved chunk dicts containing 'start', 'end', and optional 'score'/'chunk_id',
                       ordered from top-1 to top-k.
        gold_spans: List of valid (start, end) tuples representing acceptable answer spans (main + optional alt).

    Returns:
        dict:
            - hit_at_1: 1 if top-1 chunk overlaps any gold span, else 0
            - hit_at_k: 1 if any of the top-k chunks overlaps any gold span, else 0
            - reciprocal_rank: 1 / rank of first overlapping chunk (0.0 if no hits)
            - iou_top1: IoU of top-1 chunk vs the gold span it overlaps most
            - start_error_top1: signed difference (chunk.start - gold.start) for that gold span
            - within10_top1: 1 if abs(start_error_top1) <= 10.0, else 0
            - r1_iou_03: 1 if top-1 chunk has IoU >= 0.3 against any gold span (MedVidQA standard)
            - r1_iou_05: 1 if top-1 chunk has IoU >= 0.5 against any gold span
            - r1_iou_07: 1 if top-1 chunk has IoU >= 0.7 against any gold span
            - rk_iou_03: 1 if any top-k chunk has IoU >= 0.3 against any gold span
            - rk_iou_05: 1 if any top-k chunk has IoU >= 0.5 against any gold span
            - rk_iou_07: 1 if any top-k chunk has IoU >= 0.7 against any gold span
            - top1_score: score of the top-1 chunk (or 0.0)
    """
    default_res = {
        "hit_at_1": 0,
        "hit_at_k": 0,
        "reciprocal_rank": 0.0,
        "iou_top1": 0.0,
        "start_error_top1": 0.0,
        "within10_top1": 0,
        "r1_iou_03": 0,
        "r1_iou_05": 0,
        "r1_iou_07": 0,
        "rk_iou_03": 0,
        "rk_iou_05": 0,
        "rk_iou_07": 0,
        "top1_score": 0.0,
    }

    valid_spans = [
        (float(s[0]), float(s[1]))
        for s in gold_spans
        if s is not None and len(s) == 2 and float(s[1]) > float(s[0])
    ]

    if not ranked_chunks or not valid_spans:
        return default_res

    # Check hits across ranked chunks
    first_hit_rank: int | None = None
    max_iou_any_rank = 0.0
    for r_idx, chunk in enumerate(ranked_chunks, start=1):
        c_start = float(chunk["start"])
        c_end = float(chunk["end"])
        chunk_ious = [iou((c_start, c_end), g) for g in valid_spans]
        if chunk_ious:
            max_iou_any_rank = max(max_iou_any_rank, max(chunk_ious))
        if any(overlap_sec(c_start, c_end, g[0], g[1]) > 0.0 for g in valid_spans):
            if first_hit_rank is None:
                first_hit_rank = r_idx

    hit_at_k = 1 if first_hit_rank is not None else 0
    reciprocal_rank = 1.0 / first_hit_rank if first_hit_rank is not None else 0.0

    # Top-1 chunk metrics
    top1 = ranked_chunks[0]
    top1_start = float(top1["start"])
    top1_end = float(top1["end"])
    top1_score = float(top1.get("score", 0.0))

    hit_at_1 = 1 if any(overlap_sec(top1_start, top1_end, g[0], g[1]) > 0.0 for g in valid_spans) else 0

    # Find the gold span that top-1 overlaps most (highest IoU)
    # If all IoUs are 0, choose the span with the closest start time
    best_span = max(
        valid_spans,
        key=lambda g: (iou((top1_start, top1_end), g), -abs(top1_start - g[0])),
    )

    iou_top1 = iou((top1_start, top1_end), best_span)
    start_error_top1 = top1_start - best_span[0]
    within10_top1 = 1 if abs(start_error_top1) <= 10.0 else 0

    # MedVidQA standard thresholds on IoU
    r1_iou_03 = 1 if iou_top1 >= 0.3 else 0
    r1_iou_05 = 1 if iou_top1 >= 0.5 else 0
    r1_iou_07 = 1 if iou_top1 >= 0.7 else 0

    # Top-k IoU thresholds
    rk_iou_03 = 1 if max_iou_any_rank >= 0.3 else 0
    rk_iou_05 = 1 if max_iou_any_rank >= 0.5 else 0
    rk_iou_07 = 1 if max_iou_any_rank >= 0.7 else 0

    return {
        "hit_at_1": hit_at_1,
        "hit_at_k": hit_at_k,
        "reciprocal_rank": reciprocal_rank,
        "iou_top1": iou_top1,
        "start_error_top1": start_error_top1,
        "within10_top1": within10_top1,
        "r1_iou_03": r1_iou_03,
        "r1_iou_05": r1_iou_05,
        "r1_iou_07": r1_iou_07,
        "rk_iou_03": rk_iou_03,
        "rk_iou_05": rk_iou_05,
        "rk_iou_07": rk_iou_07,
        "top1_score": top1_score,
    }


def compute_bootstrap_ci(
    values: list[float] | np.ndarray,
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    random_state: int | None = 42,
) -> tuple[float, float, float]:
    """Compute empirical bootstrap confidence interval for sample mean.

    Args:
        values: 1D array-like collection of metric values.
        n_bootstraps: Number of bootstrap iterations.
        ci: Confidence interval nominal coverage (e.g. 0.95 for 95% CI).
        random_state: Optional seed for reproducible resampling.

    Returns:
        tuple[float, float, float]: (mean, ci_lower, ci_upper)
    """
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0

    mean_val = float(np.mean(arr))
    if len(arr) == 1:
        return mean_val, mean_val, mean_val

    rng = np.random.default_rng(random_state)
    n = len(arr)
    # Generate bootstrap sample means
    sample_indices = rng.integers(0, n, size=(n_bootstraps, n))
    bootstrap_means = np.mean(arr[sample_indices], axis=1)

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(bootstrap_means, alpha * 100.0))
    upper = float(np.percentile(bootstrap_means, (1.0 - alpha) * 100.0))

    return mean_val, lower, upper

