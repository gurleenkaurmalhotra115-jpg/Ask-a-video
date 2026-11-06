"""Unit tests for retrieval evaluation metrics in eval/metrics.py."""

from __future__ import annotations

import pytest

from eval.metrics import compute_bootstrap_ci, compute_retrieval_metrics, iou, overlap_sec


def test_overlap_sec_disjoint():
    assert overlap_sec(0.0, 10.0, 20.0, 30.0) == 0.0
    assert overlap_sec(50.0, 60.0, 10.0, 20.0) == 0.0


def test_overlap_sec_partial():
    assert overlap_sec(10.0, 30.0, 20.0, 40.0) == 10.0
    assert overlap_sec(20.0, 40.0, 10.0, 30.0) == 10.0


def test_overlap_sec_containment():
    # b contained in a
    assert overlap_sec(10.0, 50.0, 20.0, 30.0) == 10.0
    # a contained in b
    assert overlap_sec(20.0, 30.0, 10.0, 50.0) == 10.0


def test_overlap_sec_touching_boundaries():
    assert overlap_sec(10.0, 20.0, 20.0, 30.0) == 0.0
    assert overlap_sec(20.0, 30.0, 10.0, 20.0) == 0.0


def test_iou_identical():
    assert iou((10.0, 25.0), (10.0, 25.0)) == 1.0


def test_iou_disjoint():
    assert iou((0.0, 10.0), (20.0, 30.0)) == 0.0


def test_iou_partial():
    # [10, 30] (len 20) and [20, 40] (len 20): inter = 10, union = 20 + 20 - 10 = 30
    assert iou((10.0, 30.0), (20.0, 40.0)) == pytest.approx(10.0 / 30.0)


def test_iou_degenerate():
    assert iou((10.0, 10.0), (10.0, 20.0)) == 0.0
    assert iou((10.0, 5.0), (10.0, 20.0)) == 0.0


def test_reciprocal_rank_ranks():
    gold = [(100.0, 150.0)]

    # Rank 1 hit
    chunks_r1 = [
        {"start": 110.0, "end": 140.0, "score": 0.9},
        {"start": 0.0, "end": 50.0, "score": 0.5},
    ]
    m1 = compute_retrieval_metrics(chunks_r1, gold)
    assert m1["hit_at_1"] == 1
    assert m1["hit_at_k"] == 1
    assert m1["reciprocal_rank"] == 1.0

    # Rank 2 hit
    chunks_r2 = [
        {"start": 0.0, "end": 50.0, "score": 0.9},
        {"start": 110.0, "end": 140.0, "score": 0.8},
    ]
    m2 = compute_retrieval_metrics(chunks_r2, gold)
    assert m2["hit_at_1"] == 0
    assert m2["hit_at_k"] == 1
    assert m2["reciprocal_rank"] == 0.5

    # Rank 3 hit
    chunks_r3 = [
        {"start": 0.0, "end": 50.0, "score": 0.9},
        {"start": 60.0, "end": 80.0, "score": 0.8},
        {"start": 105.0, "end": 125.0, "score": 0.7},
        {"start": 200.0, "end": 250.0, "score": 0.6},
    ]
    m3 = compute_retrieval_metrics(chunks_r3, gold)
    assert m3["hit_at_1"] == 0
    assert m3["hit_at_k"] == 1
    assert m3["reciprocal_rank"] == pytest.approx(1.0 / 3.0)


def test_within10_top1():
    gold = [(100.0, 150.0)]

    # Difference = 106.0 - 100.0 = +6.0 (within 10)
    chunks_in = [{"start": 106.0, "end": 140.0, "score": 0.85}]
    m_in = compute_retrieval_metrics(chunks_in, gold)
    assert m_in["start_error_top1"] == pytest.approx(6.0)
    assert m_in["within10_top1"] == 1

    # Difference = 115.0 - 100.0 = +15.0 (outside 10)
    chunks_out = [{"start": 115.0, "end": 160.0, "score": 0.85}]
    m_out = compute_retrieval_metrics(chunks_out, gold)
    assert m_out["start_error_top1"] == pytest.approx(15.0)
    assert m_out["within10_top1"] == 0


def test_alt_span_selection():
    # Main span: [100, 150], Alt span: [300, 350]
    gold = [(100.0, 150.0), (300.0, 350.0)]

    # Chunk overlaps alt span perfectly
    chunk = [{"start": 300.0, "end": 350.0, "score": 0.95}]
    metrics = compute_retrieval_metrics(chunk, gold)

    assert metrics["hit_at_1"] == 1
    assert metrics["hit_at_k"] == 1
    assert metrics["reciprocal_rank"] == 1.0
    assert metrics["iou_top1"] == 1.0
    assert metrics["start_error_top1"] == 0.0
    assert metrics["within10_top1"] == 1


def test_no_hit_case_returns_cleanly():
    gold = [(100.0, 150.0)]
    chunks = [
        {"start": 0.0, "end": 20.0, "score": 0.4},
        {"start": 30.0, "end": 50.0, "score": 0.3},
    ]
    metrics = compute_retrieval_metrics(chunks, gold)

    assert metrics["hit_at_1"] == 0
    assert metrics["hit_at_k"] == 0
    assert metrics["reciprocal_rank"] == 0.0
    assert metrics["iou_top1"] == 0.0
    assert metrics["within10_top1"] == 0
    assert metrics["top1_score"] == pytest.approx(0.4)


def test_empty_chunks_or_gold():
    assert compute_retrieval_metrics([], [(10.0, 20.0)])["hit_at_k"] == 0
    assert compute_retrieval_metrics([{"start": 0.0, "end": 10.0}], [])["hit_at_k"] == 0


def test_medvidqa_iou_thresholds():
    # Gold: [100, 150] (duration 50s)
    gold = [(100.0, 150.0)]

    # Chunk with partial overlap: [120, 150] (duration 30s)
    # Inter: 30s, Union: 50s. IoU = 30/50 = 0.60
    chunks = [{"start": 120.0, "end": 150.0, "score": 0.9}]
    m = compute_retrieval_metrics(chunks, gold)
    assert m["iou_top1"] == pytest.approx(0.60)
    assert m["r1_iou_03"] == 1
    assert m["r1_iou_05"] == 1
    assert m["r1_iou_07"] == 0
    assert m["rk_iou_03"] == 1
    assert m["rk_iou_05"] == 1
    assert m["rk_iou_07"] == 0

    # Chunk with small overlap: [140, 170] (duration 30s)
    # Inter: 10s ([140, 150]), Union: 50 + 30 - 10 = 70s. IoU = 10/70 = 0.1428
    chunks_low = [{"start": 140.0, "end": 170.0, "score": 0.8}]
    m_low = compute_retrieval_metrics(chunks_low, gold)
    assert m_low["iou_top1"] == pytest.approx(10.0 / 70.0)
    assert m_low["r1_iou_03"] == 0
    assert m_low["r1_iou_05"] == 0
    assert m_low["r1_iou_07"] == 0


def test_bootstrap_ci_computation():
    vals = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0]
    mean, lower, upper = compute_bootstrap_ci(vals, n_bootstraps=500, ci=0.95, random_state=42)
    assert mean == pytest.approx(0.7)
    assert 0.3 <= lower <= 0.7
    assert 0.7 <= upper <= 1.0
    assert lower <= mean <= upper

