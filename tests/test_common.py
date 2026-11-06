"""Unit tests for eval/common.py question loading and guardrails."""

import tempfile
import pytest
import pandas as pd
from eval.common import load_questions


def test_schema_b_legacy_success():
    csv_content = """video_id,question,answer_start_sec,answer_end_sec,type,answerable
vid1,What is gradient descent?,10.0,40.0,factual,True
vid1,Who invented calculus?,,,,False
"""
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write(csv_content)
        path = f.name

    df = load_questions(path)
    assert len(df) == 2
    assert bool(df.iloc[0]["answerable"]) is True
    assert bool(df.iloc[0]["is_negative"]) is False
    assert df.iloc[0]["gold_start_sec"] == 10.0
    assert df.iloc[0]["gold_end_sec"] == 40.0
    assert bool(df.iloc[1]["answerable"]) is False
    assert bool(df.iloc[1]["is_negative"]) is True


def test_schema_a_modern_success():
    csv_content = """video_id,tier,question,gold_start_sec,gold_end_sec,gold_answer_key,is_negative
vid2,tier_a,Explain backpropagation,120.0,175.0,Chain rule applied to weights,False
vid2,tier_a,What is the capital of Mars?,,,NOT_IN_VIDEO,True
"""
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write(csv_content)
        path = f.name

    df = load_questions(path)
    assert len(df) == 2
    assert bool(df.iloc[0]["answerable"]) is True
    assert bool(df.iloc[0]["is_negative"]) is False
    assert df.iloc[0]["gold_start_sec"] == 120.0
    assert df.iloc[0]["gold_end_sec"] == 175.0
    assert bool(df.iloc[1]["is_negative"]) is True


def test_guardrail_rejects_spans_over_90_seconds():
    # 200s span should be rejected by the guardrail
    csv_content = """video_id,question,answer_start_sec,answer_end_sec,type,answerable
vid1,What is gradient descent?,100.0,305.0,factual,True
"""
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write(csv_content)
        path = f.name

    with pytest.raises(ValueError, match="exceeds the strict 90.0s maximum guardrail"):
        load_questions(path)


def test_rejection_when_end_not_greater_than_start():
    csv_content = """video_id,question,answer_start_sec,answer_end_sec,type,answerable
vid1,What is gradient descent?,50.0,40.0,factual,True
"""
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write(csv_content)
        path = f.name

    with pytest.raises(ValueError, match="end_sec .* must be greater than start_sec"):
        load_questions(path)
