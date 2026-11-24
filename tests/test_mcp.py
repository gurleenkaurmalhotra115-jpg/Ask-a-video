"""Unit tests for the VidQuery MCP Server."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

import mcp_server


def test_mcp_server_initialization_and_tools():
    """Verify that the MCP server initializes with the expected tool metadata."""
    assert mcp_server.mcp.name == "vidquery-mcp"
    # Verify tools exist as callables in the module
    assert callable(mcp_server.search_video_moments)
    assert callable(mcp_server.ask_grounded_question)
    assert callable(mcp_server.verify_citations)


def test_mcp_search_video_moments_fallback_and_structure():
    """Verify search_video_moments returns formatted moment dictionaries."""
    mock_results = [
        {"chunk_id": 1, "start": 10.0, "end": 25.0, "text": "Gradient descent computes slopes.", "score": 0.88, "strategy": "fixed_token"},
        {"chunk_id": 2, "start": 30.0, "end": 45.0, "text": "Learning rate controls step size.", "score": 0.72, "strategy": "fixed_token"},
    ]
    with patch("mcp_server.qdrant_search", return_value=mock_results):
        moments = mcp_server.search_video_moments(video_id="dummy_video", query="gradient descent", k=2)
        assert len(moments) == 2
        assert moments[0]["chunk_id"] == 1
        assert moments[0]["score"] == 0.88
        assert "Gradient descent" in moments[0]["text"]


def test_mcp_ask_grounded_question_low_confidence_gate():
    """Verify that low retrieval confidence triggers safe refusal without hallucination."""
    mock_chunks = [
        {"chunk_id": 1, "start": 0.0, "end": 10.0, "text": "Unrelated topic", "score": 0.05, "strategy": "fixed_token"}
    ]
    with patch("mcp_server.search_video_moments", return_value=mock_chunks):
        res = mcp_server.ask_grounded_question(video_id="dummy_video", question="Quantum gravity", min_confidence=0.22)
        assert "could not find sufficient information" in res["answer"]
        assert res["confidence"]["passed"] is False
        assert res["verification"]["verdict"] == "supported"


def test_mcp_ask_grounded_question_high_confidence_success():
    """Verify that confident retrieval invokes answer generation and citation verification."""
    mock_chunks = [
        {"chunk_id": 1, "start": 5.0, "end": 20.0, "text": "Attention is all you need.", "score": 0.92, "strategy": "fixed_token"}
    ]
    mock_qa = {
        "answer": "Transformers rely on self-attention [1].",
        "sources": [{"chunk_id": 1, "start": 5.0, "end": 20.0, "text": "Attention is all you need."}],
        "model_used": "gemini-2.5-flash",
        "cached": False,
    }
    mock_verif = {
        "verdict": "supported",
        "cited_indices": [1],
        "reasoning": "Direct match in chunk 1",
    }
    with patch("mcp_server.search_video_moments", return_value=mock_chunks), \
         patch("mcp_server.answer_question", return_value=mock_qa), \
         patch("mcp_server.verify_answer", return_value=mock_verif):
        res = mcp_server.ask_grounded_question(video_id="dummy_video", question="What do transformers use?")
        assert res["answer"] == "Transformers rely on self-attention [1]."
        assert res["confidence"]["passed"] is True
        assert len(res["sources"]) == 1
        assert res["verification"]["verdict"] == "supported"


def test_mcp_verify_citations_tool():
    """Verify verify_citations tool calls verify_answer and returns verdict."""
    mock_verif = {
        "verdict": "supported",
        "cited_indices": [1],
        "reasoning": "Statement is grounded in passage.",
    }
    with patch("mcp_server.verify_answer", return_value=mock_verif):
        res = mcp_server.verify_citations(
            question="What is backprop?",
            answer="Backpropagation computes gradients using the chain rule [1].",
            chunk_texts=["Backprop uses the chain rule to calculate weight gradients."],
        )
        assert res["verdict"] == "supported"
        assert res["cited_indices"] == [1]
