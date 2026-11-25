"""Model Context Protocol (MCP) Server for VidQuery (Ask-a-video).

Exposes timestamp-grounded video search, confidence-gated Q&A, and citation
verification tools to any MCP-compliant agent (Claude Desktop, Cursor, Gemini CLI).
"""

from __future__ import annotations

import os
import sys
from typing import Any

from mcp.server.mcpserver import MCPServer

from chunking import DEFAULT_PARAMS, STRATEGIES
from llm import LLMError, answer_question, assess_retrieval_confidence, verify_answer
from vectorstore import get_client as get_qdrant_client, search as qdrant_search, search_hierarchical as qdrant_search_hierarchical

# Initialize MCPServer
mcp = MCPServer(name="vidquery-mcp")


@mcp.tool()
def search_video_moments(
    video_id: str,
    query: str,
    k: int = 4,
    strategy: str = "fixed_token",
    hierarchical: bool = False,
    t_macro: float = 300.0,
    top_m_macro: int = 2,
) -> list[dict[str, Any]]:
    """Search for relevant timestamped moments within an indexed YouTube lecture.

    Args:
        video_id: 11-character YouTube video ID.
        query: Natural language search query or concept keyword.
        k: Maximum number of relevant moments/chunks to retrieve (default: 4).
        strategy: Chunking strategy to filter by ('fixed_token', 'sentence', 'semantic', 'time_window').
        hierarchical: Whether to use Two-Stage Hierarchical Video Retrieval (Algorithm 1) for long lectures.
        t_macro: Macro-window duration in seconds (default: 300s = 5m).
        top_m_macro: Top-M macro-blocks selected in Stage 1 (default: 2).

    Returns:
        List of matching moment dictionaries containing text, start_time, end_time, and similarity score.
    """
    if strategy not in STRATEGIES:
        strategy = "fixed_token"

    client = get_qdrant_client()
    if hierarchical:
        results, _ = qdrant_search_hierarchical(
            client=client,
            query=query,
            video_id=video_id,
            strategy=strategy,
            t_macro=t_macro,
            top_m_macro=top_m_macro,
            k=max(1, min(20, k)),
        )
    else:
        results = qdrant_search(
            client=client,
            query=query,
            k=max(1, min(20, k)),
            video_id=video_id,
            strategy=strategy,
        )

    moments = []
    for r in results:
        m_dict = {
            "chunk_id": r.get("chunk_id"),
            "start": r.get("start"),
            "end": r.get("end"),
            "text": r.get("text"),
            "score": round(float(r.get("score", 0.0)), 4),
            "strategy": r.get("strategy", strategy),
        }
        if "macro_block_idx" in r:
            m_dict["macro_block_idx"] = r["macro_block_idx"]
            m_dict["macro_window"] = (r.get("macro_start_sec"), r.get("macro_end_sec"))
        moments.append(m_dict)
    return moments


@mcp.tool()
def ask_grounded_question(
    video_id: str,
    question: str,
    k: int = 4,
    strategy: str = "fixed_token",
    hierarchical: bool = False,
    min_confidence: float = 0.22,
    verify_citations_agent: bool = True,
) -> dict[str, Any]:
    """Ask a question about an indexed lecture with citation grounding and confidence gating.

    If retrieval confidence is insufficient (e.g. out-of-domain probe), the question is
    safely rejected without hallucinating an answer. When valid, citations with exact
    timestamps are returned alongside multi-agent verification.

    Args:
        video_id: 11-character YouTube video ID.
        question: Natural language question regarding the video's content.
        k: Number of retrieved context chunks (default: 4).
        strategy: Chunking strategy ('fixed_token', 'sentence', 'semantic', 'time_window').
        hierarchical: Enable Two-Stage Hierarchical Video Retrieval for long lectures (>35 min).
        min_confidence: Retrieval confidence floor (default: 0.22).
        verify_citations_agent: Whether to run the second-pass citation verification agent.

    Returns:
        Dict containing answer, confidence verdict, cited timestamped sources, and verification report.
    """
    chunks = search_video_moments(
        video_id=video_id,
        query=question,
        k=k,
        strategy=strategy,
        hierarchical=hierarchical,
    )
    if not chunks:
        return {
            "answer": "I could not find relevant transcript segments for this video in the index.",
            "confidence": {"passed": False, "level": "none", "top_score": 0.0, "avg_score": 0.0},
            "sources": [],
            "verification": {"verdict": "unsupported", "reasoning": "No context chunks retrieved."},
        }

    conf = assess_retrieval_confidence(chunks, min_confidence=min_confidence)
    if not conf.get("passed", True):
        return {
            "answer": "I could not find sufficient information in the video to answer this question.",
            "confidence": conf,
            "sources": [],
            "verification": {"verdict": "supported", "reasoning": "Confidence gate rejected low-similarity context."},
        }

    try:
        qa_result = answer_question(
            question=question,
            chunks=chunks,
            min_retrieval_confidence=min_confidence,
        )
    except LLMError as err:
        return {
            "answer": f"LLM Generation Error: {err}",
            "confidence": conf,
            "sources": [],
            "verification": {"verdict": "unsupported", "reasoning": str(err)},
        }

    verification_info: dict[str, Any] = {}
    if verify_citations_agent and qa_result.get("sources"):
        verification_info = verify_answer(
            question=question,
            answer=qa_result.get("answer", ""),
            chunks=chunks,
        )

    return {
        "answer": qa_result.get("answer"),
        "confidence": conf,
        "sources": qa_result.get("sources", []),
        "model_used": qa_result.get("model_used"),
        "cached": qa_result.get("cached", False),
        "verification": verification_info,
    }


@mcp.tool()
def verify_citations(
    question: str,
    answer: str,
    chunk_texts: list[str],
) -> dict[str, Any]:
    """Verify whether an answer's factual statements are strictly supported by retrieved chunk texts.

    Runs an independent second-pass verification agent to classify evidence as supported,
    partially_supported, or unsupported.

    Args:
        question: The user query.
        answer: Candidate generated answer containing bracketed citations [1], [2], etc.
        chunk_texts: List of text passages corresponding to retrieved chunks.

    Returns:
        Dict with 'verdict' ('supported', 'partially_supported', 'unsupported'), 'cited_indices', and 'reasoning'.
    """
    dummy_chunks = [{"chunk_id": i + 1, "text": t, "start": 0.0, "end": 0.0} for i, t in enumerate(chunk_texts)]
    return verify_answer(question=question, answer=answer, chunks=dummy_chunks)


def main() -> None:
    """Run the MCP server over standard I/O (default for Claude Desktop / Cursor)."""
    mcp.run()


if __name__ == "__main__":
    main()
