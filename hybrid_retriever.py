"""Hybrid Dense-Lexical (BM25 + Bi-Encoder) Video Retriever with Reciprocal Rank Fusion (RRF).

Combines dense semantic embeddings (all-MiniLM-L6-v2) with sparse lexical BM25 matching
to prevent secondary topic collisions and capture precise mathematical variable names.
"""

from __future__ import annotations

import math
import re
from typing import Any
from collections import Counter


class BM25Okapi:
    """Self-contained, zero-dependency BM25Okapi scoring implementation."""

    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_lens = []
        self.doc_freqs: dict[str, int] = Counter()
        self.tokenized_corpus: list[list[str]] = []

        for doc in corpus:
            tokens = self._tokenize(doc)
            self.tokenized_corpus.append(tokens)
            self.doc_lens.append(len(tokens))
            unique_tokens = set(tokens)
            for t in unique_tokens:
                self.doc_freqs[t] += 1

        self.avg_doc_len = sum(self.doc_lens) / max(1, self.corpus_size)
        self.idf: dict[str, float] = {}
        for token, freq in self.doc_freqs.items():
            # Standard Lucene/BM25 IDF with smoothing
            self.idf[token] = math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple alphanumeric lowercase tokenizer."""
        return re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower())

    def get_scores(self, query: str) -> list[float]:
        """Compute BM25 scores for all corpus documents against query."""
        q_tokens = self._tokenize(query)
        scores = [0.0] * self.corpus_size

        for token in q_tokens:
            if token not in self.idf:
                continue
            token_idf = self.idf[token]
            for doc_idx, doc_tokens in enumerate(self.tokenized_corpus):
                tf = doc_tokens.count(token)
                if tf == 0:
                    continue
                doc_len = self.doc_lens[doc_idx]
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                scores[doc_idx] += token_idf * (numerator / denominator)

        return scores


def reciprocal_rank_fusion(
    dense_ranked_indices: list[int],
    lexical_ranked_indices: list[int],
    k: int = 60,
) -> list[tuple[int, float]]:
    """Compute Reciprocal Rank Fusion (RRF) scores across two ranked candidate lists."""
    rrf_scores: dict[int, float] = {}

    for rank, doc_idx in enumerate(dense_ranked_indices, start=1):
        rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (1.0 / (k + rank))

    for rank, doc_idx in enumerate(lexical_ranked_indices, start=1):
        rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (1.0 / (k + rank))

    sorted_fused = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_fused


def search_hybrid(
    query: str,
    chunks: list[dict[str, Any]],
    top_k: int = 4,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """Execute hybrid retrieval combining dense scores (if precomputed) and BM25."""
    if not chunks:
        return []

    # 1. Lexical BM25 Ranking
    corpus_texts = [c.get("text", "") for c in chunks]
    bm25 = BM25Okapi(corpus_texts)
    bm25_scores = bm25.get_scores(query)
    lexical_ranking = sorted(range(len(chunks)), key=lambda i: bm25_scores[i], reverse=True)

    # 2. Dense Ranking (from chunk similarity scores if present, else fallback)
    dense_scores = [c.get("score", 0.0) for c in chunks]
    dense_ranking = sorted(range(len(chunks)), key=lambda i: dense_scores[i], reverse=True)

    # 3. Fuse with RRF
    fused = reciprocal_rank_fusion(dense_ranking, lexical_ranking, k=rrf_k)

    results = []
    for doc_idx, fused_score in fused[:top_k]:
        chunk_copy = dict(chunks[doc_idx])
        chunk_copy["hybrid_rrf_score"] = round(fused_score, 4)
        chunk_copy["bm25_score"] = round(bm25_scores[doc_idx], 4)
        results.append(chunk_copy)

    return results


if __name__ == "__main__":
    sample_chunks = [
        {"chunk_id": "c1", "text": "In this lecture we introduce gradient descent and learning rate parameter alpha.", "score": 0.45},
        {"chunk_id": "c2", "text": "Here is the formal proof of batch gradient descent parameter update rule with theta.", "score": 0.42},
        {"chunk_id": "c3", "text": "Recapping our discussion on regularized linear regression and cost functions.", "score": 0.38},
    ]
    query = "batch gradient descent parameter update rule theta"
    res = search_hybrid(query, sample_chunks, top_k=2)
    print("Hybrid Search Results:")
    for r in res:
        print(f"  [{r['chunk_id']}] RRF Score: {r['hybrid_rrf_score']} | BM25: {r['bm25_score']} | {r['text'][:60]}...")
