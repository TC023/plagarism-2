"""Semantic layer for Python code similarity.

Layer 3 is embeddings-only. This module wraps the embedding analysis so callers
can keep importing semantic-layer functions without behavioral execution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from embeddings import DEFAULT_EMBEDDING_CACHE_PATH, analyze_embedding_similarity


def analyze_semantic_similarity(
    code_a: str,
    code_b: str,
    preprocessed: bool = False,
    openai_model: Optional[str] = None,
    cache_path: Path = DEFAULT_EMBEDDING_CACHE_PATH,
) -> dict[str, Any]:
    """Compute semantic similarity using embeddings only."""
    embedding_results = analyze_embedding_similarity(
        code_a,
        code_b,
        preprocessed=preprocessed,
        openai_model=openai_model,
        cache_path=cache_path,
    )
    score = float(embedding_results.get("embedding_score", 0.0))

    return {
        "semantic_mode": "embeddings",
        "semantic_runnable": "si",
        "semantic_reason": "ok",
        "embedding_model": embedding_results.get("embedding_model", "unknown"),
        "embedding_cosine_similarity": float(embedding_results.get("embedding_cosine_similarity", 0.0)),
        "embedding_feature_count": int(embedding_results.get("embedding_feature_count", 0)),
        "semantic_score": score,
    }


def print_semantic_report(results: dict[str, Any]) -> None:
    """Print a compact semantic report for embeddings-only layer."""
    print("=== Capa 3: Similitud semantica por embeddings ===")
    print(f"- Mode: {results.get('semantic_mode', 'embeddings')}")
    print(f"- Modelo: {results.get('embedding_model', 'unknown')}")
    print(f"- Features: {results.get('embedding_feature_count', 0)}")
    print(f"- Cosine similarity: {float(results.get('embedding_cosine_similarity', 0.0)):.4f}")
    print(f"- Semantic score: {float(results.get('semantic_score', 0.0)):.4f}")


if __name__ == "__main__":
    code_a = """
def sum_even_numbers(values):
    total = 0
    for number in values:
        if number % 2 == 0:
            total += number
    return total
"""

    code_b = """
def add_pairs(items):
    result = 0
    for item in items:
        if item % 2 == 0:
            result = result + item
    return result
"""

    analysis = analyze_semantic_similarity(code_a, code_b)
    print_semantic_report(analysis)
