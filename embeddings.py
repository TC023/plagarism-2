"""Embedding layer for Python code similarity.

This layer builds lightweight TF-IDF embeddings from normalized lexical tokens
and compares snippets using cosine similarity.
"""

from __future__ import annotations

from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from lexical_statistical_layer import normalize_tokens, preprocess_code, tokenize_code


def _normalized_token_text(code: str, preprocessed: bool = False) -> str:
    """Convert code into a normalized token string for vectorization."""
    clean_code = code if preprocessed else preprocess_code(code)
    tokens = tokenize_code(clean_code)
    normalized = normalize_tokens(tokens)
    return " ".join(normalized)


def analyze_embedding_similarity(code_a: str, code_b: str, preprocessed: bool = False) -> dict[str, Any]:
    """Analyze similarity using TF-IDF embeddings and cosine similarity."""
    text_a = _normalized_token_text(code_a, preprocessed=preprocessed)
    text_b = _normalized_token_text(code_b, preprocessed=preprocessed)

    if not text_a and not text_b:
        return {
            "embedding_model": "tfidf",
            "embedding_cosine_similarity": 1.0,
            "embedding_feature_count": 0,
            "embedding_score": 1.0,
        }

    if not text_a or not text_b:
        return {
            "embedding_model": "tfidf",
            "embedding_cosine_similarity": 0.0,
            "embedding_feature_count": 0,
            "embedding_score": 0.0,
        }

    vectorizer = TfidfVectorizer(lowercase=False, token_pattern=r"[^\s]+")
    matrix = vectorizer.fit_transform([text_a, text_b])
    score = float(cosine_similarity(matrix[0], matrix[1])[0][0])

    return {
        "embedding_model": "tfidf",
        "embedding_cosine_similarity": score,
        "embedding_feature_count": int(matrix.shape[1]),
        "embedding_score": score,
    }


def print_embedding_report(results: dict[str, Any]) -> None:
    """Print a compact embedding layer report."""
    print("=== Capa 4: Similitud por embeddings ===")
    print(f"- Modelo: {results['embedding_model']}")
    print(f"- Features: {results['embedding_feature_count']}")
    print(f"- Cosine similarity: {results['embedding_cosine_similarity']:.4f}")
    print(f"- Embedding score: {results['embedding_score']:.4f}")


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

    analysis = analyze_embedding_similarity(code_a, code_b)
    print_embedding_report(analysis)