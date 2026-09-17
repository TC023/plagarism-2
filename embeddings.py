"""Embedding layer for Python code similarity.

This layer builds lightweight TF-IDF embeddings from normalized lexical tokens
and compares snippets using cosine similarity.
"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from lexical_statistical_layer import normalize_tokens, preprocess_code, tokenize_code

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None
    _OPENAI_AVAILABLE = False


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_EMBEDDING_CACHE_PATH = PROJECT_ROOT / "results" / "openai_embeddings_cache.json"


def _normalized_token_text(code: str, preprocessed: bool = False) -> str:
    """Return clean code for embedding. OpenAI embeddings handle tokenization internally."""
    return code if preprocessed else preprocess_code(code)


def _cache_key(model: str, text: str) -> str:
    """Build a stable key for one embedding input."""
    digest = sha256(text.encode("utf-8")).hexdigest()
    return f"{model}:{digest}"


def load_embedding_cache(cache_path: Path = DEFAULT_EMBEDDING_CACHE_PATH) -> dict[str, list[float]]:
    """Load cached embeddings from disk."""
    if not cache_path.exists():
        print(f"[embeddings] Cache file not found: {cache_path}")
        return {}

    try:
        with cache_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError):
        print(f"[embeddings] Cache file unreadable/corrupt, starting empty: {cache_path}")
        return {}

    if not isinstance(data, dict):
        print(f"[embeddings] Cache file has invalid format, starting empty: {cache_path}")
        return {}

    cache: dict[str, list[float]] = {}
    for key, value in data.items():
        if isinstance(key, str) and isinstance(value, list):
            cache[key] = [float(item) for item in value]
    print(f"[embeddings] Loaded {len(cache)} cached embeddings from disk: {cache_path}")
    return cache


def save_embedding_cache(cache: dict[str, list[float]], cache_path: Path = DEFAULT_EMBEDDING_CACHE_PATH) -> None:
    """Persist cached embeddings to disk."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as file:
        json.dump(cache, file, ensure_ascii=False, indent=2, sort_keys=True)


def _openai_embeddings(
    texts: list[str],
    model: str = "text-embedding-3-small",
    cache_path: Path = DEFAULT_EMBEDDING_CACHE_PATH,
) -> list[list[float]]:
    """Request embeddings from OpenAI for a list of texts.

    Requires `OPENAI_API_KEY` in the environment.
    """
    cache = load_embedding_cache(cache_path)
    embeddings: list[list[float] | None] = [None] * len(texts)
    missing_texts: list[str] = []
    missing_indices: list[int] = []

    for index, text in enumerate(texts):
        cached_embedding = cache.get(_cache_key(model, text))
        if cached_embedding is not None:
            embeddings[index] = cached_embedding
            continue

        missing_texts.append(text)
        missing_indices.append(index)

    cache_hits = len(texts) - len(missing_texts)
    print(
        f"[embeddings] Cache status -> hits: {cache_hits}, misses: {len(missing_texts)}, model: {model}"
    )

    if missing_texts:
        if not _OPENAI_AVAILABLE:
            raise RuntimeError("openai package is not installed; install with `pip install openai`")

        print(f"[embeddings] Requesting {len(missing_texts)} embeddings from OpenAI API")
        client = OpenAI()
        response = client.embeddings.create(model=model, input=missing_texts)

        for index, item in zip(missing_indices, response.data):
            embedding = [float(value) for value in item.embedding]
            embeddings[index] = embedding
            cache[_cache_key(model, texts[index])] = embedding

        save_embedding_cache(cache, cache_path)
        print(f"[embeddings] Saved updated cache to disk: {cache_path}")

    return [embedding for embedding in embeddings if embedding is not None]


def analyze_embedding_similarity(
    code_a: str,
    code_b: str,
    preprocessed: bool = False,
    openai_model: Optional[str] = None,
    cache_path: Path = DEFAULT_EMBEDDING_CACHE_PATH,
) -> dict[str, Any]:
    """Analyze similarity using either TF-IDF or OpenAI embeddings.

    - `method` can be "tfidf" (default) or "openai".
    - `openai_model` selects the OpenAI embedding model (defaults to `text-embedding-3-small`).
    """
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

    model_name = openai_model or "text-embedding-3-small"
    embeddings = _openai_embeddings([text_a, text_b], model=model_name, cache_path=cache_path)
    a = np.array(embeddings[0], dtype=float)
    b = np.array(embeddings[1], dtype=float)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    score = float(np.dot(a, b) / denom) if denom != 0 else 0.0

    return {
        "embedding_model": model_name,
        "embedding_cosine_similarity": score,
        "embedding_feature_count": int(a.shape[0]),
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