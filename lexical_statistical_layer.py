"""Prototype lexical-statistical layer for Python code similarity.

This module implements only the first stage of a layered plagiarism-detection
architecture: basic preprocessing plus lexical/statistical similarity metrics.
It intentionally does not include AST comparison, embeddings, semantic models,
neural networks, or a final trained classifier.
"""

from __future__ import annotations

import ast
import io
import keyword
import math
import re
import tokenize
from collections import Counter, defaultdict
from typing import Any


TokenDistribution = dict[str, float]
TransitionMatrix = dict[str, dict[str, float]]


IGNORED_TOKEN_TYPES = {
    tokenize.ENCODING,
    tokenize.ENDMARKER,
    tokenize.NEWLINE,
    tokenize.NL,
    tokenize.INDENT,
    tokenize.DEDENT,
    tokenize.COMMENT,
}

PYTHON_KEYWORDS = set(keyword.kwlist)
BOOLEAN_LITERALS = {"True", "False", "None"}

CTRL_TOKENS = {
    "if",
    "elif",
    "else",
    "for",
    "while",
    "try",
    "except",
    "finally",
    "with",
    "match",
    "case",
}
FUNC_TOKENS = {"def", "return", "lambda", "yield", "print", "input"}
OP_TOKENS = {
    "+",
    "-",
    "*",
    "/",
    "//",
    "%",
    "**",
    "=",
    "+=",
    "-=",
    "*=",
    "/=",
    "//=",
    "%=",
    "**=",
    ":=",
}
LOGIC_TOKENS = {"and", "or", "not", "==", "!=", "<", ">", "<=", ">=", "is", "in"}
PUNCTUATION_TOKENS = {"(", ")", "[", "]", "{", "}", ",", ":", ".", ";"}
NORMALIZABLE_BUILTINS = {"print", "input"}


def _docstring_ranges(code: str) -> set[tuple[int, int]]:
    """Return line/column spans for module, class, and function docstrings."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()

    ranges: set[tuple[int, int]] = set()
    nodes: list[ast.AST] = [tree]
    nodes.extend(node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)))

    for node in nodes:
        body = getattr(node, "body", [])
        if not body:
            continue
        first_statement = body[0]
        if not (
            isinstance(first_statement, ast.Expr)
            and isinstance(first_statement.value, ast.Constant)
            and isinstance(first_statement.value.value, str)
        ):
            continue

        start_line = getattr(first_statement, "lineno", None)
        end_line = getattr(first_statement, "end_lineno", None)
        if start_line is not None and end_line is not None:
            ranges.add((start_line, end_line))

    return ranges


def _is_string_literal(token: str) -> bool:
    """Check whether a token looks like a Python string literal."""
    return bool(re.fullmatch(r"(?is)(?:[rubf]*)('''.*'''|\"\"\".*\"\"\"|'.*'|\".*\")", token))


def _is_number_literal(token: str) -> bool:
    """Check whether a token looks like a Python numeric literal."""
    return bool(
        re.fullmatch(
            r"(?ix)"
            r"(?:"
            r"0[bB][01_]+|"
            r"0[oO][0-7_]+|"
            r"0[xX][0-9a-f_]+|"
            r"(?:\d[\d_]*\.?[\d_]*|\.\d[\d_]*)(?:[eE][+-]?\d[\d_]*)?"
            r")j?",
            token,
        )
    )


def _clean_whitespace(code: str) -> str:
    """Remove blank lines and trailing spaces while preserving indentation."""
    clean_lines = [line.rstrip() for line in code.splitlines() if line.strip()]
    return "\n".join(clean_lines).strip()


def preprocess_code(code: str) -> str:
    """Remove comments/docstrings and normalize whitespace while preserving indentation."""
    if not isinstance(code, str):
        raise TypeError("code must be a string")

    docstring_ranges = _docstring_ranges(code)
    lines = code.splitlines()

    # Find the column where the comment starts on each line via the tokenizer,
    # so a '#' inside a string literal is never mistaken for a comment.
    comment_cols: dict[int, int] = {}
    tokenized = True
    try:
        stream = io.StringIO(code).readline
        for token_info in tokenize.generate_tokens(stream):
            if token_info.type == tokenize.COMMENT:
                comment_cols[token_info.start[0]] = token_info.start[1]
    except (tokenize.TokenError, SyntaxError):
        tokenized = False

    cleaned_lines = []
    for i, line in enumerate(lines, start=1):
        # Skip docstring lines
        if any(start <= i <= end for start, end in docstring_ranges):
            continue

        # Remove comments
        if i in comment_cols:
            line = line[: comment_cols[i]]
        elif not tokenized and "#" in line:
            line = line.split("#", 1)[0]

        cleaned_lines.append(line.rstrip())

    return "\n".join(cleaned_lines).strip()


def tokenize_code(code: str) -> list[str]:
    """Tokenize Python code and return relevant lexical tokens."""
    if not isinstance(code, str):
        raise TypeError("code must be a string")

    tokens: list[str] = []
    try:
        stream = io.StringIO(code).readline
        for token_info in tokenize.generate_tokens(stream):
            if token_info.type in IGNORED_TOKEN_TYPES:
                continue
            tokens.append(token_info.string)
    except tokenize.TokenError as exc:
        raise ValueError(f"Could not tokenize code: {exc}") from exc

    return tokens


def normalize_tokens(tokens: list[str]) -> list[str]:
    """Normalize identifiers and literals while preserving key syntax tokens."""
    normalized: list[str] = []

    for token in tokens:
        if token in BOOLEAN_LITERALS or _is_number_literal(token) or _is_string_literal(token):
            normalized.append("LITERAL")
        elif token in PYTHON_KEYWORDS or token in NORMALIZABLE_BUILTINS or token in OP_TOKENS or token in LOGIC_TOKENS:
            normalized.append(token)
        elif token.isidentifier():
            normalized.append("ID")
        else:
            normalized.append(token)

    return normalized


def jaccard_similarity(tokens_a: list[str], tokens_b: list[str]) -> float:
    """Compute Jaccard similarity over unique tokens."""
    set_a = set(tokens_a)
    set_b = set(tokens_b)

    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0

    return len(set_a & set_b) / len(set_a | set_b)


def tfidf_cosine_similarity(tokens_a: list[str], tokens_b: list[str]) -> float:
    """Compute cosine similarity over a small TF-IDF representation."""
    documents = [tokens_a, tokens_b]
    vocabulary = sorted(set(tokens_a) | set(tokens_b))

    if not vocabulary:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0

    document_frequency = {
        token: sum(1 for document in documents if token in document)
        for token in vocabulary
    }

    vectors: list[list[float]] = []
    for document in documents:
        counts = Counter(document)
        total = len(document)
        vector: list[float] = []

        for token in vocabulary:
            term_frequency = counts[token] / total
            inverse_document_frequency = math.log((1 + len(documents)) / (1 + document_frequency[token])) + 1
            vector.append(term_frequency * inverse_document_frequency)

        vectors.append(vector)

    dot_product = sum(value_a * value_b for value_a, value_b in zip(vectors[0], vectors[1]))
    norm_a = math.sqrt(sum(value * value for value in vectors[0]))
    norm_b = math.sqrt(sum(value * value for value in vectors[1]))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def token_distribution(tokens: list[str]) -> TokenDistribution:
    """Convert a token sequence into a probability distribution."""
    if not tokens:
        return {}

    counts = Counter(tokens)
    total = len(tokens)
    return {token: count / total for token, count in counts.items()}


def shannon_entropy(distribution: TokenDistribution) -> float:
    """Measure uncertainty/diversity in a token distribution."""
    return -sum(probability * math.log2(probability) for probability in distribution.values() if probability > 0)


def kl_divergence(dist_p: TokenDistribution, dist_q: TokenDistribution, epsilon: float = 1e-10) -> float:
    """Compute KL divergence D_KL(P || Q) with epsilon smoothing."""
    if epsilon <= 0:
        raise ValueError("epsilon must be greater than zero")

    vocabulary = set(dist_p) | set(dist_q)
    if not vocabulary:
        return 0.0

    smoothed_p = {token: dist_p.get(token, 0.0) + epsilon for token in vocabulary}
    smoothed_q = {token: dist_q.get(token, 0.0) + epsilon for token in vocabulary}

    total_p = sum(smoothed_p.values())
    total_q = sum(smoothed_q.values())

    return sum(
        (smoothed_p[token] / total_p) * math.log((smoothed_p[token] / total_p) / (smoothed_q[token] / total_q))
        for token in vocabulary
    )


def categorize_tokens(tokens: list[str]) -> list[str]:
    """Map normalized tokens into broad lexical categories."""
    categories: list[str] = []

    for token in tokens:
        if token in CTRL_TOKENS:
            categories.append("CTRL")
        elif token in FUNC_TOKENS:
            categories.append("FUNC")
        elif token in OP_TOKENS:
            categories.append("OP")
        elif token in LOGIC_TOKENS:
            categories.append("LOGIC")
        elif token == "LITERAL":
            categories.append("LITERAL")
        elif token == "ID":
            categories.append("ID")
        else:
            categories.append("OTHER")

    return categories


def markov_transition_matrix(states: list[str]) -> TransitionMatrix:
    """Build a probabilistic transition matrix between token categories."""
    if len(states) < 2:
        return {}

    transition_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for current_state, next_state in zip(states, states[1:]):
        transition_counts[current_state][next_state] += 1

    matrix: TransitionMatrix = {}
    for current_state, next_counts in transition_counts.items():
        total = sum(next_counts.values())
        matrix[current_state] = {next_state: count / total for next_state, count in next_counts.items()}

    return matrix


def compare_markov_matrices(matrix_a: TransitionMatrix, matrix_b: TransitionMatrix) -> float:
    """Compare transition matrices using average absolute distance."""
    states = set(matrix_a) | set(matrix_b)
    next_states = {next_state for matrix in (matrix_a, matrix_b) for row in matrix.values() for next_state in row}
    all_states = states | next_states

    if not all_states:
        return 1.0

    differences: list[float] = []
    for current_state in all_states:
        for next_state in all_states:
            probability_a = matrix_a.get(current_state, {}).get(next_state, 0.0)
            probability_b = matrix_b.get(current_state, {}).get(next_state, 0.0)
            differences.append(abs(probability_a - probability_b))

    average_distance = sum(differences) / len(differences)
    return max(0.0, min(1.0, 1.0 - average_distance))


def analyze_lexical_statistical_similarity(
    code_a: str,
    code_b: str,
    preprocessed: bool = False,
) -> dict[str, Any]:
    """Run Layer-1 lexical/statistical metrics over raw or clean code."""
    clean_code_a = code_a if preprocessed else preprocess_code(code_a)
    clean_code_b = code_b if preprocessed else preprocess_code(code_b)

    tokens_a = tokenize_code(clean_code_a)
    tokens_b = tokenize_code(clean_code_b)

    normalized_tokens_a = normalize_tokens(tokens_a)
    normalized_tokens_b = normalize_tokens(tokens_b)

    jaccard = jaccard_similarity(normalized_tokens_a, normalized_tokens_b)
    cosine = tfidf_cosine_similarity(normalized_tokens_a, normalized_tokens_b)

    distribution_a = token_distribution(normalized_tokens_a)
    distribution_b = token_distribution(normalized_tokens_b)
    entropy_a = shannon_entropy(distribution_a)
    entropy_b = shannon_entropy(distribution_b)

    kl_a_to_b = kl_divergence(distribution_a, distribution_b)
    kl_b_to_a = kl_divergence(distribution_b, distribution_a)
    symmetric_kl = (kl_a_to_b + kl_b_to_a) / 2
    kl_similarity = 1 / (1 + symmetric_kl)

    entropy_difference = abs(entropy_a - entropy_b)

    categories_a = categorize_tokens(normalized_tokens_a)
    categories_b = categorize_tokens(normalized_tokens_b)
    matrix_a = markov_transition_matrix(categories_a)
    matrix_b = markov_transition_matrix(categories_b)
    markov_similarity = compare_markov_matrices(matrix_a, matrix_b)

    lexical_statistical_score = (
        0.30 * jaccard
        + 0.30 * cosine
        + 0.20 * markov_similarity
        + 0.20 * kl_similarity
    )

    return {
        "tokens_a": tokens_a,
        "tokens_b": tokens_b,
        "normalized_tokens_a": normalized_tokens_a,
        "normalized_tokens_b": normalized_tokens_b,
        "jaccard_similarity": jaccard,
        "tfidf_cosine_similarity": cosine,
        "entropy_a": entropy_a,
        "entropy_b": entropy_b,
        "kl_similarity": kl_similarity,
        "kl_divergence_a_to_b": kl_a_to_b,
        "kl_divergence_b_to_a": kl_b_to_a,
        "markov_similarity": markov_similarity,
        "entropy_difference": entropy_difference,
        "lexical_statistical_score": lexical_statistical_score,
    }


def print_analysis_report(results: dict[str, Any]) -> None:
    """Print a clear report for the example execution."""
    metric_descriptions = {
        "jaccard_similarity": "solapamiento de tokens normalizados únicos",
        "tfidf_cosine_similarity": "similitud de frecuencia ponderada de tokens",
        "entropy_a": "diversidad/incertidumbre de tokens en Código A",
        "entropy_b": "diversidad/incertidumbre de tokens en Código B",
        "kl_divergence_a_to_b": "diferencia de distribución de A respecto a B",
        "kl_divergence_b_to_a": "diferencia de distribución de B respecto a A",
        "markov_similarity": "similitud del patrón de transiciones entre categorías",
        "lexical_statistical_score": "fusión ponderada inicial de la Capa 1",
    }

    print("=== Capa 1: Similitud lexico-estadistica ===")
    print("\nTokens normalizados A:")
    print(results["normalized_tokens_a"])
    print("\nTokens normalizados B:")
    print(results["normalized_tokens_b"])

    print("\nMetricas:")
    for metric, description in metric_descriptions.items():
        value = results[metric]
        print(f"- {metric}: {value:.4f}  -> {description}")

    print("\nInterpretacion:")
    print("- Valores cercanos a 1 en similitudes sugieren mayor parecido lexico-estadistico.")
    print("- KL cercano a 0 indica distribuciones de tokens mas parecidas.")
    print("- Este score no es una probabilidad final de plagio; solo representa la Capa 1.")


if __name__ == "__main__":
    code_a = '''
def sum_even_numbers(values):
    """Return the sum of even numbers in a list."""
    total = 0
    for number in values:
        if number % 2 == 0:
            total += number
    return total
'''

    code_b = '''
def add_pairs(items):
    # Accumulate values divisible by two.
    result = 0
    for item in items:
        if item % 2 == 0:
            result = result + item
    return result
'''

    analysis = analyze_lexical_statistical_similarity(code_a, code_b)
    print_analysis_report(analysis)
