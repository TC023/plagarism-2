"""Structural layer for Python code similarity based on AST features.

This module implements a lightweight Layer 2. It uses Python's built-in AST to
compare structural patterns, including a simple ordered Tree Edit Distance.
"""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Any

from lexical_statistical_layer import preprocess_code

try:
    from apted import APTED, Config
except ImportError:
    APTED = None
    Config = object


IGNORED_AST_NODE_TYPES = {"Load", "Store", "Del"}


@dataclass(frozen=True)
class ComparableASTNode:
    """Small immutable tree node used for structural edit distance."""

    label: str
    children: tuple[ComparableASTNode, ...] = ()


if APTED is not None:

    class ASTAptedConfig(Config):
        """APTED edit costs for compact AST nodes."""

        def children(self, node: ComparableASTNode) -> tuple[ComparableASTNode, ...]:
            """Return ordered children for APTED."""
            return node.children

        def rename(self, node_a: ComparableASTNode, node_b: ComparableASTNode) -> int:
            """Charge one edit when AST node labels differ."""
            return 0 if node_a.label == node_b.label else 1

        def insert(self, node: ComparableASTNode) -> int:
            """Charge one edit for inserting a node."""
            return 1

        def delete(self, node: ComparableASTNode) -> int:
            """Charge one edit for deleting a node."""
            return 1

else:
    ASTAptedConfig = None


def parse_python_ast(code: str, preprocessed: bool = False) -> ast.AST:
    """Parse raw or preprocessed Python code into an AST."""
    if not isinstance(code, str):
        raise TypeError("code must be a string")

    clean_code = code if preprocessed else preprocess_code(code)
    if not clean_code:
        return ast.Module(body=[], type_ignores=[])

    try:
        return ast.parse(clean_code)
    except SyntaxError as exc:
        raise ValueError(f"Could not parse Python code: {exc}") from exc


def ast_node_type_sequence(tree: ast.AST) -> list[str]:
    """Return a preorder-like sequence of AST node type names."""
    sequence: list[str] = []

    def visit(node: ast.AST) -> None:
        node_type = type(node).__name__
        if node_type not in IGNORED_AST_NODE_TYPES:
            sequence.append(node_type)
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    return sequence


def ast_node_distribution(node_types: list[str]) -> dict[str, float]:
    """Convert AST node types into a probability distribution."""
    if not node_types:
        return {}

    counts = Counter(node_types)
    total = len(node_types)
    return {node_type: count / total for node_type, count in counts.items()}


def ast_jaccard_similarity(nodes_a: list[str], nodes_b: list[str]) -> float:
    """Compute Jaccard similarity over unique AST node types."""
    set_a = set(nodes_a)
    set_b = set(nodes_b)

    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0

    return len(set_a & set_b) / len(set_a | set_b)


def ast_sequence_similarity(nodes_a: list[str], nodes_b: list[str]) -> float:
    """Compare AST preorder node sequences using SequenceMatcher."""
    if not nodes_a and not nodes_b:
        return 1.0
    if not nodes_a or not nodes_b:
        return 0.0

    return SequenceMatcher(None, nodes_a, nodes_b).ratio()


def ast_depth(tree: ast.AST) -> int:
    """Return the maximum depth of an AST."""
    children = list(ast.iter_child_nodes(tree))
    if not children:
        return 1

    return 1 + max(ast_depth(child) for child in children)


def numeric_similarity(value_a: int, value_b: int) -> float:
    """Compare two non-negative numeric AST properties as a 0-1 similarity."""
    if value_a < 0 or value_b < 0:
        raise ValueError("numeric values must be non-negative")
    if value_a == 0 and value_b == 0:
        return 1.0

    denominator = max(value_a, value_b)
    if denominator == 0:
        return 0.0

    return 1.0 - (abs(value_a - value_b) / denominator)


def ast_to_comparable_tree(tree: ast.AST) -> ComparableASTNode:
    """Convert Python's AST into a compact ordered tree for comparison."""

    def convert(node: ast.AST) -> ComparableASTNode | None:
        node_type = type(node).__name__
        converted_children = tuple(
            child_tree
            for child in ast.iter_child_nodes(node)
            if (child_tree := convert(child)) is not None
        )

        if node_type in IGNORED_AST_NODE_TYPES:
            return None

        return ComparableASTNode(node_type, converted_children)

    comparable_tree = convert(tree)
    if comparable_tree is None:
        return ComparableASTNode("Empty")

    return comparable_tree


def comparable_tree_size(node: ComparableASTNode) -> int:
    """Count nodes in a comparable AST tree."""
    return 1 + sum(comparable_tree_size(child) for child in node.children)


def tree_edit_distance(tree_a: ComparableASTNode, tree_b: ComparableASTNode) -> int:
    """Compute ordered tree edit distance with unit relabel cost.

    Insertions and deletions are charged by subtree size. Substitution compares
    node labels and recursively aligns ordered child forests.
    """

    @lru_cache(maxsize=None)
    def subtree_size(node: ComparableASTNode) -> int:
        return 1 + sum(subtree_size(child) for child in node.children)

    @lru_cache(maxsize=None)
    def distance(node_a: ComparableASTNode, node_b: ComparableASTNode) -> int:
        relabel_cost = 0 if node_a.label == node_b.label else 1
        return relabel_cost + forest_distance(node_a.children, node_b.children)

    @lru_cache(maxsize=None)
    def forest_distance(
        forest_a: tuple[ComparableASTNode, ...],
        forest_b: tuple[ComparableASTNode, ...],
    ) -> int:
        rows = len(forest_a) + 1
        columns = len(forest_b) + 1
        dp = [[0 for _ in range(columns)] for _ in range(rows)]

        for row in range(1, rows):
            dp[row][0] = dp[row - 1][0] + subtree_size(forest_a[row - 1])
        for column in range(1, columns):
            dp[0][column] = dp[0][column - 1] + subtree_size(forest_b[column - 1])

        for row in range(1, rows):
            for column in range(1, columns):
                delete_cost = dp[row - 1][column] + subtree_size(forest_a[row - 1])
                insert_cost = dp[row][column - 1] + subtree_size(forest_b[column - 1])
                substitute_cost = dp[row - 1][column - 1] + distance(
                    forest_a[row - 1],
                    forest_b[column - 1],
                )
                dp[row][column] = min(delete_cost, insert_cost, substitute_cost)

        return dp[-1][-1]

    return distance(tree_a, tree_b)


def tree_edit_similarity(tree_a: ComparableASTNode, tree_b: ComparableASTNode) -> float:
    """Normalize Tree Edit Distance as a 0-1 similarity score."""
    max_size = max(comparable_tree_size(tree_a), comparable_tree_size(tree_b))
    if max_size == 0:
        return 1.0

    distance = tree_edit_distance(tree_a, tree_b)
    similarity = 1.0 - (distance / max_size)
    return max(0.0, min(1.0, similarity))


def apted_tree_edit_distance(tree_a: ComparableASTNode, tree_b: ComparableASTNode) -> int | None:
    """Compute APTED distance when the optional apted package is installed."""
    if APTED is None or ASTAptedConfig is None:
        return None

    return int(APTED(tree_a, tree_b, ASTAptedConfig()).compute_edit_distance())


def apted_tree_edit_similarity(tree_a: ComparableASTNode, tree_b: ComparableASTNode) -> float | None:
    """Normalize APTED distance as a 0-1 similarity score."""
    distance = apted_tree_edit_distance(tree_a, tree_b)
    if distance is None:
        return None

    max_size = max(comparable_tree_size(tree_a), comparable_tree_size(tree_b))
    if max_size == 0:
        return 1.0

    similarity = 1.0 - (distance / max_size)
    return max(0.0, min(1.0, similarity))


def analyze_structural_similarity(
    code_a: str,
    code_b: str,
    preprocessed: bool = False,
) -> dict[str, Any]:
    """Run Layer-2 AST-based structural similarity metrics."""
    tree_a = parse_python_ast(code_a, preprocessed=preprocessed)
    tree_b = parse_python_ast(code_b, preprocessed=preprocessed)
    comparable_tree_a = ast_to_comparable_tree(tree_a)
    comparable_tree_b = ast_to_comparable_tree(tree_b)

    node_types_a = ast_node_type_sequence(tree_a)
    node_types_b = ast_node_type_sequence(tree_b)

    node_count_a = len(node_types_a)
    node_count_b = len(node_types_b)
    depth_a = ast_depth(tree_a)
    depth_b = ast_depth(tree_b)

    node_type_jaccard = ast_jaccard_similarity(node_types_a, node_types_b)
    sequence_similarity = ast_sequence_similarity(node_types_a, node_types_b)
    node_count_similarity = numeric_similarity(node_count_a, node_count_b)
    depth_similarity = numeric_similarity(depth_a, depth_b)
    edit_distance = tree_edit_distance(comparable_tree_a, comparable_tree_b)
    edit_similarity = tree_edit_similarity(comparable_tree_a, comparable_tree_b)
    apted_distance = apted_tree_edit_distance(comparable_tree_a, comparable_tree_b)
    apted_similarity = apted_tree_edit_similarity(comparable_tree_a, comparable_tree_b)
    apted_similarity_for_score = apted_similarity if apted_similarity is not None else edit_similarity

    structural_score = (
        0.30 * apted_similarity_for_score
        + 0.25 * sequence_similarity
        + 0.20 * node_type_jaccard
        + 0.15 * node_count_similarity
        + 0.10 * depth_similarity
    )

    return {
        "ast_node_types_a": node_types_a,
        "ast_node_types_b": node_types_b,
        "ast_node_distribution_a": ast_node_distribution(node_types_a),
        "ast_node_distribution_b": ast_node_distribution(node_types_b),
        "ast_node_count_a": node_count_a,
        "ast_node_count_b": node_count_b,
        "ast_depth_a": depth_a,
        "ast_depth_b": depth_b,
        "ast_node_type_jaccard": node_type_jaccard,
        "ast_sequence_similarity": sequence_similarity,
        "ast_node_count_similarity": node_count_similarity,
        "ast_depth_similarity": depth_similarity,
        "tree_edit_distance": edit_distance,
        "tree_edit_similarity": edit_similarity,
        "apted_available": "si" if apted_distance is not None else "no",
        "apted_tree_edit_distance": apted_distance if apted_distance is not None else "",
        "apted_tree_edit_similarity": apted_similarity if apted_similarity is not None else "",
        "structural_score": structural_score,
    }


def print_structural_report(results: dict[str, Any]) -> None:
    """Print a compact report for Layer-2 structural metrics."""
    print("=== Capa 2: Similitud estructural basada en AST ===")
    print(f"- AST node type Jaccard: {results['ast_node_type_jaccard']:.4f}")
    print(f"- AST sequence similarity: {results['ast_sequence_similarity']:.4f}")
    print(f"- AST node count similarity: {results['ast_node_count_similarity']:.4f}")
    print(f"- AST depth similarity: {results['ast_depth_similarity']:.4f}")
    print(f"- Tree Edit Distance: {results['tree_edit_distance']}")
    print(f"- Tree Edit similarity: {results['tree_edit_similarity']:.4f}")
    print(f"- APTED available: {results['apted_available']}")
    if results["apted_available"] == "si":
        print(f"- APTED Tree Edit Distance: {results['apted_tree_edit_distance']}")
        print(f"- APTED Tree Edit similarity: {results['apted_tree_edit_similarity']:.4f}")
    print(f"- Structural score: {results['structural_score']:.4f}")


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

    analysis = analyze_structural_similarity(code_a, code_b)
    print_structural_report(analysis)
