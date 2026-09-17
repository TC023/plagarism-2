"""AST visualization helpers for Python source code.

The visualizations focus on structure by default: node types are shown, while
identifier names and literal values are hidden unless include_values=True.
"""

from __future__ import annotations

import ast
from pathlib import Path

from structural_layer import IGNORED_AST_NODE_TYPES, parse_python_ast


DEFAULT_OUTPUT_DIR = Path("results") / "ast_visualizations"


def _node_label(node: ast.AST, include_values: bool = False) -> str:
    """Build a compact label for one AST node."""
    node_type = type(node).__name__
    if not include_values:
        return node_type

    if isinstance(node, ast.FunctionDef):
        return f"FunctionDef({node.name})"
    if isinstance(node, ast.AsyncFunctionDef):
        return f"AsyncFunctionDef({node.name})"
    if isinstance(node, ast.ClassDef):
        return f"ClassDef({node.name})"
    if isinstance(node, ast.arg):
        return f"arg({node.arg})"
    if isinstance(node, ast.Name):
        return f"Name({node.id})"
    if isinstance(node, ast.Constant):
        return f"Constant({node.value!r})"

    return node_type


def _visible_children(node: ast.AST) -> list[ast.AST]:
    """Return children excluding low-signal context nodes."""
    return [child for child in ast.iter_child_nodes(node) if type(child).__name__ not in IGNORED_AST_NODE_TYPES]


def ast_to_indented_text(code: str, include_values: bool = False) -> str:
    """Render Python code's AST as an indented text tree."""
    tree = parse_python_ast(code)
    lines: list[str] = []

    def visit(node: ast.AST, depth: int) -> None:
        if type(node).__name__ in IGNORED_AST_NODE_TYPES:
            return

        lines.append(f"{'  ' * depth}- {_node_label(node, include_values)}")
        for child in _visible_children(node):
            visit(child, depth + 1)

    visit(tree, 0)
    return "\n".join(lines)


def _escape_mermaid_label(label: str) -> str:
    """Escape labels for Mermaid flowcharts."""
    return label.replace("\\", "\\\\").replace('"', '\\"')


def ast_to_mermaid(code: str, include_values: bool = False) -> str:
    """Render Python code's AST as a Mermaid flowchart."""
    tree = parse_python_ast(code)
    lines = ["flowchart TD"]
    counter = 0

    def visit(node: ast.AST) -> str:
        nonlocal counter

        node_id = f"n{counter}"
        counter += 1
        label = _escape_mermaid_label(_node_label(node, include_values))
        lines.append(f'    {node_id}["{label}"]')

        for child in _visible_children(node):
            child_id = visit(child)
            lines.append(f"    {node_id} --> {child_id}")

        return node_id

    visit(tree)
    return "\n".join(lines)


def _escape_dot_label(label: str) -> str:
    """Escape labels for Graphviz DOT."""
    return label.replace("\\", "\\\\").replace('"', '\\"')


def ast_to_dot(code: str, include_values: bool = False) -> str:
    """Render Python code's AST as Graphviz DOT text."""
    tree = parse_python_ast(code)
    lines = [
        "digraph AST {",
        "  rankdir=TB;",
        '  node [shape=box, style="rounded", fontname="Menlo"];',
    ]
    counter = 0

    def visit(node: ast.AST) -> str:
        nonlocal counter

        node_id = f"n{counter}"
        counter += 1
        label = _escape_dot_label(_node_label(node, include_values))
        lines.append(f'  {node_id} [label="{label}"];')

        for child in _visible_children(node):
            child_id = visit(child)
            lines.append(f"  {node_id} -> {child_id};")

        return node_id

    visit(tree)
    lines.append("}")
    return "\n".join(lines)


def export_ast_visualizations(
    code: str,
    name: str,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    include_values: bool = False,
) -> dict[str, Path]:
    """Export AST visualizations as text, Mermaid Markdown, and DOT files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in name)

    text_path = output_dir / f"{safe_name}.txt"
    mermaid_path = output_dir / f"{safe_name}.md"
    dot_path = output_dir / f"{safe_name}.dot"

    text_path.write_text(ast_to_indented_text(code, include_values), encoding="utf-8")
    mermaid_path.write_text(f"```mermaid\n{ast_to_mermaid(code, include_values)}\n```\n", encoding="utf-8")
    dot_path.write_text(ast_to_dot(code, include_values), encoding="utf-8")

    return {
        "text": text_path,
        "mermaid": mermaid_path,
        "dot": dot_path,
    }


if __name__ == "__main__":
    sample_code = """
def sum_even_numbers(values):
    total = 0
    for number in values:
        if number % 2 == 0:
            total += number
    return total
"""

    paths = export_ast_visualizations(sample_code, "sum_even_numbers")
    print("AST visualizations exported:")
    for format_name, path in paths.items():
        print(f"- {format_name}: {path}")
