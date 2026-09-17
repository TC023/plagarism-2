"""Behavioral semantic layer for Python code similarity.

This module compares Python functions by executing them with controlled inputs
inside a subprocess. It is intended for small academic examples, not for running
untrusted code in a strong security sandbox.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

from lexical_statistical_layer import preprocess_code


DEFAULT_TIMEOUT_SECONDS = 2.0


@dataclass(frozen=True)
class FunctionInfo:
    """Basic callable metadata extracted from Python source code."""

    name: str
    required_args: int
    total_positional_args: int
    has_varargs: bool


SAFE_EXECUTION_RUNNER = r"""
import json
import traceback

payload = json.loads(input())

safe_builtins = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "pow": pow,
    "print": print,
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}

namespace = {
    "__builtins__": safe_builtins,
    "__name__": "__semantic_execution__",
}

try:
    exec(payload["code"], namespace, namespace)
    function = namespace[payload["function_name"]]
    result = function(*payload["args"])
    response = {
        "status": "return",
        "type": type(result).__name__,
        "repr": repr(result),
        "jsonable": True,
        "value": result,
    }
    json.dumps(response)
except TypeError as exc:
    response = {
        "status": "exception",
        "type": type(exc).__name__,
        "repr": str(exc),
        "jsonable": True,
        "value": None,
    }
except Exception as exc:
    response = {
        "status": "exception",
        "type": type(exc).__name__,
        "repr": str(exc),
        "traceback": traceback.format_exc(limit=1),
        "jsonable": True,
        "value": None,
    }

try:
    print(json.dumps(response))
except TypeError:
    response["jsonable"] = False
    response["value"] = None
    print(json.dumps(response))
"""


def extract_first_function_info(code: str, preprocessed: bool = False) -> FunctionInfo | None:
    """Return metadata for the first function defined in raw or clean code."""
    clean_code = code if preprocessed else preprocess_code(code)
    if not clean_code:
        return None

    try:
        tree = ast.parse(clean_code)
    except SyntaxError as exc:
        raise ValueError(f"Could not parse Python code: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            positional_args = list(node.args.posonlyargs) + list(node.args.args)
            total_positional_args = len(positional_args)
            required_args = total_positional_args - len(node.args.defaults)
            return FunctionInfo(
                name=node.name,
                required_args=required_args,
                total_positional_args=total_positional_args,
                has_varargs=node.args.vararg is not None,
            )

    return None


def choose_comparable_arity(function_a: FunctionInfo, function_b: FunctionInfo) -> int | None:
    """Choose a positional arity supported by both functions."""
    max_a = function_a.total_positional_args if not function_a.has_varargs else max(function_a.total_positional_args, 3)
    max_b = function_b.total_positional_args if not function_b.has_varargs else max(function_b.total_positional_args, 3)

    lower_bound = max(function_a.required_args, function_b.required_args)
    upper_bound = min(max_a, max_b)

    if lower_bound > upper_bound:
        return None

    return lower_bound


def generate_test_inputs(arity: int) -> list[tuple[Any, ...]]:
    """Generate deterministic test inputs for small Python functions."""
    if arity < 0:
        raise ValueError("arity must be non-negative")

    if arity == 0:
        return [()]
    if arity == 1:
        return [
            ([1, 2, 3],),
            ([0, 1, 2, 3],),
            ([],),
            ([2, -1, 4],),
            (3,),
            (0,),
            (10,),
            (100,),
            (-5,),
            ("abc",),
        ]
    if arity == 2:
        return [
            ([1, 2, 3], 2),
            ([1, 2, 3], 4),
            ("John", "Doe"),
            ("  ana", "lopez  "),
            (10, 2),
            (0, 1),
            ([1, 2], [1, 3]),
        ]

    base_values: list[Any] = [1, 2, 3, [1, 2, 3], "x"]
    return [tuple(base_values[index % len(base_values)] for index in range(arity))]


def execute_function_once(
    code: str,
    function_name: str,
    args: tuple[Any, ...],
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    preprocessed: bool = False,
) -> dict[str, Any]:
    """Execute one function call in a subprocess and return a normalized result."""
    payload = {
        "code": code if preprocessed else preprocess_code(code),
        "function_name": function_name,
        "args": list(args),
    }

    try:
        completed = subprocess.run(
            [sys.executable, "-c", SAFE_EXECUTION_RUNNER],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "type": "TimeoutExpired",
            "repr": f"Timed out after {timeout_seconds} seconds",
        }

    if completed.returncode != 0:
        return {
            "status": "runner_error",
            "type": "RunnerError",
            "repr": completed.stderr.strip(),
        }

    try:
        return json.loads(completed.stdout.strip())
    except json.JSONDecodeError:
        return {
            "status": "runner_error",
            "type": "InvalidJSON",
            "repr": completed.stdout.strip(),
        }


def _results_match(result_a: dict[str, Any], result_b: dict[str, Any]) -> bool:
    """Return whether two successful function results are equivalent."""
    return (
        result_a.get("status") == "return"
        and result_b.get("status") == "return"
        and result_a.get("type") == result_b.get("type")
        and result_a.get("repr") == result_b.get("repr")
    )


def compare_execution_results(executions: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize paired execution results as behavioral similarity metrics."""
    total_cases = len(executions)
    both_return_cases = [
        execution
        for execution in executions
        if execution["result_a"].get("status") == "return" and execution["result_b"].get("status") == "return"
    ]
    matching_return_cases = [
        execution
        for execution in both_return_cases
        if _results_match(execution["result_a"], execution["result_b"])
    ]
    both_exception_cases = [
        execution
        for execution in executions
        if execution["result_a"].get("status") == "exception" and execution["result_b"].get("status") == "exception"
    ]
    matching_exception_cases = [
        execution
        for execution in both_exception_cases
        if execution["result_a"].get("type") == execution["result_b"].get("type")
    ]

    successful_overlap = len(both_return_cases) / total_cases if total_cases else 0.0
    output_similarity = len(matching_return_cases) / len(both_return_cases) if both_return_cases else 0.0
    exception_similarity = len(matching_exception_cases) / len(both_exception_cases) if both_exception_cases else 0.0
    behavioral_score = output_similarity if both_return_cases else 0.25 * exception_similarity

    return {
        "total_cases": total_cases,
        "both_return_count": len(both_return_cases),
        "matching_return_count": len(matching_return_cases),
        "both_exception_count": len(both_exception_cases),
        "matching_exception_count": len(matching_exception_cases),
        "successful_overlap": successful_overlap,
        "output_similarity": output_similarity,
        "exception_similarity": exception_similarity,
        "behavioral_score": behavioral_score,
    }


def analyze_semantic_similarity(
    code_a: str,
    code_b: str,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    preprocessed: bool = False,
) -> dict[str, Any]:
    """Compare two Python snippets by executing their first functions."""
    clean_code_a = code_a if preprocessed else preprocess_code(code_a)
    clean_code_b = code_b if preprocessed else preprocess_code(code_b)
    function_a = extract_first_function_info(clean_code_a, preprocessed=True)
    function_b = extract_first_function_info(clean_code_b, preprocessed=True)

    if function_a is None or function_b is None:
        return {
            "semantic_mode": "behavioral_execution",
            "semantic_runnable": "no",
            "semantic_reason": "missing_function",
            "semantic_score": 0.0,
            "execution_cases": [],
        }

    arity = choose_comparable_arity(function_a, function_b)
    if arity is None:
        return {
            "semantic_mode": "behavioral_execution",
            "semantic_runnable": "no",
            "semantic_reason": "incompatible_function_arity",
            "function_a": function_a.name,
            "function_b": function_b.name,
            "semantic_score": 0.0,
            "execution_cases": [],
        }

    execution_cases = []
    for args in generate_test_inputs(arity):
        result_a = execute_function_once(clean_code_a, function_a.name, args, timeout_seconds, preprocessed=True)
        result_b = execute_function_once(clean_code_b, function_b.name, args, timeout_seconds, preprocessed=True)
        execution_cases.append(
            {
                "args": args,
                "result_a": result_a,
                "result_b": result_b,
                "outputs_match": _results_match(result_a, result_b),
            }
        )

    summary = compare_execution_results(execution_cases)

    return {
        "semantic_mode": "behavioral_execution",
        "semantic_runnable": "si",
        "semantic_reason": "ok",
        "function_a": function_a.name,
        "function_b": function_b.name,
        "tested_arity": arity,
        "execution_cases": execution_cases,
        **summary,
        "semantic_score": summary["behavioral_score"],
    }


def print_semantic_report(results: dict[str, Any]) -> None:
    """Print a compact behavioral semantic report."""
    print("=== Capa 3: Similitud semantica por ejecucion ===")
    print(f"- Runnable: {results['semantic_runnable']}")
    print(f"- Reason: {results['semantic_reason']}")
    if results["semantic_runnable"] == "si":
        print(f"- Function A: {results['function_a']}")
        print(f"- Function B: {results['function_b']}")
        print(f"- Tested arity: {results['tested_arity']}")
        print(f"- Total cases: {results['total_cases']}")
        print(f"- Both returned: {results['both_return_count']}")
        print(f"- Matching returns: {results['matching_return_count']}")
        print(f"- Output similarity: {results['output_similarity']:.4f}")
        print(f"- Successful overlap: {results['successful_overlap']:.4f}")
    print(f"- Semantic score: {results['semantic_score']:.4f}")


if __name__ == "__main__":
    code_a = """
def square_numbers(values):
    result = []
    for value in values:
        result.append(value * value)
    return result
"""

    code_b = """
def square_numbers_compact(values):
    return [item ** 2 for item in values]
"""

    analysis = analyze_semantic_similarity(code_a, code_b)
    print_semantic_report(analysis)
