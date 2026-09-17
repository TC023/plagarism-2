"""Procesa el dataset de plagio y exporta métricas por par de código.

Lee `cheating_dataset_clean.csv`, carga los archivos en `../cases/` y calcula las
métricas de las capas existentes del proyecto.

El resultado se exporta a `results/cheating_dataset_results.csv`.

Reglas para clone_type en la salida:
- Si Label es 0, clone_type se guarda como 0.
- En los demás casos con Label 1, se conserva Clone_type (si está presente).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from embeddings import analyze_embedding_similarity
from lexical_statistical_layer import analyze_lexical_statistical_similarity, preprocess_code
from structural_layer import analyze_structural_similarity
from string_prefilter import analyze_string_prefilter

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
DEFAULT_CASES_DIR = PROJECT_ROOT / "cases"
DEFAULT_INPUT_CSV = PROJECT_ROOT / "cheating_dataset_clean.csv"
TEST_INPUT_CSV = PROJECT_ROOT / "test_input.csv"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "results" / "dataset_result.csv"


def parse_dataset_csv(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV de entrada no encontrado: {csv_path}")

    rows: list[dict[str, str]] = []
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for index, row in enumerate(reader, start=1):
            if "File_1" not in row or "File_2" not in row or "Label" not in row:
                raise ValueError("CSV debe contener columnas File_1, File_2, Label y Clone_type")
            rows.append({
                "file_a": row["File_1"].strip(),
                "file_b": row["File_2"].strip(),
                "label": row["Label"].strip(),
                "clone_type": row.get("Clone_type", "").strip(),
                "row_index": index,
            })
    return rows


def load_source_file(cases_dir: Path, filename: str) -> str:
    path = cases_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Archivo de caso no encontrado: {path}")
    return path.read_text(encoding="utf-8")


def build_metric_row(
    file_a: str,
    file_b: str,
    label: int,
    clone_type: int,
    lexical_results: dict[str, Any],
    structural_results: dict[str, Any],
    embedding_results: dict[str, Any],
    string_results: dict[str, Any],
    error: str = "",
) -> dict[str, Any]:
    # Exporta solo metadatos base + features acordadas en features.MD.
    return {
        "file_a": file_a,
        "file_b": file_b,
        "label": label,
        "clone_type": clone_type,
        "error": error,
        "string_similarity_ratio": float(string_results.get("similarity_ratio", 0.0)),
        "jaccard_similarity": float(lexical_results.get("jaccard_similarity", 0.0)),
        "tfidf_cosine_similarity": float(lexical_results.get("tfidf_cosine_similarity", 0.0)),
        "entropy_difference": float(lexical_results.get("entropy_difference", 0.0)),
        "markov_similarity": float(lexical_results.get("markov_similarity", 0.0)),
        "kl_similarity": float(lexical_results.get("kl_similarity", 0.0)),
        "ast_node_type_jaccard": float(structural_results.get("ast_node_type_jaccard", 0.0)),
        "ast_sequence_similarity": float(structural_results.get("ast_sequence_similarity", 0.0)),
        "ast_node_count_similarity": float(structural_results.get("ast_node_count_similarity", 0.0)),
        "ast_depth_similarity": float(structural_results.get("ast_depth_similarity", 0.0)),
        "tree_edit_similarity": float(structural_results.get("tree_edit_similarity", 0.0)),
        "apted_tree_edit_similarity": structural_results.get("apted_tree_edit_similarity", ""),
        "embedding_cosine_similarity": float(embedding_results.get("embedding_cosine_similarity", 0.0)),
    }


def compute_metrics_for_pair(cases_dir: Path, file_a: str, file_b: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    code_a = load_source_file(cases_dir, file_a)
    code_b = load_source_file(cases_dir, file_b)

    clean_a = preprocess_code(code_a)
    clean_b = preprocess_code(code_b)

    lexical_results = analyze_lexical_statistical_similarity(clean_a, clean_b, preprocessed=True)
    structural_results = analyze_structural_similarity(clean_a, clean_b, preprocessed=True)
    embedding_results = analyze_embedding_similarity(clean_a, clean_b, preprocessed=True)
    string_results = analyze_string_prefilter(clean_a, clean_b, threshold=0.95, preprocessed=True)

    return lexical_results, structural_results, embedding_results, string_results


def export_to_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print("No hay filas para exportar.")
        return

    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(
    input_csv: Path,
    output_csv: Path,
    cases_dir: Path,
) -> None:
    dataset_rows = parse_dataset_csv(input_csv)
    results: list[dict[str, Any]] = []

    total = len(dataset_rows)
    if total == 0:
        print("No hay filas en el CSV de entrada.")
        export_to_csv(results, output_csv)
        return

    last_percent = -1
    for idx, row in enumerate(dataset_rows, start=1):
        file_a = row["file_a"]
        file_b = row["file_b"]
        label = int(row["label"]) if row["label"].isdigit() else 0
        clone_type = int(row["clone_type"]) if row["clone_type"].isdigit() else 0

        try:
            lexical_results, structural_results, embedding_results, string_results = compute_metrics_for_pair(
                cases_dir,
                file_a,
                file_b,
            )
            result_row = build_metric_row(
                file_a,
                file_b,
                label,
                clone_type,
                lexical_results,
                structural_results,
                embedding_results,
                string_results,
            )
        except Exception as exc:
            result_row = build_metric_row(
                file_a,
                file_b,
                label,
                clone_type,
                {},
                {},
                {},
                {},
                error=str(exc),
            )

        results.append(result_row)

        percent = int((idx / total) * 100)
        if percent != last_percent or idx == total:
            print(f"Procesado {idx}/{total} ({percent}%)", flush=True)
            last_percent = percent

    export_to_csv(results, output_csv)
    print(f"Exportado {len(results)} filas a {output_csv}")



def main() -> None:
    run(DEFAULT_INPUT_CSV, DEFAULT_OUTPUT_CSV, DEFAULT_CASES_DIR)


if __name__ == "__main__":
    main()