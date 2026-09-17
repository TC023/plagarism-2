r"""
Uso del script:

    python predict_clone_type_casetest.py

Opcionalmente puedes cambiar solo el modelo y las features con flags:

    python predict_clone_type_casetest.py \
        --model-path .\model_outputs\best_clone_type_model.joblib \
        --features-path .\model_outputs\best_features.json

Valores fijos del script:
    - cases-dir: .\casestest
    - output-csv: results/casestest_clone_type_predictions.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from joblib import load

from embeddings import analyze_embedding_similarity
from lexical_statistical_layer import analyze_lexical_statistical_similarity, preprocess_code
from structural_layer import analyze_structural_similarity
from string_prefilter import analyze_string_prefilter

CLONE_TYPE_DESCRIPTIONS = {
    0: "No clone / distinto",
    1: "Copy-paste / copia directa",
    2: "Renamed / reformateado",
    3: "Modificado mismo objetivo",
}

DEFAULT_MODEL_PATH = Path("model_outputs/best_clone_type_model.joblib")
DEFAULT_FEATURES_PATH = Path("model_outputs/best_features.json")
DEFAULT_CASES_DIR = Path("casestest")
DEFAULT_OUTPUT_CSV = Path("results/casestest_clone_type_predictions.csv")


def find_pair_dirs(cases_dir: Path) -> list[Path]:
    pairs: list[Path] = []
    if not cases_dir.exists():
        raise FileNotFoundError(f"No existe el directorio de casos: {cases_dir}")

    for child in sorted(cases_dir.iterdir()):
        if child.is_dir():
            pairs.append(child)
    return pairs


def find_pair_files(pair_dir: Path) -> tuple[Path, Path]:
    py_files = sorted([p for p in pair_dir.glob("*.py") if p.is_file()])
    if len(py_files) == 2:
        return py_files[0], py_files[1]

    a_file = None
    b_file = None
    for p in py_files:
        name = p.stem.lower()
        if name.endswith("_a") or name.endswith("a"):
            a_file = p
        elif name.endswith("_b") or name.endswith("b"):
            b_file = p
    if a_file and b_file:
        return a_file, b_file

    raise ValueError(
        f"No se pudo identificar un par de archivos en {pair_dir}. Esperado exactamente 2 archivos .py o nombres con A/B."
    )


def load_features(features_path: Path) -> list[str]:
    if not features_path.exists():
        raise FileNotFoundError(f"No existe el archivo de features: {features_path}")
    with features_path.open("r", encoding="utf-8") as f:
        features = json.load(f)
    if not isinstance(features, list):
        raise ValueError(f"El contenido de {features_path} no es una lista de nombres de features.")
    return features


def compute_features(code_a: str, code_b: str) -> dict[str, float]:
    clean_a = preprocess_code(code_a)
    clean_b = preprocess_code(code_b)

    lexical_results = analyze_lexical_statistical_similarity(clean_a, clean_b, preprocessed=True)
    structural_results = analyze_structural_similarity(clean_a, clean_b, preprocessed=True)
    embedding_results = analyze_embedding_similarity(clean_a, clean_b, preprocessed=True)
    string_results = analyze_string_prefilter(clean_a, clean_b, threshold=0.95, preprocessed=True)

    def float_or_zero(value: object) -> float:
        try:
            return float(value)
        except Exception:
            return 0.0

    return {
        "string_similarity_ratio": float_or_zero(string_results.get("similarity_ratio", 0.0)),
        "jaccard_similarity": float_or_zero(lexical_results.get("jaccard_similarity", 0.0)),
        "tfidf_cosine_similarity": float_or_zero(lexical_results.get("tfidf_cosine_similarity", 0.0)),
        "entropy_difference": float_or_zero(lexical_results.get("entropy_difference", 0.0)),
        "markov_similarity": float_or_zero(lexical_results.get("markov_similarity", 0.0)),
        "kl_similarity": float_or_zero(lexical_results.get("kl_similarity", 0.0)),
        "ast_node_type_jaccard": float_or_zero(structural_results.get("ast_node_type_jaccard", 0.0)),
        "ast_sequence_similarity": float_or_zero(structural_results.get("ast_sequence_similarity", 0.0)),
        "ast_node_count_similarity": float_or_zero(structural_results.get("ast_node_count_similarity", 0.0)),
        "ast_depth_similarity": float_or_zero(structural_results.get("ast_depth_similarity", 0.0)),
        "tree_edit_similarity": float_or_zero(structural_results.get("tree_edit_similarity", 0.0)),
        "apted_tree_edit_similarity": float_or_zero(structural_results.get("apted_tree_edit_similarity", 0.0)),
        "embedding_cosine_similarity": float_or_zero(embedding_results.get("embedding_cosine_similarity", 0.0)),
    }


def build_prediction_rows(
    model_path: Path,
    features_path: Path,
    cases_dir: Path,
) -> list[dict[str, object]]:
    model = load(model_path)
    features = load_features(features_path)

    rows: list[dict[str, object]] = []
    pair_dirs = find_pair_dirs(cases_dir)

    for pair_dir in pair_dirs:
        file_a_path, file_b_path = find_pair_files(pair_dir)
        code_a = file_a_path.read_text(encoding="utf-8")
        code_b = file_b_path.read_text(encoding="utf-8")

        feature_values = compute_features(code_a, code_b)
        feature_row = {name: feature_values.get(name, 0.0) for name in features}
        X = pd.DataFrame([feature_row], columns=features)

        prediction = model.predict(X)[0]
        rows.append(
            {
                "pair_dir": pair_dir.name,
                "file_a": str(file_a_path),
                "file_b": str(file_b_path),
                "predicted_clone_type": int(prediction),
                "description": CLONE_TYPE_DESCRIPTIONS.get(int(prediction), "Desconocido"),
                **feature_row,
            }
        )

    return rows


def save_predictions(rows: list[dict[str, object]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Guardado CSV de predicciones en: {output_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predice clone_type para pares de codigo usando un modelo entrenado."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Ruta al modelo joblib entrenado.",
    )
    parser.add_argument(
        "--features-path",
        type=Path,
        default=DEFAULT_FEATURES_PATH,
        help="Ruta al JSON con la lista de features usadas por el modelo.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_prediction_rows(args.model_path, args.features_path, DEFAULT_CASES_DIR)
    if not rows:
        print("No se encontraron pares para predecir.")
        return

    for row in rows:
        print(
            f"{row['pair_dir']}: clone_type={row['predicted_clone_type']} "
            f"({row['description']})"
        )

    save_predictions(rows, DEFAULT_OUTPUT_CSV)


if __name__ == "__main__":
    main()
