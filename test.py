"""Campo de pruebas para la Capa 1 lexico-estadistica.

Este archivo no contiene la logica del analizador. Solo define casos de prueba
y usa las funciones publicas de lexical_statistical_layer.py.
"""

from __future__ import annotations

import csv
from pathlib import Path

from ast_visualizer import export_ast_visualizations
from embeddings import analyze_embedding_similarity, print_embedding_report
from lexical_statistical_layer import analyze_lexical_statistical_similarity, preprocess_code
from semantic_layer import analyze_semantic_similarity
from structural_layer import analyze_structural_similarity
from string_prefilter import analyze_string_prefilter, print_string_prefilter_report


TestCase = tuple[str, str, str, str, str, str, str]

RESULTS_DIR = Path("results")
CSV_OUTPUT_PATH = RESULTS_DIR / "lexical_statistical_results.csv"
STRUCTURAL_CSV_OUTPUT_PATH = RESULTS_DIR / "structural_results.csv"
SEMANTIC_CSV_OUTPUT_PATH = RESULTS_DIR / "semantic_results.csv"
EMBEDDING_CSV_OUTPUT_PATH = RESULTS_DIR / "embedding_results.csv"
COMBINED_CSV_OUTPUT_PATH = RESULTS_DIR / "combined_layer_results.csv"
AST_VISUALIZATION_DIR = RESULTS_DIR / "ast_visualizations"


TEST_CASES: list[TestCase] = [
    (
        "Misma logica con nombres cambiados",
        """
def sum_even_numbers(values):
    total = 0
    for number in values:
        if number % 2 == 0:
            total = total + number
    return total
""",
        """
def WEARZQ(items):
    result = 0
    for item in items:
        if item % 2 == 0:
            result = result + item
    return result
""",
        "alta",
        "alta",
        "alta",
        "Debe dar similitud alta: cambian nombres, pero la estructura lexica es parecida.",
    ),
    (
        "Misma logica con comentarios y docstrings diferentes",
        """
def count_positive(numbers):
    \"\"\"Count values greater than zero.\"\"\"
    total = 0
    for value in numbers:
        if value > 0:
            total += 1
    return total
""",
        """
def positives(data):
    # This comment should not affect the lexical score.
    \"\"\"Different documentation text.\"\"\"
    counter = 0
    for element in data:
        if element > 0:
            counter = counter + 1
    return counter
""",
        "alta",
        "alta",
        "alta",
        "Debe dar similitud alta: comentarios y docstrings se eliminan en preprocesamiento.",
    ),
    (
        "Mismo patron con while en lugar de for",
        """
def sum_values(values):
    total = 0
    for value in values:
        total += value
    return total
""",
        """
def sum_values_indexed(values):
    total = 0
    index = 0
    while index < len(values):
        total += values[index]
        index += 1
    return total
""",
        "media",
        "media",
        "alta",
        "Debe quedar en similitud media: la intencion es parecida, pero cambia el flujo lexico.",
    ),
    (
        "Codigo distinto",
        """
def factorial(n):
    result = 1
    while n > 1:
        result *= n
        n -= 1
    return result
""",
        """
def greet_user(name):
    message = "Hello, " + name
    print(message)
    return message
        """,
        "media",
        "media",
        "baja",
        "Debe bajar la similitud: cambian tokens, operadores y flujo de control.",
    ),
    (
        "Algoritmos de busqueda similares",
        """
def contains_value(values, target):
    for value in values:
        if value == target:
            return True
    return False
""",
        """
def find_item(collection, wanted):
    for item in collection:
        if item == wanted:
            return True
    return False
""",
        "alta",
        "alta",
        "alta",
        "Debe dar similitud muy alta: se conserva el mismo patron de control y retorno.",
    ),
    (
        "Misma salida con implementacion muy distinta",
        """
def square_numbers(values):
    result = []
    for value in values:
        result.append(value * value)
    return result
""",
        """
def square_numbers_compact(values):
    return [item ** 2 for item in values]
""",
        "media",
        "baja",
        "alta",
        "Debe bajar respecto a casos identicos: mismo objetivo, pero forma lexica diferente.",
    ),
    (
        "Cambio de operadores de comparacion",
        """
def filter_adults(ages):
    adults = []
    for age in ages:
        if age >= 18:
            adults.append(age)
    return adults
""",
        """
def filter_minors(values):
    minors = []
    for value in values:
        if value < 18:
            minors.append(value)
    return minors
""",
        "alta",
        "alta",
        "baja",
        "Debe mostrar similitud parcial: estructura parecida, condicion logica diferente.",
    ),
    (
        "Misma operacion con literales distintos",
        """
def discount(price):
    return price * 0.90
""",
        """
def apply_tax(amount):
    return amount * 1.16
        """,
        "alta",
        "alta",
        "baja",
        "Debe mantener similitud alta: los literales numericos se normalizan como LITERAL.",
    ),
    (
        "Funciones con orden de instrucciones cambiado",
        """
def normalize_name(first, last):
    first = first.strip().title()
    last = last.strip().title()
    return first + " " + last
""",
        """
def format_name(a, b):
    b = b.strip().title()
    a = a.strip().title()
    return a + " " + b
""",
        "alta",
        "alta",
        "alta",
        "Debe dar similitud alta pero no perfecta: tokens parecidos con orden parcialmente cambiado.",
    ),
    (
        "Uso de funciones integradas distinto",
        """
def average(values):
    total = 0
    for value in values:
        total += value
    return total / len(values)
""",
        """
def mean(items):
    return sum(items) / len(items)
""",
        "media",
        "media",
        "alta",
        "Debe quedar en similitud media: misma idea matematica, expresion lexica mas compacta.",
    ),
    (
        "Caso extremo: codigo vacio contra funcion",
        "",
        """
def identity(value):
    return value
""",
        "baja",
        "baja",
        "baja",
        "Debe dar similitud baja: un lado no aporta tokens.",
    ),
]


def classify_similarity(score: float) -> str:
    """Classify the Layer-1 score using provisional research thresholds."""
    if score >= 0.80:
        return "alta"
    if score >= 0.50:
        return "media"
    return "baja"


def run_layers_once(code_a: str, code_b: str) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    """Preprocess a code pair once and run every analysis layer."""
    clean_code_a = preprocess_code(code_a)
    clean_code_b = preprocess_code(code_b)

    lexical_results = analyze_lexical_statistical_similarity(clean_code_a, clean_code_b, preprocessed=True)
    structural_results = analyze_structural_similarity(clean_code_a, clean_code_b, preprocessed=True)
    semantic_results = analyze_semantic_similarity(clean_code_a, clean_code_b, preprocessed=True)
    embedding_results = analyze_embedding_similarity(clean_code_a, clean_code_b, preprocessed=True)

    return lexical_results, structural_results, semantic_results, embedding_results


def build_result_row(
    name: str,
    expected_lexical: str,
    expected_structural: str,
    expected_semantic: str,
    note: str,
    lexical_results: dict[str, object],
    structural_results: dict[str, object],
    semantic_results: dict[str, object],
    embedding_results: dict[str, object],
    string_results: dict[str, object],
) -> dict[str, str | float]:
    """Build the CSV/report row from already computed layer results."""
    lexical_score = float(lexical_results["lexical_statistical_score"])
    structural_score = float(structural_results["structural_score"])
    semantic_score = float(semantic_results["semantic_score"])
    lexical_observed = classify_similarity(lexical_score)
    structural_observed = classify_similarity(structural_score)
    semantic_observed = classify_similarity(semantic_score)

    return {
        "caso": name,
        "similitud_esperada_lexica": expected_lexical,
        "similitud_esperada_estructural": expected_structural,
        "similitud_esperada_semantica": expected_semantic,
        "similitud_observada_lexica": lexical_observed,
        "similitud_observada_estructural": structural_observed,
        "similitud_observada_semantica": semantic_observed,
        "coincide_lexica": "si" if lexical_observed == expected_lexical else "no",
        "coincide_estructural": "si" if structural_observed == expected_structural else "no",
        "coincide_semantica": "si" if semantic_observed == expected_semantic else "no",
        "jaccard_similarity": float(lexical_results["jaccard_similarity"]),
        "tfidf_cosine_similarity": float(lexical_results["tfidf_cosine_similarity"]),
        "markov_similarity": float(lexical_results["markov_similarity"]),
        "kl_divergence_a_to_b": float(lexical_results["kl_divergence_a_to_b"]),
        "kl_divergence_b_to_a": float(lexical_results["kl_divergence_b_to_a"]),
        "lexical_statistical_score": lexical_score,
        "string_exact_match": string_results.get("exact_match", False),
        "string_similarity_ratio": float(string_results.get("similarity_ratio", 0.0)),
        "ast_node_type_jaccard": float(structural_results["ast_node_type_jaccard"]),
        "ast_sequence_similarity": float(structural_results["ast_sequence_similarity"]),
        "ast_node_count_similarity": float(structural_results["ast_node_count_similarity"]),
        "ast_depth_similarity": float(structural_results["ast_depth_similarity"]),
        "tree_edit_distance": structural_results["tree_edit_distance"],
        "tree_edit_similarity": float(structural_results["tree_edit_similarity"]),
        "apted_available": structural_results["apted_available"],
        "apted_tree_edit_distance": structural_results["apted_tree_edit_distance"],
        "apted_tree_edit_similarity": structural_results["apted_tree_edit_similarity"],
        "structural_score": structural_score,
        "semantic_runnable": semantic_results["semantic_runnable"],
        "semantic_reason": semantic_results["semantic_reason"],
        "semantic_function_a": semantic_results.get("function_a", ""),
        "semantic_function_b": semantic_results.get("function_b", ""),
        "semantic_tested_arity": semantic_results.get("tested_arity", ""),
        "semantic_total_cases": semantic_results.get("total_cases", 0),
        "semantic_both_return_count": semantic_results.get("both_return_count", 0),
        "semantic_matching_return_count": semantic_results.get("matching_return_count", 0),
        "semantic_successful_overlap": semantic_results.get("successful_overlap", 0.0),
        "semantic_output_similarity": semantic_results.get("output_similarity", 0.0),
        "semantic_score": semantic_score,
        "embedding_model": embedding_results.get("embedding_model", "tfidf"),
        "embedding_feature_count": int(embedding_results.get("embedding_feature_count", 0)),
        "embedding_cosine_similarity": float(embedding_results.get("embedding_cosine_similarity", 0.0)),
        "embedding_score": float(embedding_results.get("embedding_score", 0.0)),
        "nota": note,
    }


def export_rows_to_csv(rows: list[dict[str, str | float]], fieldnames: list[str], output_path: Path) -> None:
    """Export selected playground metrics to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fieldnames} for row in rows)


def export_all_results(rows: list[dict[str, str | float]]) -> None:
    """Export lexical, structural, and combined CSV files."""
    lexical_fieldnames = [
        "caso",
        "similitud_esperada_lexica",
        "similitud_observada_lexica",
        "coincide_lexica",
        "string_exact_match",
        "string_similarity_ratio",
        "jaccard_similarity",
        "tfidf_cosine_similarity",
        "markov_similarity",
        "kl_divergence_a_to_b",
        "kl_divergence_b_to_a",
        "lexical_statistical_score",
        "nota",
    ]

    structural_fieldnames = [
        "caso",
        "similitud_esperada_estructural",
        "similitud_observada_estructural",
        "coincide_estructural",
        "ast_node_type_jaccard",
        "ast_sequence_similarity",
        "ast_node_count_similarity",
        "ast_depth_similarity",
        "tree_edit_distance",
        "tree_edit_similarity",
        "apted_available",
        "apted_tree_edit_distance",
        "apted_tree_edit_similarity",
        "structural_score",
        "nota",
    ]

    semantic_fieldnames = [
        "caso",
        "similitud_esperada_semantica",
        "similitud_observada_semantica",
        "coincide_semantica",
        "semantic_runnable",
        "semantic_reason",
        "semantic_function_a",
        "semantic_function_b",
        "semantic_tested_arity",
        "semantic_total_cases",
        "semantic_both_return_count",
        "semantic_matching_return_count",
        "semantic_successful_overlap",
        "semantic_output_similarity",
        "semantic_score",
        "nota",
    ]

    embedding_fieldnames = [
        "caso",
        "embedding_model",
        "embedding_feature_count",
        "embedding_cosine_similarity",
        "embedding_score",
        "nota",
    ]

    combined_fieldnames = [
        "caso",
        "similitud_esperada_lexica",
        "similitud_esperada_estructural",
        "similitud_esperada_semantica",
        "similitud_observada_lexica",
        "similitud_observada_estructural",
        "similitud_observada_semantica",
        "coincide_lexica",
        "coincide_estructural",
        "coincide_semantica",
        "string_exact_match",
        "string_similarity_ratio",
        "jaccard_similarity",
        "tfidf_cosine_similarity",
        "markov_similarity",
        "kl_divergence_a_to_b",
        "kl_divergence_b_to_a",
        "lexical_statistical_score",
        "ast_node_type_jaccard",
        "ast_sequence_similarity",
        "ast_node_count_similarity",
        "ast_depth_similarity",
        "tree_edit_distance",
        "tree_edit_similarity",
        "apted_available",
        "apted_tree_edit_distance",
        "apted_tree_edit_similarity",
        "structural_score",
        "semantic_runnable",
        "semantic_reason",
        "semantic_output_similarity",
        "semantic_successful_overlap",
        "semantic_score",
        "embedding_model",
        "embedding_feature_count",
        "embedding_cosine_similarity",
        "embedding_score",
        "nota",
    ]

    export_rows_to_csv(rows, lexical_fieldnames, CSV_OUTPUT_PATH)
    export_rows_to_csv(rows, structural_fieldnames, STRUCTURAL_CSV_OUTPUT_PATH)
    export_rows_to_csv(rows, semantic_fieldnames, SEMANTIC_CSV_OUTPUT_PATH)
    export_rows_to_csv(rows, embedding_fieldnames, EMBEDDING_CSV_OUTPUT_PATH)
    export_rows_to_csv(rows, combined_fieldnames, COMBINED_CSV_OUTPUT_PATH)


def print_compact_report(
    name: str,
    code_a: str,
    code_b: str,
    expected_lexical: str,
    expected_structural: str,
    expected_semantic: str,
    note: str,
) -> dict[str, str | float]:
    """Run one test case and print the most useful metrics."""
    # Preprocess and run layers
    clean_code_a = preprocess_code(code_a)
    clean_code_b = preprocess_code(code_b)

    lexical_results, structural_results, semantic_results, embedding_results = run_layers_once(code_a, code_b)
    string_results = analyze_string_prefilter(code_a, code_b, threshold=0.98)
    lexical_observed = classify_similarity(lexical_results["lexical_statistical_score"])
    structural_observed = classify_similarity(structural_results["structural_score"])
    semantic_observed = classify_similarity(semantic_results["semantic_score"])

    print("=" * 72)
    print(f"Caso: {name}")
    print(note)
    print(f"Similitud esperada Capa 1: {expected_lexical}")
    print(f"Similitud esperada Capa 2: {expected_structural}")
    print(f"Similitud esperada Capa 3: {expected_semantic}")
    print(f"Similitud observada Capa 1: {lexical_observed}")
    print(f"Similitud observada Capa 2: {structural_observed}")
    print(f"Similitud observada Capa 3: {semantic_observed}")
    print("-" * 72)
    print(f"Jaccard:        {lexical_results['jaccard_similarity']:.4f}")
    print(f"TF-IDF coseno:  {lexical_results['tfidf_cosine_similarity']:.4f}")
    print(f"Markov:         {lexical_results['markov_similarity']:.4f}")
    print(f"KL A->B:        {lexical_results['kl_divergence_a_to_b']:.4f}")
    print(f"KL B->A:        {lexical_results['kl_divergence_b_to_a']:.4f}")
    print(f"Score Capa 1:   {lexical_results['lexical_statistical_score']:.4f}")
    print(f"AST Jaccard:    {structural_results['ast_node_type_jaccard']:.4f}")
    print(f"AST secuencia:  {structural_results['ast_sequence_similarity']:.4f}")
    print(f"AST tamano:     {structural_results['ast_node_count_similarity']:.4f}")
    print(f"AST profundidad:{structural_results['ast_depth_similarity']:.4f}")
    print(f"Tree Edit Dist: {structural_results['tree_edit_distance']}")
    print(f"Tree Edit Sim:  {structural_results['tree_edit_similarity']:.4f}")
    print(f"APTED activo:   {structural_results['apted_available']}")
    if structural_results["apted_available"] == "si":
        print(f"APTED Dist:     {structural_results['apted_tree_edit_distance']}")
        print(f"APTED Sim:      {structural_results['apted_tree_edit_similarity']:.4f}")
    print(f"Score Capa 2:   {structural_results['structural_score']:.4f}")
    print(f"Runnable Capa 3:{semantic_results['semantic_runnable']}")
    print(f"Output Sim:     {semantic_results.get('output_similarity', 0.0):.4f}")
    print(f"Overlap exito:  {semantic_results.get('successful_overlap', 0.0):.4f}")
    print(f"Score Capa 3:   {semantic_results['semantic_score']:.4f}")
    print()
    # Prefiltro Tipo 1
    print_string_prefilter_report(string_results)
    print()
    print_embedding_report(embedding_results)
    print("\nTokens normalizados A:")
    print(lexical_results["normalized_tokens_a"])
    print("\nTokens normalizados B:")
    print(lexical_results["normalized_tokens_b"])
    print()

    return build_result_row(
        name,
        expected_lexical,
        expected_structural,
        expected_semantic,
        note,
        lexical_results,
        structural_results,
        semantic_results,
        embedding_results,
        string_results,
    )


def main() -> None:
    """Execute all playground cases."""
    print("Campo de pruebas: detector lexico-estadistico para codigo Python\n")
    rows = []
    first_code_pair: tuple[str, str, str] | None = None
    for name, code_a, code_b, expected_lexical, expected_structural, expected_semantic, note in TEST_CASES:
        if first_code_pair is None:
            first_code_pair = (name, code_a, code_b)
        rows.append(print_compact_report(name, code_a, code_b, expected_lexical, expected_structural, expected_semantic, note))

    export_all_results(rows)
    print(f"Resultados exportados a: {CSV_OUTPUT_PATH}")
    print(f"Resultados estructurales exportados a: {STRUCTURAL_CSV_OUTPUT_PATH}")
    print(f"Resultados semanticos exportados a: {SEMANTIC_CSV_OUTPUT_PATH}")
    print(f"Resultados de embeddings exportados a: {EMBEDDING_CSV_OUTPUT_PATH}")
    print(f"Resultados combinados exportados a: {COMBINED_CSV_OUTPUT_PATH}")

    if first_code_pair is not None:
        _, code_a, code_b = first_code_pair
        paths_a = export_ast_visualizations(code_a, "case_01_code_a", AST_VISUALIZATION_DIR)
        paths_b = export_ast_visualizations(code_b, "case_01_code_b", AST_VISUALIZATION_DIR)
        print(f"Visualizaciones AST exportadas a: {AST_VISUALIZATION_DIR}")
        print(f"- Codigo A Mermaid: {paths_a['mermaid']}")
        print(f"- Codigo B Mermaid: {paths_b['mermaid']}")


if __name__ == "__main__":
    main()
