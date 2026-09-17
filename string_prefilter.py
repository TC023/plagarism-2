"""Capa 0 para detectar duplicados tipo 1 (copias casi exactas).

Esta capa preprocesa el codigo fuente: elimina comentarios, docstrings y
whitespaces innecesarios, y despues aplica comparaciones rapidas para
descartar clones casi exactos antes de pasar a las capas mas costosas.

No pretende sustituir otras capas; es una eliminacion temprana de
candidatos obvios para ahorrar trabajo a las etapas posteriores.
"""

from __future__ import annotations

import re
import difflib
from typing import Tuple, Any

from lexical_statistical_layer import preprocess_code


def preprocess_for_type1(code: str, preprocessed: bool = False) -> str:
    """Preprocesa `code` para deteccion Type-1.

    - Si `preprocessed` es False, llama a `preprocess_code` para quitar
        comentarios, docstrings y normalizar espacios base.
    - Elimina lineas en blanco, recorta espacios por linea y colapsa
        secuencias de whitespace en un solo espacio.
    - Devuelve una cadena compacta y estable para comparar.
    """
    if not isinstance(code, str):
        raise TypeError("code must be a string")

    clean = code if preprocessed else preprocess_code(code)
    # Quitar líneas vacías y recortar espacios por línea
    lines = [line.strip() for line in clean.splitlines() if line.strip()]
    joined = "\n".join(lines)
    # Colapsar cualquier whitespace (incluye nuevas líneas) a espacios simples
    collapsed = re.sub(r"\s+", " ", joined)
    return collapsed.strip()


def exact_match(code_a: str, code_b: str, preprocessed: bool = False) -> bool:
    """Devuelve True si las versiones normalizadas son idénticas."""
    return preprocess_for_type1(code_a, preprocessed) == preprocess_for_type1(code_b, preprocessed)


def similarity_ratio(code_a: str, code_b: str, preprocessed: bool = False) -> float:
    """Devuelve un ratio de similitud (0..1) usando SequenceMatcher.

    Es rápido y suficiente para detectar copias casi exactas; para búsquedas
    a gran escala se podría usar shingling/minhash.
    """
    a = preprocess_for_type1(code_a, preprocessed)
    b = preprocess_for_type1(code_b, preprocessed)

    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    return difflib.SequenceMatcher(None, a, b).ratio()


def is_near_duplicate(code_a: str, code_b: str, threshold: float = 0.95, preprocessed: bool = False) -> Tuple[bool, float]:
    """Indica si dos fragmentos son duplicados cercanos según `threshold`.

    Retorna (es_duplicado, ratio).
    """
    ratio = similarity_ratio(code_a, code_b, preprocessed)
    return (ratio >= threshold, ratio)


def analyze_string_prefilter(code_a: str, code_b: str, threshold: float = 0.95, preprocessed: bool = False) -> dict[str, Any]:
    """Analiza dos fragmentos y devuelve métricas para prefiltro tipo‑1.

    Resultado dict contiene:
    - `normalized_a`, `normalized_b`: versiones normalizadas
    - `exact_match`: True si son idénticos
    - `similarity_ratio`: ratio (0..1)
    - `threshold`: umbral usado
    - `is_near_duplicate`: True si ratio >= threshold
    """
    normalized_a = preprocess_for_type1(code_a, preprocessed)
    normalized_b = preprocess_for_type1(code_b, preprocessed)
    exact = normalized_a == normalized_b
    ratio = similarity_ratio(normalized_a, normalized_b, preprocessed=True)
    near = ratio >= threshold

    return {
        "normalized_a": normalized_a,
        "normalized_b": normalized_b,
        "exact_match": exact,
        "similarity_ratio": ratio,
        "threshold": threshold,
        "is_near_duplicate": near,
    }


def print_string_prefilter_report(results: dict[str, Any]) -> None:
    """Imprime un informe compacto similar a las otras capas."""
    print("=== Capa 0: Prefiltro de strings (Tipo 1) ===")
    print(f"- Exact match: {results['exact_match']}")
    print(f"- Similarity ratio: {results['similarity_ratio']:.4f}")
    print(f"- Threshold: {results['threshold']:.2f}")
    print(f"- Near duplicate: {results['is_near_duplicate']}")


if __name__ == "__main__":
    # Demo rápido
    code_a = '''
def sum_even_numbers(values):
    # Sum even numbers
    total = 0
    for number in values:
        if number % 2 == 0:
            total += number
            total += number
            total += number
            total += number
            total += number
            total += number
            total += number
    return total
'''

    code_b = '''
def sum_even_numbers(values):
    total=0
    for number in values:
        if number%2==0:
            total+=number
            total += number
            total += number
            total += number
            total += number
            total += number
            total += number
    return total
'''

    results = analyze_string_prefilter(code_a, code_b, threshold=0.98)
    print_string_prefilter_report(results)
