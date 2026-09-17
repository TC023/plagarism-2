r"""
Entrena modelos para predecir clone_type a partir de metricas de similitud.

Uso:
    python train_model.py

Este script corre siempre con la configuracion equivalente a:
    python train_model.py --csv .\results\dataset_result.csv --compare --output-dir model_outputs

Modelos disponibles:
    - linear_svc
    - logistic_regression
    - random_forest

Notas:
    - La columna objetivo es clone_type.
    - El clone_type 0 se conserva como clase 0.
    - El CSV final de predicciones NO muestra clone_type_real_mapeado.
    - Se genera un CSV con predicciones por par.
    - Se generan CSVs con importancia de metricas/features.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    StratifiedKFold,
    cross_validate,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "model_outputs"
DEFAULT_CSV = PROJECT_ROOT / "results" / "dataset_result.csv"
DEFAULT_RANDOM_STATE = 42
DEFAULT_TOP_N_FEATURES = 10
DEFAULT_CLASS3_WEIGHT = 1.0
DEFAULT_COMPARE_MODE = True
DEFAULT_STABILITY_PENALTY = 0.5

TARGET_COL = "clone_type"


# Limpieza conservadora para quitar negativos que parecen clones demasiado similares.
# Se aplica solo a clone_type = 0, que es donde mas ruido introducen los pares
# casi duplicados que confunden la frontera con la clase 3.
SIMILARITY_CLEANING_COLUMNS = [
    "string_similarity_ratio",
    "jaccard_similarity",
    "tfidf_cosine_similarity",
    "ast_node_type_jaccard",
    "embedding_cosine_similarity",
]
MIN_CLEANING_COLUMNS = 4
NEGATIVE_SIMILARITY_THRESHOLD = 0.79

# Columnas que normalmente NO conviene usar como features porque identifican archivos,
# etiquetas o infojrmacion que podria causar fuga de informacion.
DEFAULT_EXCLUDE_COLS = {
    "clone_type",
    "label",
    "file_a",
    "file_b",
}

PAIR_INFO_CANDIDATES = [
    "file_a",
    "file_b",
    "filename_a",
    "filename_b",
    "path_a",
    "path_b",
    "label",
]

GROUP_A_COLUMNS = ["file_a", "filename_a", "File_1", "file1"]
GROUP_B_COLUMNS = ["file_b", "filename_b", "File_2", "file2"]


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
            return x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: str, b: str) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            self.parent[root_b] = root_a


def find_pair_columns(df: pd.DataFrame) -> tuple[str, str]:
    file_a_col = next((c for c in GROUP_A_COLUMNS if c in df.columns), None)
    file_b_col = next((c for c in GROUP_B_COLUMNS if c in df.columns), None)

    if file_a_col is None or file_b_col is None:
        raise ValueError(
            "No se encontraron las columnas de archivos de par en el dataset. "
            "Se espera alguna de: file_a, filename_a, File_1, file1 y file_b, filename_b, File_2, file2."
        )

    return file_a_col, file_b_col


def get_pair_group_ids(df: pd.DataFrame) -> np.ndarray:
    file_a_col, file_b_col = find_pair_columns(df)
    uf = UnionFind()
    names: set[str] = set()

    file_a_values = df[file_a_col].astype(str).str.strip().tolist()
    file_b_values = df[file_b_col].astype(str).str.strip().tolist()

    for file_a, file_b in zip(file_a_values, file_b_values):
        names.add(file_a)
        names.add(file_b)
        if file_a and file_b:
            uf.union(file_a, file_b)

    root_names = sorted({uf.find(name) for name in names})
    root_to_id = {root: idx for idx, root in enumerate(root_names)}

    groups = np.array([root_to_id[uf.find(file_a)] for file_a in file_a_values], dtype=int)
    return groups


def make_group_cv(groups: np.ndarray, n_splits: int = 5) -> GroupKFold | StratifiedKFold:
    unique_groups = np.unique(groups)
    if len(unique_groups) >= 2:
        n_splits = min(n_splits, len(unique_groups))
        if n_splits >= 2:
            return GroupKFold(n_splits=n_splits)

    return StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def group_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_idx, test_idx = next(splitter.split(X, y, groups))
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]


def to_json_safe(obj: Any) -> Any:
    """Convierte objetos de numpy/pandas a tipos serializables en JSON."""
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def build_similarity_score(df: pd.DataFrame) -> pd.Series:
    """Calcula una puntuacion compuesta para detectar negativos demasiado similares."""

    available_columns = [col for col in SIMILARITY_CLEANING_COLUMNS if col in df.columns]
    if len(available_columns) < MIN_CLEANING_COLUMNS:
        return pd.Series(np.nan, index=df.index, dtype=float)

    score_frame = df[available_columns].apply(pd.to_numeric, errors="coerce")
    score = score_frame.mean(axis=1)

    # Evita puntuar filas que solo tienen una o dos metricas disponibles.
    valid_counts = score_frame.notna().sum(axis=1)
    score = score.where(valid_counts >= MIN_CLEANING_COLUMNS)
    return score


def load_dataset(csv_path: str | Path) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Lee el CSV, prepara X/y y devuelve tambien el dataframe original.

    Reglas:
        - Usa clone_type como objetivo.
        - Conserva 0 como clase 0.
        - Convierte 4 -> 0, por consistencia con el CSV nuevo.
        - Usa como features columnas numericas y booleanas.
        - Excluye columnas identificadoras y de etiqueta.
                - Elimina pares negativos muy similares que no aportan señal y confunden
                    la separacion entre clone_type 0 y 3.
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"No encontre el CSV en: {csv_path}\n"
            "Revisa la ruta. Ejemplo: --csv \"results/cheating_dataset_results.csv\""
        )

    df = pd.read_csv(csv_path)

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"No existe la columna objetivo '{TARGET_COL}'.\n"
            f"Columnas disponibles: {list(df.columns)}"
        )

    # Objetivo original para entrenamiento.
    y_original = pd.to_numeric(df[TARGET_COL], errors="coerce")

    if y_original.isna().any():
        bad_rows = df[y_original.isna()].index.tolist()
        raise ValueError(
            f"Hay valores no numericos o vacios en {TARGET_COL}. Filas: {bad_rows[:20]}"
        )

    # IMPORTANTE: en esta version el clone_type 0 se queda como 0.
    # No hacemos conversiones especiales de 4 -> 0; conservamos los valores tal cual.
    y = y_original.astype(int)

    similarity_score = build_similarity_score(df)
    noisy_negative_mask = (y == 0) & similarity_score.notna() & (similarity_score >= NEGATIVE_SIMILARITY_THRESHOLD)
    removed_rows = int(noisy_negative_mask.sum())

    if removed_rows > 0:
        print(
            "Limpieza del dataset: se eliminaron "
            f"{removed_rows} pares clone_type=0 con similitud alta "
            f"(score >= {NEGATIVE_SIMILARITY_THRESHOLD:.2f})."
        )
        df = df.loc[~noisy_negative_mask].copy()
        y = y.loc[~noisy_negative_mask].copy()
        y_original = y_original.loc[~noisy_negative_mask].copy()

    # Seleccion de features: numericas y booleanas, excluyendo identificadores/target.
    candidate_cols = []
    for col in df.columns:
        if col in DEFAULT_EXCLUDE_COLS:
            continue

        # Intentar convertir a numerico. Si tiene suficientes valores numericos, se usa.
        numeric_col = pd.to_numeric(df[col], errors="coerce")
        non_null_ratio = numeric_col.notna().mean()

        if non_null_ratio >= 0.80:
            candidate_cols.append(col)

    if not candidate_cols:
        raise ValueError(
            "No se encontraron columnas numericas para entrenar. "
            "Revisa que tu CSV tenga metricas numericas."
        )

    X = df[candidate_cols].copy()

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    return X, y, df


def build_preprocessors(feature_names: list[str]) -> Tuple[ColumnTransformer, ColumnTransformer]:
    """Crea preprocessors con escalado y sin escalado."""

    scaled_preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                feature_names,
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    unscaled_preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                feature_names,
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return scaled_preprocessor, unscaled_preprocessor


def build_models(feature_names: list[str], random_state: int = 42, class_weight_map: dict | None = None) -> Dict[str, Pipeline]:
    """Construye los modelos disponibles."""

    scaled_preprocessor, unscaled_preprocessor = build_preprocessors(feature_names)

    return {
        "linear_svc": Pipeline(
            steps=[
                ("preprocess", scaled_preprocessor),
                (
                    "model",
                    LinearSVC(
                        class_weight=class_weight_map if class_weight_map is not None else "balanced",
                        C=1.0,
                        max_iter=50000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "logistic_regression": Pipeline(
            steps=[
                ("preprocess", scaled_preprocessor),
                (
                    "model",
                    LogisticRegression(
                        class_weight=class_weight_map if class_weight_map is not None else "balanced",
                        max_iter=5000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocess", unscaled_preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=200,
                        class_weight=class_weight_map if class_weight_map is not None else "balanced",
                        max_depth=None,
                        min_samples_leaf=2,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def evaluate_model(
    model_name: str,
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
    groups: np.ndarray | None = None,
    random_state: int = 42,
) -> dict:
    """Entrena/evalua un modelo con holdout y cross-validation."""

    labels = sorted(y.unique().tolist())

    if groups is not None:
        X_train, X_test, y_train, y_test = group_train_test_split(
            X,
            y,
            groups=groups,
            test_size=0.25,
            random_state=random_state,
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            stratify=y,
            random_state=random_state,
        )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_test,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    cv = make_group_cv(groups, n_splits=5)
    cv_folds = cv.n_splits

    cv_results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        groups=groups if isinstance(cv, GroupKFold) else None,
        scoring={
            "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
            "f1_macro": "f1_macro",
        },
        return_train_score=False,
    )

    results = {
        "model_name": model_name,
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "features": list(X.columns),
        "class_distribution_after_mapping": {
            str(k): int(v) for k, v in y.value_counts().sort_index().items()
        },
        "holdout": {
            "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "labels_order": labels,
        },
        "cv": {
            "accuracy_mean": float(cv_results["test_accuracy"].mean()),
            "accuracy_std": float(cv_results["test_accuracy"].std()),
            "balanced_accuracy_mean": float(cv_results["test_balanced_accuracy"].mean()),
            "balanced_accuracy_std": float(cv_results["test_balanced_accuracy"].std()),
            "f1_macro_mean": float(cv_results["test_f1_macro"].mean()),
            "f1_macro_std": float(cv_results["test_f1_macro"].std()),
            "accuracy_by_fold": cv_results["test_accuracy"].tolist(),
            "balanced_accuracy_by_fold": cv_results["test_balanced_accuracy"].tolist(),
            "f1_macro_by_fold": cv_results["test_f1_macro"].tolist(),
        },
    }

    print("\n" + "=" * 80)
    print(f"Modelo: {model_name}")
    print("=" * 80)
    print(f"Filas: {len(X)}")
    print(f"Features usadas: {X.shape[1]}")
    print("Distribucion de clone_type usado para entrenamiento:")
    print({int(k): int(v) for k, v in y.value_counts().sort_index().items()})

    print("\nHoldout test")
    print(f"Accuracy: {acc:.3f}")
    print(f"Balanced accuracy: {bal_acc:.3f}")
    print("\nClassification report:")
    print(report_text)
    print("Matriz de confusion")
    print(f"Orden de etiquetas: {labels}")
    print(cm)

    print(f"\nCross-validation {cv_folds} folds")
    print(
        f"Accuracy: {results['cv']['accuracy_mean']:.3f} "
        f"+- {results['cv']['accuracy_std']:.3f}"
    )
    print(
        f"Balanced accuracy: {results['cv']['balanced_accuracy_mean']:.3f} "
        f"+- {results['cv']['balanced_accuracy_std']:.3f}"
    )
    print(
        f"F1 macro: {results['cv']['f1_macro_mean']:.3f} "
        f"+- {results['cv']['f1_macro_std']:.3f}"
    )

    # Guardar reporte JSON individual.
    report_path = output_dir / f"{model_name}_clone_type_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(to_json_safe(results), f, indent=2, ensure_ascii=False)

    cv_predictions = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        groups=groups if isinstance(cv, GroupKFold) else None,
    )
    cv_report = classification_report(
        y,
        cv_predictions,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    class_0_precision = float(cv_report.get("0", {}).get("precision", 0.0))
    class_3_recall = float(cv_report.get("3", {}).get("recall", 0.0))
    priority_score = (class_0_precision + class_3_recall) / 2.0

    results["cv"]["class_0_precision"] = class_0_precision
    results["cv"]["class_3_recall"] = class_3_recall
    results["cv"]["priority_score"] = priority_score

    print(
        f"Prioridad clase 0/3: precision_0={class_0_precision:.3f}, "
        f"recall_3={class_3_recall:.3f}, score={priority_score:.3f}"
    )

    return results


def save_cv_errors(
    model_name: str,
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    original_df: pd.DataFrame,
    output_dir: Path,
    groups: np.ndarray | None = None,
    random_state: int = 42,
) -> Path:
    """Guarda errores de cross-validation para un modelo especifico."""

    cv = make_group_cv(groups, n_splits=5)

    y_pred_cv = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        groups=groups if isinstance(cv, GroupKFold) else None,
    )
    error_mask = y_pred_cv != y.values

    error_df = pd.DataFrame()
    error_df["row_index"] = original_df.index

    for col in PAIR_INFO_CANDIDATES:
        if col in original_df.columns:
            error_df[col] = original_df[col].values

    if TARGET_COL in original_df.columns:
        error_df["clone_type_original"] = original_df[TARGET_COL].values

    error_df[f"{model_name}_pred"] = y_pred_cv
    error_df[f"{model_name}_acerto"] = y_pred_cv == y.values

    error_df = error_df[error_mask].copy()

    output_path = output_dir / f"{model_name}_cv_errors.csv"
    error_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"- Errores CV {model_name}: {output_path}")

    if len(error_df) == 0:
        print(f"  No hubo errores en CV para {model_name}.")
    else:
        print(f"  Errores encontrados en CV para {model_name}: {len(error_df)}")

    return output_path


def save_cv_predictions_by_pair(
    models: Dict[str, Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
    original_df: pd.DataFrame,
    output_dir: Path,
    groups: np.ndarray | None = None,
    random_state: int = 42,
) -> Path:
    """
    Guarda predicciones por par usando cross-validation.

    Importante:
        Cada prediccion se hace con un modelo que NO vio esa fila durante entrenamiento.
        No se guarda clone_type_real_mapeado porque el usuario pidio quitarlo.
    """

    cv = make_group_cv(groups, n_splits=5)

    results_df = pd.DataFrame()
    results_df["row_index"] = original_df.index

    for col in PAIR_INFO_CANDIDATES:
        if col in original_df.columns:
            results_df[col] = original_df[col].values

    if TARGET_COL in original_df.columns:
        results_df["clone_type_original"] = original_df[TARGET_COL].values

    for model_name, model in models.items():
        y_pred = cross_val_predict(
            model,
            X,
            y,
            cv=cv,
            groups=groups if isinstance(cv, GroupKFold) else None,
        )
        results_df[f"{model_name}_pred"] = y_pred
        results_df[f"{model_name}_acerto"] = y_pred == y.values

    output_path = output_dir / "predicciones_por_par_cv.csv"
    results_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"- Predicciones por par CV: {output_path}")
    return output_path


def get_feature_names_from_pipeline(model: Pipeline, X: pd.DataFrame) -> list[str]:
    """Obtiene nombres de features despues del preprocesamiento."""
    preprocess = model.named_steps.get("preprocess")

    if preprocess is None:
        return list(X.columns)

    try:
        return list(preprocess.get_feature_names_out())
    except Exception:
        return list(X.columns)


def save_linear_svc_feature_importance(
    fitted_model: Pipeline,
    X: pd.DataFrame,
    output_dir: Path,
    top_n: int = 10,
) -> Tuple[Path | None, Path | None]:
    """
    Guarda importancia de features por clase para LinearSVC.

    Para LinearSVC:
        - coefficient positivo: empuja hacia esa clase.
        - coefficient negativo: aleja de esa clase.
        - importance_abs: fuerza del peso sin importar signo.
    """

    clf = fitted_model.named_steps.get("model")

    if clf is None or not hasattr(clf, "coef_"):
        print("- LinearSVC feature importance: no disponible.")
        return None, None

    feature_names = get_feature_names_from_pipeline(fitted_model, X)
    classes = list(clf.classes_)

    rows = []
    for class_index, class_label in enumerate(classes):
        coefficients = clf.coef_[class_index]

        for feature, coef in zip(feature_names, coefficients):
            rows.append(
                {
                    "clone_type": int(class_label),
                    "feature": feature,
                    "coefficient": float(coef),
                    "importance_abs": float(abs(coef)),
                    "interpretation": "empuja_hacia_la_clase" if coef > 0 else "aleja_de_la_clase",
                }
            )

    importance_df = pd.DataFrame(rows).sort_values(
        by=["clone_type", "importance_abs"],
        ascending=[True, False],
    )

    full_path = output_dir / "linear_svc_feature_importance_by_class.csv"
    importance_df.to_csv(full_path, index=False, encoding="utf-8-sig")

    top_df = (
        importance_df.sort_values(
            by=["clone_type", "importance_abs"],
            ascending=[True, False],
        )
        .groupby("clone_type", as_index=False)
        .head(top_n)
    )

    top_path = output_dir / "linear_svc_top_features_by_class.csv"
    top_df.to_csv(top_path, index=False, encoding="utf-8-sig")

    print(f"- Importancia LinearSVC completa: {full_path}")
    print(f"- Top {top_n} features LinearSVC por clase: {top_path}")

    return full_path, top_path


def save_random_forest_feature_importance(
    fitted_model: Pipeline,
    X: pd.DataFrame,
    output_dir: Path,
    top_n: int = 15,
) -> Tuple[Path | None, Path | None]:
    """
    Guarda importancia global de features para Random Forest.

    Nota:
        Random Forest no da importancia por clase de forma directa con feature_importances_.
        Da importancia global para todo el modelo.
    """

    clf = fitted_model.named_steps.get("model")

    if clf is None or not hasattr(clf, "feature_importances_"):
        print("- Random Forest feature importance: no disponible.")
        return None, None

    feature_names = get_feature_names_from_pipeline(fitted_model, X)
    importances = clf.feature_importances_

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values(by="importance", ascending=False)

    full_path = output_dir / "random_forest_feature_importance.csv"
    importance_df.to_csv(full_path, index=False, encoding="utf-8-sig")

    top_path = output_dir / "random_forest_top_features.csv"
    importance_df.head(top_n).to_csv(top_path, index=False, encoding="utf-8-sig")

    print(f"- Importancia Random Forest completa: {full_path}")
    print(f"- Top {top_n} features Random Forest: {top_path}")

    return full_path, top_path


def save_model_comparison_csv(comparison: dict, output_dir: Path) -> Path:
    """Guarda una tabla resumida de comparacion de modelos."""

    rows = []
    for model_name, result in comparison.items():
        rows.append(
            {
                "model": model_name,
                "holdout_accuracy": result["holdout"]["accuracy"],
                "holdout_balanced_accuracy": result["holdout"]["balanced_accuracy"],
                "cv_accuracy_mean": result["cv"]["accuracy_mean"],
                "cv_accuracy_std": result["cv"]["accuracy_std"],
                "cv_balanced_accuracy_mean": result["cv"]["balanced_accuracy_mean"],
                "cv_balanced_accuracy_std": result["cv"]["balanced_accuracy_std"],
                "cv_f1_macro_mean": result["cv"]["f1_macro_mean"],
                "cv_f1_macro_std": result["cv"]["f1_macro_std"],
                "cv_class_0_precision": result["cv"].get("class_0_precision", 0.0),
                "cv_class_3_recall": result["cv"].get("class_3_recall", 0.0),
                "cv_priority_score": result["cv"].get("priority_score", 0.0),
                "cv_priority_minus_std": (
                    result["cv"].get("priority_score", 0.0)
                    - (DEFAULT_STABILITY_PENALTY * result["cv"]["f1_macro_std"])
                ),
            }
        )

    comparison_df = pd.DataFrame(rows).sort_values(
        by="cv_priority_minus_std",
        ascending=False,
    )

    output_path = output_dir / "model_comparison.csv"
    comparison_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"- Comparacion CSV: {output_path}")
    return output_path


def main() -> None:
    output_dir = Path(DEFAULT_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y, original_df = load_dataset(DEFAULT_CSV)
    groups = get_pair_group_ids(original_df)

    # Construir mapa de pesos por clase si el usuario pide mayor peso para clase 3.
    class_weight_map = None
    if DEFAULT_CLASS3_WEIGHT != 1.0:
        unique_labels = sorted(y.unique().tolist())
        # Default 1.0 para todas las clases, luego ajustar 3 si existe.
        class_weight_map = {int(lbl): 1.0 for lbl in unique_labels}
        if 3 in class_weight_map:
            class_weight_map[3] = float(DEFAULT_CLASS3_WEIGHT)
        else:
            # Si la etiqueta 3 no existe en y, no hacer nada especial.
            class_weight_map = None

    models = build_models(
        feature_names=list(X.columns),
        random_state=DEFAULT_RANDOM_STATE,
        class_weight_map=class_weight_map,
    )

    models_to_run = models if DEFAULT_COMPARE_MODE else {"linear_svc": models["linear_svc"]}

    comparison = {}

    for model_name, model in models_to_run.items():
        result = evaluate_model(
            model_name=model_name,
            model=model,
            X=X,
            y=y,
            output_dir=output_dir,
            groups=groups,
            random_state=DEFAULT_RANDOM_STATE,
        )
        comparison[model_name] = result

    # Elegir mejor modelo por prioridad 0/3 penalizada por variabilidad CV
    # para reducir riesgo de sobreajuste.
    best_model_name = max(
        comparison,
        key=lambda name: (
            comparison[name]["cv"].get("priority_score", 0.0)
            - (DEFAULT_STABILITY_PENALTY * comparison[name]["cv"].get("f1_macro_std", 0.0))
        ),
    )

    print("\n" + "#" * 80)
    print(f"Mejor modelo por prioridad 0/3 con penalizacion de inestabilidad: {best_model_name}")
    print(
        f"Score prioridad CV: {comparison[best_model_name]['cv'].get('priority_score', 0.0):.3f} "
        f"| score penalizado: {comparison[best_model_name]['cv'].get('priority_score', 0.0) - (DEFAULT_STABILITY_PENALTY * comparison[best_model_name]['cv'].get('f1_macro_std', 0.0)):.3f} "
        f"(precision_0={comparison[best_model_name]['cv'].get('class_0_precision', 0.0):.3f}, "
        f"recall_3={comparison[best_model_name]['cv'].get('class_3_recall', 0.0):.3f})"
    )
    print("#" * 80)

    # Entrenar el mejor modelo con TODO el dataset para guardarlo como modelo final.
    best_model = models[best_model_name]
    best_model.fit(X, y)

    model_path = output_dir / f"{best_model_name}_clone_type_model.joblib"
    dump(best_model, model_path)

    features_path = output_dir / f"{best_model_name}_features.json"
    with features_path.open("w", encoding="utf-8") as f:
        json.dump(list(X.columns), f, indent=2, ensure_ascii=False)

    # Guardar comparacion completa.
    comparison_json_path = output_dir / "model_comparison.json"
    with comparison_json_path.open("w", encoding="utf-8") as f:
        json.dump(to_json_safe(comparison), f, indent=2, ensure_ascii=False)

    # Guardar comparacion resumida en CSV.
    save_model_comparison_csv(comparison, output_dir)

    # Guardar predicciones por par para TODOS los modelos disponibles, no solo el mejor.
    save_cv_predictions_by_pair(
        models=models,
        X=X,
        y=y,
        original_df=original_df,
        output_dir=output_dir,
        groups=groups,
        random_state=DEFAULT_RANDOM_STATE,
    )

    # Guardar errores CV del mejor modelo.
    save_cv_errors(
        model_name=best_model_name,
        model=models[best_model_name],
        X=X,
        y=y,
        original_df=original_df,
        output_dir=output_dir,
        groups=groups,
        random_state=DEFAULT_RANDOM_STATE,
    )

    # Guardar importancia de metricas/features para modelos interpretables.
    # LinearSVC: importancia por clase.
    linear_model = models["linear_svc"]
    linear_model.fit(X, y)
    save_linear_svc_feature_importance(
        fitted_model=linear_model,
        X=X,
        output_dir=output_dir,
        top_n=DEFAULT_TOP_N_FEATURES,
    )

    # Random Forest: importancia global.
    rf_model = models["random_forest"]
    rf_model.fit(X, y)
    save_random_forest_feature_importance(
        fitted_model=rf_model,
        X=X,
        output_dir=output_dir,
        top_n=max(DEFAULT_TOP_N_FEATURES, 15),
    )

    print("\nArchivos guardados:")
    print(f"- Modelo final: {model_path}")
    print(f"- Features: {features_path}")
    print(f"- Comparacion JSON: {comparison_json_path}")
    print(f"- Comparacion CSV: {output_dir / 'model_comparison.csv'}")
    print(f"- Predicciones por par CV: {output_dir / 'predicciones_por_par_cv.csv'}")
    print(f"- Errores CV mejor modelo: {output_dir / f'{best_model_name}_cv_errors.csv'}")
    print(f"- Top features LinearSVC por clase: {output_dir / 'linear_svc_top_features_by_class.csv'}")
    print(f"- Importancia Random Forest: {output_dir / 'random_forest_top_features.csv'}")


if __name__ == "__main__":
    main()
