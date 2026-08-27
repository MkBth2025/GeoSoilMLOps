from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

try:
    # Normal use: python -m src.evaluate_all_class
    from .evaluate_class import (
        evaluate_model_file,
        rank_evaluated_classifiers,
    )
except ImportError:
    # Direct-script fallback when both files are in the same folder.
    from evaluate_class import (
        evaluate_model_file,
        rank_evaluated_classifiers,
    )


def _load_params(path: str) -> dict:
    params_path = Path(path)
    if not params_path.exists():
        print(f"[EVAL_ALL_CLASS][WARN] Parameters file not found: {params_path}")
        return {}
    with params_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _is_classification_model(path: Path) -> bool:
    """Identify classifier files while excluding regression packages."""
    name = path.name.lower()

    if name.startswith("best_classifier"):
        return True

    classification_markers = (
        "_class.pkl",
        "_classifier.pkl",
        "_classification.pkl",
    )
    if any(name.endswith(marker) for marker in classification_markers):
        return True

    # trainclass model packages normally include "_class" in their filename.
    return "_class" in path.stem.lower()


def _normalize_class_names(value: Any) -> dict[int, str]:
    if isinstance(value, dict):
        out = {}
        for key, name in value.items():
            try:
                out[int(key)] = str(name)
            except (TypeError, ValueError):
                continue
        return out
    if isinstance(value, (list, tuple)):
        return {index: str(name) for index, name in enumerate(value)}
    return {}


def _class_labels_and_names(detail: dict) -> tuple[list[int], dict[int, str]]:
    metadata = detail.get("metadata", {}) or {}
    names = _normalize_class_names(metadata.get("class_names"))
    labels = set(names)
    for payload in (detail.get("predictions", {}) or {}).values():
        payload = payload or {}
        for key in ("y_true", "y_pred"):
            for value in payload.get(key) or []:
                try:
                    labels.add(int(value))
                except (TypeError, ValueError):
                    pass
    ordered = sorted(labels)
    if not ordered:
        ordered = [0, 1, 2]
    for label in ordered:
        names.setdefault(label, f"Class_{label}")
    return ordered, names


def _safe_probability_column(name: str, label: int) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(name)).strip("_")
    return f"prob_{cleaned or f'class_{label}'}"


def _prediction_rows(detail: dict) -> list[dict]:
    """Convert nested prediction details to a long, CSV-friendly table."""
    metadata = detail.get("metadata", {}) or {}
    predictions = detail.get("predictions", {}) or {}

    rows: list[dict] = []
    for split, payload in predictions.items():
        payload = payload or {}
        y_true = payload.get("y_true") or []
        y_pred = payload.get("y_pred") or []
        probabilities = payload.get("probabilities")

        n = min(len(y_true), len(y_pred))
        for index in range(n):
            row = {
                "model": metadata.get("model"),
                "feature_set": metadata.get("feature_set"),
                "target": metadata.get("target"),
                "file": metadata.get("file"),
                "split": split,
                "row_index": index,
                "y_true": y_true[index],
                "y_pred": y_pred[index],
                "correct": int(y_true[index] == y_pred[index]),
            }

            if probabilities is not None and index < len(probabilities):
                current = probabilities[index]
                if isinstance(current, (list, tuple)):
                    class_labels, class_names = _class_labels_and_names(detail)
                    for probability_index, probability in enumerate(current):
                        if probability_index >= len(class_labels):
                            break
                        label = class_labels[probability_index]
                        column = _safe_probability_column(class_names[label], label)
                        row[column] = probability
                    if len(current):
                        row["predicted_confidence"] = float(
                            np.max(np.asarray(current, dtype=float))
                        )

            rows.append(row)

    return rows


def _class_report_rows(detail: dict) -> list[dict]:
    """Flatten per-class precision, recall, F1, and support."""
    metadata = detail.get("metadata", {}) or {}
    metrics = detail.get("metrics", {}) or {}

    rows: list[dict] = []
    for split, split_metrics in metrics.items():
        report = (split_metrics or {}).get("classification_report", {}) or {}
        _, class_names = _class_labels_and_names(detail)
        for class_name in class_names.values():
            values = report.get(class_name, {}) or {}
            rows.append(
                {
                    "model": metadata.get("model"),
                    "feature_set": metadata.get("feature_set"),
                    "target": metadata.get("target"),
                    "file": metadata.get("file"),
                    "split": split,
                    "class_name": class_name,
                    "precision": values.get("precision", np.nan),
                    "recall": values.get("recall", np.nan),
                    "f1_score": values.get("f1-score", np.nan),
                    "support": values.get("support", np.nan),
                }
            )
    return rows


def _confusion_rows(detail: dict) -> list[dict]:
    """Flatten confusion matrices for every classifier and split."""
    metadata = detail.get("metadata", {}) or {}
    metrics = detail.get("metrics", {}) or {}
    class_labels, class_name_map = _class_labels_and_names(detail)
    class_names = [class_name_map[label] for label in class_labels]

    rows: list[dict] = []
    for split, split_metrics in metrics.items():
        matrix = (split_metrics or {}).get("confusion_matrix", [])
        matrix = np.asarray(matrix)
        expected = len(class_names)
        if matrix.shape != (expected, expected):
            continue

        for true_index, true_name in enumerate(class_names):
            for pred_index, pred_name in enumerate(class_names):
                rows.append(
                    {
                        "model": metadata.get("model"),
                        "feature_set": metadata.get("feature_set"),
                        "target": metadata.get("target"),
                        "file": metadata.get("file"),
                        "split": split,
                        "true_class": true_name,
                        "predicted_class": pred_name,
                        "count": int(matrix[true_index, pred_index]),
                    }
                )
    return rows


def _best_per_feature_set(ranked: pd.DataFrame) -> pd.DataFrame:
    if ranked.empty or "feature_set" not in ranked.columns:
        return pd.DataFrame()

    ordered = ranked.sort_values(
        ["Selection Rank", "cv_macro_f1", "val_macro_f1"],
        ascending=[True, False, False],
        na_position="last",
    )
    return (
        ordered.groupby(["target", "feature_set"], dropna=False, as_index=False)
        .first()
        .sort_values(
            ["target", "feature_set", "Selection Rank"],
            ascending=[True, True, True],
        )
        .reset_index(drop=True)
    )


def run(
    data_dir: str,
    models_dir: str,
    reports_dir: str,
    params_path: str = "params.yaml",
    out_name: str = "all_evaluations_class",
    skip_best: bool = True,
) -> None:
    data_path = Path(data_dir)
    models_path = Path(models_dir)
    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)
    params = _load_params(params_path)

    all_pickles = sorted(models_path.glob("*.pkl"))
    model_paths = [path for path in all_pickles if _is_classification_model(path)]

    if skip_best:
        model_paths = [
            path
            for path in model_paths
            if not path.name.lower().startswith("best")
        ]

    if not model_paths:
        available = ", ".join(path.name for path in all_pickles)
        print(
            "[EVAL_ALL_CLASS][WARN] No classification model files were found. "
            "Evaluation will terminate cleanly. "
            f"Available pickle files: {available or 'none'}"
        )
        return

    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    class_report_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for index, model_path in enumerate(model_paths, start=1):
        print(
            f"[EVAL_ALL_CLASS] {index}/{len(model_paths)}: "
            f"{model_path.name}"
        )
        try:
            row, detail = evaluate_model_file(
                model_path=model_path,
                data_dir=data_path,
                params=params,
            )
            rows.append(row)
            details.append(detail)
            prediction_rows.extend(_prediction_rows(detail))
            class_report_rows.extend(_class_report_rows(detail))
            confusion_rows.extend(_confusion_rows(detail))
        except Exception as error:
            print(
                f"[EVAL_ALL_CLASS][WARN] {model_path.name}: {error}"
            )
            failures.append(
                {
                    "file": model_path.name,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )

    if not rows:
        failure_text = "; ".join(
            f"{item['file']}: {item['error']}" for item in failures
        )
        print(
            "[EVAL_ALL_CLASS][WARN] No classifiers were evaluated successfully. "
            "Evaluation will terminate cleanly. "
            f"Failures: {failure_text or 'unknown'}"
        )

        # Preserve a failure report when models were discovered but all failed.
        failures_path = reports_path / "classification_evaluation_failures.csv"
        pd.DataFrame(
            failures,
            columns=["file", "error_type", "error"],
        ).to_csv(
            failures_path,
            index=False,
            encoding="utf-8-sig",
        )

        print(f"[EVAL_ALL_CLASS] Failures: {failures_path}")
        return

    raw_df = pd.DataFrame(rows)
    ranked_df = rank_evaluated_classifiers(raw_df)
    best_per_fs_df = _best_per_feature_set(ranked_df)

    summary_path = reports_path / f"{out_name}.csv"
    json_path = reports_path / f"{out_name}.json"
    ranking_path = reports_path / "classification_evaluation_ranking.csv"
    paper_path = reports_path / "Results_classification_evaluation_summary.csv"
    best_per_fs_path = reports_path / "best_classifier_per_featureset_evaluation.csv"
    predictions_path = reports_path / "classification_predictions_long.csv"
    class_reports_path = reports_path / "classification_per_class_metrics.csv"
    confusion_path = reports_path / "classification_confusion_matrices_long.csv"
    failures_path = reports_path / "classification_evaluation_failures.csv"
    metadata_path = reports_path / "evaluate_all_class_metadata.json"

    ranked_df.to_csv(
        summary_path,
        index=False,
        float_format="%.6f",
        encoding="utf-8-sig",
    )
    ranked_df.to_csv(
        ranking_path,
        index=False,
        float_format="%.6f",
        encoding="utf-8-sig",
    )
    ranked_df.to_json(
        json_path,
        orient="records",
        indent=2,
        force_ascii=False,
    )

    paper_columns = [
        "Selection Rank",
        "Test Performance Rank",
        "Generalization Rank",
        "target",
        "model",
        "feature_set",
        "train_macro_f1",
        "cv_macro_f1",
        "cv_macro_f1_std",
        "val_macro_f1",
        "test_macro_f1",
        "train_balanced_accuracy",
        "val_balanced_accuracy",
        "test_balanced_accuracy",
        "train_cv_gap",
        "train_val_gap",
        "cv_val_gap",
        "val_test_gap",
        "abs_train_cv_gap",
        "abs_cv_val_gap",
        "generalization_diagnosis",
        "file",
    ]
    paper_df = ranked_df[
        [column for column in paper_columns if column in ranked_df.columns]
    ].copy()
    paper_df.to_csv(
        paper_path,
        index=False,
        float_format="%.6f",
        encoding="utf-8-sig",
    )

    best_per_fs_df.to_csv(
        best_per_fs_path,
        index=False,
        float_format="%.6f",
        encoding="utf-8-sig",
    )
    pd.DataFrame(prediction_rows).to_csv(
        predictions_path,
        index=False,
        float_format="%.6f",
        encoding="utf-8-sig",
    )
    pd.DataFrame(class_report_rows).to_csv(
        class_reports_path,
        index=False,
        float_format="%.6f",
        encoding="utf-8-sig",
    )
    pd.DataFrame(confusion_rows).to_csv(
        confusion_path,
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        failures,
        columns=["file", "error_type", "error"],
    ).to_csv(
        failures_path,
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "data_dir": str(data_path),
        "models_dir": str(models_path),
        "reports_dir": str(reports_path),
        "params_file": params_path,
        "models_discovered": len(model_paths),
        "models_evaluated_successfully": len(ranked_df),
        "models_failed": len(failures),
        "skip_best_aliases": skip_best,
        "cross_validation_uses_training_partition_only": True,
        "validation_and_test_cv_performed": False,
        "test_metrics_used_for_selection": False,
        "selection_weights": {
            "cv_macro_f1": 0.40,
            "validation_macro_f1": 0.25,
            "cv_macro_f1_std": 0.15,
            "absolute_train_cv_gap": 0.15,
            "validation_balanced_error": 0.05,
        },
        "class_definitions": [
            {
                "target": detail.get("metadata", {}).get("target"),
                "class_names": detail.get("metadata", {}).get("class_names"),
                "class_boundaries": detail.get("metadata", {}).get("class_boundaries"),
            }
            for detail in details
        ],
        "outputs": [
            summary_path.name,
            json_path.name,
            ranking_path.name,
            paper_path.name,
            best_per_fs_path.name,
            predictions_path.name,
            class_reports_path.name,
            confusion_path.name,
            failures_path.name,
        ],
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    print(f"[EVAL_ALL_CLASS] Summary: {summary_path}")
    print(f"[EVAL_ALL_CLASS] Ranking: {ranking_path}")
    print(f"[EVAL_ALL_CLASS] Paper table: {paper_path}")
    print(f"[EVAL_ALL_CLASS] Best per feature set: {best_per_fs_path}")
    print(f"[EVAL_ALL_CLASS] Predictions: {predictions_path}")
    print(f"[EVAL_ALL_CLASS] Per-class metrics: {class_reports_path}")
    print(f"[EVAL_ALL_CLASS] Confusion matrices: {confusion_path}")
    print(f"[EVAL_ALL_CLASS] Failures: {failures_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate all classification models using "
            "training-only repeated/grouped CV and separate selection, test, "
            "and generalization rankings."
        )
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--models_dir", required=True)
    parser.add_argument("--reports_dir", required=True)
    parser.add_argument(
        "--params",
        default="params.yaml",
        help="Parameter file; defaults to params.yaml.",
    )
    parser.add_argument(
        "--out_name",
        default="all_evaluations_class",
        help=(
            "Base filename for the complete CSV and JSON reports. "
            "Defaults to all_evaluations_class."
        ),
    )
    parser.add_argument(
        "--include_best",
        action="store_true",
        help=(
            "Also evaluate copied best*.pkl aliases. They are skipped by "
            "default because they duplicate ordinary classifier packages."
        ),
    )
    args = parser.parse_args()

    run(
        data_dir=args.data,
        models_dir=args.models_dir,
        reports_dir=args.reports_dir,
        params_path=args.params,
        out_name=args.out_name,
        skip_best=not args.include_best,
    )


if __name__ == "__main__":
    main()
