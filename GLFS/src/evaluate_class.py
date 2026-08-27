from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GroupKFold,
    RepeatedStratifiedKFold,
    cross_val_score,
)


CLASS_NAMES = {0: "Inactive", 1: "Normal", 2: "Active"}
CLASS_LABELS = [0, 1, 2]


def _load_params(path: str) -> dict:
    params_path = Path(path)
    if not params_path.exists():
        print(f"[EVAL_CLASS][WARN] Parameters file not found: {params_path}")
        return {}
    with params_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _ensure_2d(X: Any) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix, got shape {X.shape}")
    return X


def _ensure_1d(y: Any) -> np.ndarray:
    return np.asarray(y).ravel()


def target_to_three_classes(values: Any, boundaries: Any) -> np.ndarray:
    """Convert a continuous target to classes using saved model boundaries."""
    if not isinstance(boundaries, dict):
        raise ValueError("Classification model package does not contain class boundaries.")
    lower = boundaries.get("lower")
    upper = boundaries.get("upper")
    try:
        lower = float(lower)
        upper = float(upper)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid saved class boundaries: {boundaries!r}") from error
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError(f"Invalid saved class boundaries: {boundaries!r}")
    values = np.asarray(values, dtype=float).ravel()
    classes = np.ones(values.shape, dtype=int)
    classes[values < lower] = 0
    classes[values > upper] = 2
    return classes




def encode_saved_target(values: Any, package: dict) -> np.ndarray:
    """Encode raw target values according to metadata saved with the classifier."""
    info = package.get("class_boundary_values") or {}
    mode = str(package.get("classification_mode") or info.get("mode") or "threshold").lower()
    if mode == "categorical":
        class_values = [str(v) for v in info.get("class_values", [])]
        if len(class_values) < 2:
            raise ValueError("Categorical classifier package is missing class_values metadata.")
        mapping = {value: i for i, value in enumerate(class_values)}
        series = pd.Series(_ensure_1d(values))
        text = series.astype(str)
        unknown = sorted(set(text.unique()) - set(mapping))
        if unknown:
            raise ValueError(f"Evaluation data contain unseen class labels: {unknown}")
        return text.map(mapping).to_numpy(dtype=int)
    boundaries = info if isinstance(info, dict) and "lower" in info and "upper" in info else package.get("class_boundaries")
    return target_to_three_classes(values, boundaries)


def activate_package_classes(package: dict) -> None:
    """Set metric labels/names from a saved model package."""
    global CLASS_LABELS, CLASS_NAMES
    info = package.get("class_boundary_values") or {}
    if str(package.get("classification_mode") or info.get("mode") or "").lower() == "categorical":
        class_values = [str(v) for v in info.get("class_values", [])]
        CLASS_LABELS = list(range(len(class_values)))
        saved_names = package.get("class_names") or {}
        CLASS_NAMES = {i: str(saved_names.get(i, saved_names.get(str(i), class_values[i]))) for i in CLASS_LABELS}
    else:
        CLASS_LABELS = [0, 1, 2]
        saved_names = package.get("class_names") or CLASS_NAMES
        CLASS_NAMES = {i: str(saved_names.get(i, saved_names.get(str(i), CLASS_NAMES.get(i, i)))) for i in CLASS_LABELS}

def parse_model_filename(filename: str) -> tuple[str | None, str | None, str | None]:
    stem = Path(filename).stem
    if stem.startswith("best"):
        return "best", None, None
    if stem.endswith("_class"):
        stem = stem[:-len("_class")]
    parts = stem.split("_")
    if len(parts) < 3:
        return None, None, None
    return parts[0], parts[1], "_".join(parts[2:])


def load_model_package(path: Path) -> dict:
    loaded = joblib.load(path)

    if isinstance(loaded, dict):
        model = loaded.get("model")
        if model is None:
            model = next((v for v in loaded.values() if hasattr(v, "predict")), None)
        if model is None:
            raise ValueError(f"No classifier was found in {path.name}")
        return {
            "model": model,
            "model_type": loaded.get("model_type"),
            "feature_set": loaded.get("feature_set") or loaded.get("fs"),
            "target": loaded.get("target"),
            "feature_names": list(loaded.get("feature_names") or []),
            "class_names": loaded.get("class_names", CLASS_NAMES),
            "class_boundaries": loaded.get("class_boundaries", {}),
            "class_boundary_values": loaded.get("class_boundary_values", {}),
            "classification_mode": loaded.get("classification_mode", (loaded.get("class_boundary_values", {}) or {}).get("mode", "threshold")),
            "best_params": loaded.get("best_params", {}),
            "cv_train_macro_f1": loaded.get("cv_train_macro_f1"),
            "cv_train_macro_f1_std": loaded.get("cv_train_macro_f1_std"),
            "cv_macro_f1": loaded.get("cv_macro_f1"),
            "cv_macro_f1_std": loaded.get("cv_macro_f1_std"),
            "pipeline_contains_preprocessing": bool(
                loaded.get("pipeline_contains_preprocessing", hasattr(model, "steps"))
            ),
            "raw_metadata": loaded,
        }

    return {
        "model": loaded,
        "model_type": type(loaded).__name__,
        "feature_set": None,
        "target": None,
        "feature_names": [],
        "class_names": CLASS_NAMES,
        "class_boundaries": {},
        "class_boundary_values": {},
        "classification_mode": "threshold",
        "best_params": {},
        "cv_train_macro_f1": None,
        "cv_train_macro_f1_std": None,
        "cv_macro_f1": None,
        "cv_macro_f1_std": None,
        "pipeline_contains_preprocessing": hasattr(loaded, "steps"),
        "raw_metadata": {},
    }


def validate_feature_count(X: np.ndarray, package: dict, split: str) -> np.ndarray:
    X = _ensure_2d(X)
    expected = len(package["feature_names"])
    if not expected:
        expected = getattr(package["model"], "n_features_in_", None)
    if expected is not None and X.shape[1] != int(expected):
        raise ValueError(
            f"{split} feature mismatch: classifier expects {expected} columns, "
            f"but data contain {X.shape[1]}. Feature padding/truncation is disabled."
        )
    return X


def _softmax(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    scores = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    denom = np.sum(exp_scores, axis=1, keepdims=True)
    denom[denom <= 0] = 1.0
    return exp_scores / denom


def predict_probabilities(model: Any, X: np.ndarray) -> np.ndarray | None:
    """Return probability-like scores aligned to activity classes [0, 1, 2].

    True predict_proba output is preferred. When unavailable, decision_function
    scores are converted with a softmax so ROC/AUC can still use continuous
    class scores. Test labels are never used to construct these scores.
    """
    X = _ensure_2d(X)
    classes = np.asarray(getattr(model, "classes_", CLASS_LABELS))

    raw = None
    if hasattr(model, "predict_proba"):
        try:
            raw = np.asarray(model.predict_proba(X), dtype=float)
        except Exception:
            raw = None

    if raw is None and hasattr(model, "decision_function"):
        try:
            decision = np.asarray(model.decision_function(X), dtype=float)
            if decision.ndim == 1:
                decision = np.column_stack([-decision, decision])
            raw = _softmax(decision)
        except Exception:
            raw = None

    if raw is None or raw.ndim != 2:
        return None

    if len(classes) != raw.shape[1]:
        # Most multiclass estimators expose classes_. If they do not, only
        # accept the conventional 3-column order.
        if raw.shape[1] == len(CLASS_LABELS):
            classes = np.asarray(CLASS_LABELS)
        else:
            return None

    aligned = np.zeros((raw.shape[0], len(CLASS_LABELS)), dtype=float)
    for source_index, class_value in enumerate(classes):
        try:
            label_index = CLASS_LABELS.index(int(class_value))
        except (ValueError, TypeError):
            continue
        aligned[:, label_index] = raw[:, source_index]

    row_sums = aligned.sum(axis=1, keepdims=True)
    valid = np.isfinite(row_sums[:, 0]) & (row_sums[:, 0] > 0)
    if np.any(valid):
        aligned[valid] /= row_sums[valid]
    if np.any(~valid):
        aligned[~valid] = 1.0 / len(CLASS_LABELS)

    return aligned


def safe_metrics(y_true: Any, y_pred: Any, probabilities: np.ndarray | None = None) -> dict:
    y_true = _ensure_1d(y_true)
    y_pred = _ensure_1d(y_pred)
    n = min(len(y_true), len(y_pred))
    y_true = y_true[:n]
    y_pred = y_pred[:n]

    mask = np.isfinite(y_true.astype(float)) & np.isfinite(y_pred.astype(float))
    y_true = y_true[mask].astype(int)
    y_pred = y_pred[mask].astype(int)

    metric_names = [
        "accuracy", "balanced_accuracy", "macro_precision", "macro_recall",
        "macro_f1", "weighted_f1", "mcc", "cohen_kappa", "log_loss",
        "roc_auc_ovr_macro", "roc_auc_ovr_weighted",
    ]
    if len(y_true) == 0:
        result = {name: np.nan for name in metric_names}
        result["confusion_matrix"] = []
        result["classification_report"] = {}
        return result

    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=CLASS_LABELS, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=CLASS_LABELS, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=CLASS_LABELS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=CLASS_LABELS, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "log_loss": np.nan,
        "roc_auc_ovr_macro": np.nan,
        "roc_auc_ovr_weighted": np.nan,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=CLASS_LABELS).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=CLASS_LABELS,
            target_names=[CLASS_NAMES[label] for label in CLASS_LABELS],
            output_dict=True, zero_division=0,
        ),
    }

    if probabilities is not None:
        probabilities = np.asarray(probabilities, dtype=float)[:n]
        probabilities = probabilities[mask]
        if probabilities.ndim == 2 and probabilities.shape[1] == len(CLASS_LABELS):
            try:
                result["log_loss"] = float(log_loss(y_true, probabilities, labels=CLASS_LABELS))
            except Exception:
                pass
            try:
                result["roc_auc_ovr_macro"] = float(
                    roc_auc_score(y_true, probabilities, labels=CLASS_LABELS, multi_class="ovr", average="macro")
                )
                result["roc_auc_ovr_weighted"] = float(
                    roc_auc_score(y_true, probabilities, labels=CLASS_LABELS, multi_class="ovr", average="weighted")
                )
            except Exception:
                pass
    return result


def compute_generalization_diagnostics(row: dict) -> dict:
    def num(name: str) -> float:
        try:
            value = float(row.get(name, np.nan))
            return value if np.isfinite(value) else np.nan
        except (TypeError, ValueError):
            return np.nan

    train = num("train_macro_f1")
    cv_mean = num("cv_macro_f1")
    cv_std = num("cv_macro_f1_std")
    val = num("val_macro_f1")
    test = num("test_macro_f1")

    train_cv_gap = train - cv_mean if np.isfinite(train) and np.isfinite(cv_mean) else np.nan
    train_val_gap = train - val if np.isfinite(train) and np.isfinite(val) else np.nan
    cv_val_gap = cv_mean - val if np.isfinite(cv_mean) and np.isfinite(val) else np.nan
    val_test_gap = val - test if np.isfinite(val) and np.isfinite(test) else np.nan

    possible_underfitting = bool(
        np.isfinite(train) and np.isfinite(cv_mean) and np.isfinite(val)
        and train < 0.55 and cv_mean < 0.50 and val < 0.50
    )
    possible_overfitting = bool(np.isfinite(train_cv_gap) and train_cv_gap >= 0.15)
    high_cv_instability = bool(np.isfinite(cv_std) and cv_std >= 0.15)
    split_sensitive = bool(np.isfinite(cv_val_gap) and abs(cv_val_gap) >= 0.15)
    stable_generalization = bool(
        np.isfinite(train_cv_gap) and abs(train_cv_gap) <= 0.08
        and np.isfinite(cv_mean) and cv_mean >= 0.50
        and np.isfinite(cv_std) and cv_std < 0.15
    )

    if possible_underfitting:
        diagnosis = "Possible underfitting"
    elif possible_overfitting and high_cv_instability:
        diagnosis = "Overfitting with high CV instability"
    elif possible_overfitting:
        diagnosis = "Possible overfitting"
    elif high_cv_instability:
        diagnosis = "High CV instability"
    elif split_sensitive:
        diagnosis = "Split-sensitive"
    elif stable_generalization:
        diagnosis = "Stable generalization"
    else:
        diagnosis = "Moderate generalization"

    return {
        "train_cv_gap": train_cv_gap,
        "train_val_gap": train_val_gap,
        "cv_val_gap": cv_val_gap,
        "val_test_gap": val_test_gap,
        "abs_train_cv_gap": abs(train_cv_gap) if np.isfinite(train_cv_gap) else np.nan,
        "abs_cv_val_gap": abs(cv_val_gap) if np.isfinite(cv_val_gap) else np.nan,
        "abs_val_test_gap": abs(val_test_gap) if np.isfinite(val_test_gap) else np.nan,
        "possible_overfitting": possible_overfitting,
        "possible_underfitting": possible_underfitting,
        "high_cv_instability": high_cv_instability,
        "split_sensitive": split_sensitive,
        "stable_generalization": stable_generalization,
        "generalization_diagnosis": diagnosis,
    }


def training_cv_scores(model: Any, X_train: np.ndarray, y_train: np.ndarray,
                       data_dir: Path, target: str, params: dict) -> tuple[np.ndarray, str, int, int]:
    """Calculate training-only grouped/repeated CV scores."""
    evaluation = params.get("evaluation", {}) or {}
    classification = params.get("classification", {}) or {}

    random_state = int(classification.get("random_state", evaluation.get("random_state", 42)))
    folds_requested = int(classification.get("cv_splits", evaluation.get("cv_splits", params.get("cv_splits", 5))))
    repeats = max(1, int(classification.get("repeated_cv_repeats", evaluation.get("repeated_cv_repeats", 3))))

    class_counts = np.bincount(y_train.astype(int), minlength=len(CLASS_LABELS))
    positive_counts = class_counts[class_counts > 0]
    if len(positive_counts) < 2:
        return np.array([], dtype=float), "unavailable", 0, 0

    max_stratified_folds = int(np.min(positive_counts))
    folds = min(folds_requested, max_stratified_folds)
    if folds < 2:
        return np.array([], dtype=float), "unavailable", folds, 0

    groups = None
    for group_path in (data_dir / f"groups_train_{target}.joblib", data_dir / "groups_train.joblib"):
        if group_path.exists():
            candidate = np.asarray(joblib.load(group_path)).ravel()
            if len(candidate) == len(y_train):
                groups = candidate
                break

    if groups is not None and len(np.unique(groups)) >= folds:
        cv = GroupKFold(n_splits=folds)
        mode = "group"
        repeats_used = 1
    else:
        groups = None
        cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=random_state)
        mode = "repeated_stratified_kfold"
        repeats_used = repeats

    scores = cross_val_score(
        clone(model), X_train, y_train, groups=groups, cv=cv,
        scoring="f1_macro", n_jobs=-1, error_score=np.nan,
    )
    scores = np.asarray(scores, dtype=float)
    return scores[np.isfinite(scores)], mode, folds, repeats_used


def auto_select_model_file(models_dir: Path, reports_dir: Path) -> str:
    for candidate_name in ("best_classifier.pkl", "best_overall_classifier.pkl", "best.pkl"):
        candidate = models_dir / candidate_name
        if candidate.exists():
            print(f"[EVAL_CLASS] Automatically selected classifier: {candidate_name}")
            return candidate_name

    ranking_candidates = [
        reports_dir / "ClassificationEnhancedFinalRanking.csv",
        reports_dir / "Classification_EnhancedFinalRanking.csv",
        reports_dir / "classification_evaluation_ranking.csv",
        reports_dir / "Results_classification_scientific_ranking.csv",
        reports_dir / "classification_ranking.csv",
        reports_dir / "classification_summary.csv",
    ]
    rank_columns = ["Recommended Rank", "recommended_rank", "Selection Rank", "selection_rank", "statistical_rank", "Overall Rank", "overall_rank"]

    for ranking_path in ranking_candidates:
        if not ranking_path.exists():
            continue
        try:
            ranking = pd.read_csv(ranking_path)
        except Exception as error:
            print(f"[EVAL_CLASS][WARN] Could not read {ranking_path}: {error}")
            continue
        if ranking.empty or "file" not in ranking.columns:
            continue
        rank_column = next((column for column in rank_columns if column in ranking.columns), None)
        if rank_column:
            ranking[rank_column] = pd.to_numeric(ranking[rank_column], errors="coerce")
            ranking = ranking.sort_values(rank_column, ascending=True, na_position="last")
        for value in ranking["file"].dropna().astype(str):
            filename = Path(value).name
            if filename.lower().startswith("best"):
                continue
            if (models_dir / filename).exists():
                print(f"[EVAL_CLASS] Automatically selected classifier from {ranking_path.name}: {filename}")
                return filename

    ordinary_models = sorted(path.name for path in models_dir.glob("*.pkl") if not path.name.lower().startswith("best"))
    if len(ordinary_models) == 1:
        return ordinary_models[0]
    available = ", ".join(path.name for path in sorted(models_dir.glob("*.pkl")))
    raise FileNotFoundError(
        "Could not automatically determine the classifier. Expected a best classifier "
        "or a ranking CSV with a valid file column. Available model files: "
        f"{available or 'none'}"
    )


def evaluate_model_file(model_path: Path, data_dir: Path, params: dict) -> tuple[dict, dict]:
    package = load_model_package(model_path)
    activate_package_classes(package)
    parsed_model, parsed_fs, parsed_target = parse_model_filename(model_path.name)

    model = package["model"]
    model_name = package["model_type"] or parsed_model or type(model).__name__
    feature_set = package["feature_set"] or parsed_fs
    target = package["target"] or parsed_target

    if not feature_set or not target:
        raise ValueError(
            f"Could not determine feature set and target for {model_path.name}. "
            "Use a trained classification model package containing metadata."
        )

    split_metrics: dict[str, dict] = {}
    split_predictions: dict[str, dict] = {}

    for split in ("train", "val", "test"):
        x_path = data_dir / f"X_{split}_{feature_set}.joblib"
        y_path = data_dir / f"y_{split}_{target}.joblib"

        if not x_path.exists() or not y_path.exists():
            split_metrics[split] = {}
            split_predictions[split] = {}
            continue

        X = validate_feature_count(joblib.load(x_path), package, split=split)
        y_raw = _ensure_1d(joblib.load(y_path))
        y_true = encode_saved_target(y_raw, package)
        y_pred = _ensure_1d(model.predict(X)).astype(int)
        probabilities = predict_probabilities(model, X)
        metrics = safe_metrics(y_true, y_pred, probabilities)

        split_metrics[split] = metrics
        split_predictions[split] = {
            "y_true": y_true.tolist(),
            "y_pred": y_pred.tolist(),
            "probabilities": probabilities.tolist() if probabilities is not None else None,
            "probability_source": (
                "predict_proba" if hasattr(model, "predict_proba")
                else "decision_function" if hasattr(model, "decision_function")
                else None
            ),
        }

    train_metrics = split_metrics.get("train", {})
    val_metrics = split_metrics.get("val", {})
    test_metrics = split_metrics.get("test", {})

    cv_scores = np.array([], dtype=float)
    cv_mode = "unavailable"
    cv_folds = 0
    cv_repeats = 0

    x_train_path = data_dir / f"X_train_{feature_set}.joblib"
    y_train_path = data_dir / f"y_train_{target}.joblib"
    if x_train_path.exists() and y_train_path.exists():
        X_train = validate_feature_count(joblib.load(x_train_path), package, split="train")
        y_train = encode_saved_target(joblib.load(y_train_path), package)
        cv_scores, cv_mode, cv_folds, cv_repeats = training_cv_scores(
            model, X_train, y_train, data_dir, target, params
        )

    if len(cv_scores):
        cv_macro_f1 = float(np.mean(cv_scores))
        cv_macro_f1_std = float(np.std(cv_scores, ddof=0))
    else:
        cv_macro_f1 = package.get("cv_macro_f1")
        cv_macro_f1_std = package.get("cv_macro_f1_std")

    row = {
        "model": model_name,
        "feature_set": feature_set,
        "fs": feature_set,
        "target": target,
        "file": model_path.name,
        "feature_count": len(package["feature_names"]),
        "feature_names": "|".join(package["feature_names"]),
        "classes": "|".join(str(CLASS_NAMES[i]) for i in CLASS_LABELS),
        "class_boundaries": json.dumps(package["class_boundaries"], ensure_ascii=False, sort_keys=True),
        "pipeline_contains_preprocessing": package["pipeline_contains_preprocessing"],
        "best_params": json.dumps(package["best_params"], ensure_ascii=False, sort_keys=True, default=str),
        "cv_mode": cv_mode,
        "cv_folds": cv_folds,
        "cv_repeats": cv_repeats,
        "cv_score_count": int(len(cv_scores)),
        "cv_macro_f1": cv_macro_f1,
        "cv_macro_f1_std": cv_macro_f1_std,
        "saved_cv_train_macro_f1": package.get("cv_train_macro_f1"),
        "saved_cv_train_macro_f1_std": package.get("cv_train_macro_f1_std"),
    }

    metric_names = [
        "accuracy", "balanced_accuracy", "macro_precision", "macro_recall",
        "macro_f1", "weighted_f1", "mcc", "cohen_kappa", "log_loss",
        "roc_auc_ovr_macro", "roc_auc_ovr_weighted",
    ]
    for split, metrics in split_metrics.items():
        for metric_name in metric_names:
            row[f"{split}_{metric_name}"] = metrics.get(metric_name, np.nan)
        row[f"n_{split}"] = len(split_predictions.get(split, {}).get("y_true", []))

    row["train_macro_f1"] = row.get("train_macro_f1", np.nan)
    row["val_macro_f1"] = row.get("val_macro_f1", np.nan)
    row["test_macro_f1"] = row.get("test_macro_f1", np.nan)
    row["val_balanced_accuracy"] = row.get("val_balanced_accuracy", np.nan)
    row["test_balanced_accuracy"] = row.get("test_balanced_accuracy", np.nan)
    row["val_balanced_error"] = 1.0 - row["val_balanced_accuracy"] if np.isfinite(row["val_balanced_accuracy"]) else np.nan
    row["test_balanced_error"] = 1.0 - row["test_balanced_accuracy"] if np.isfinite(row["test_balanced_accuracy"]) else np.nan

    row.update(compute_generalization_diagnostics(row))

    detail = {
        "metadata": {
            "model": model_name,
            "feature_set": feature_set,
            "target": target,
            "file": model_path.name,
            "feature_names": package["feature_names"],
            "class_names": package["class_names"],
            "classification_mode": package.get("classification_mode"),
            "class_boundaries": package["class_boundaries"],
            "class_boundary_values": package.get("class_boundary_values", {}),
            "best_params": package["best_params"],
            "pipeline_contains_preprocessing": package["pipeline_contains_preprocessing"],
        },
        "cross_validation": {
            "mode": cv_mode,
            "folds": cv_folds,
            "repeats": cv_repeats,
            "scores": cv_scores.tolist(),
            "macro_f1_mean": cv_macro_f1,
            "macro_f1_std": cv_macro_f1_std,
            "training_only": True,
        },
        "metrics": split_metrics,
        "predictions": split_predictions,
        "generalization": {key: row[key] for key in (
            "train_cv_gap", "train_val_gap", "cv_val_gap", "val_test_gap",
            "abs_train_cv_gap", "abs_cv_val_gap", "abs_val_test_gap",
            "possible_overfitting", "possible_underfitting", "high_cv_instability",
            "split_sensitive", "stable_generalization", "generalization_diagnosis",
        )},
    }
    return row, detail


def rank_evaluated_classifiers(df: pd.DataFrame) -> pd.DataFrame:
    """Create separate development-selection, test and generalization ranks."""
    if df is None or df.empty:
        return df
    ranked = df.copy()

    numeric_columns = [
        "cv_macro_f1", "cv_macro_f1_std", "val_macro_f1", "val_balanced_error",
        "test_macro_f1", "test_balanced_accuracy", "test_mcc",
        "abs_train_cv_gap", "abs_cv_val_gap",
    ]
    for column in numeric_columns:
        if column not in ranked.columns:
            ranked[column] = np.nan
        ranked[column] = pd.to_numeric(ranked[column], errors="coerce")

    def rank_column(column: str, ascending: bool) -> pd.Series:
        return ranked[column].rank(ascending=ascending, method="average", na_option="bottom")

    # Keep legacy ranking schema for compatibility with evaluate_all_class.py.
    ranked["rank_cv_macro_f1"] = rank_column("cv_macro_f1", False)
    ranked["rank_val_macro_f1"] = rank_column("val_macro_f1", False)
    ranked["rank_cv_stability"] = rank_column("cv_macro_f1_std", True)
    ranked["rank_train_cv_gap"] = rank_column("abs_train_cv_gap", True)
    ranked["rank_val_balanced_error"] = rank_column("val_balanced_error", True)

    # If validation is unavailable, do not let missing validation columns distort
    # the selection rank; use the same CV-only 60/20/20 fallback as predict_class.py.
    has_validation = ranked["val_macro_f1"].notna().any() and ranked["val_balanced_error"].notna().any()
    if has_validation:
        ranked["selection_rank_score"] = (
            0.40 * ranked["rank_cv_macro_f1"]
            + 0.25 * ranked["rank_val_macro_f1"]
            + 0.15 * ranked["rank_cv_stability"]
            + 0.15 * ranked["rank_train_cv_gap"]
            + 0.05 * ranked["rank_val_balanced_error"]
        )
    else:
        ranked["selection_rank_score"] = (
            0.60 * ranked["rank_cv_macro_f1"]
            + 0.20 * ranked["rank_cv_stability"]
            + 0.20 * ranked["rank_train_cv_gap"]
        )

    ranked["Selection Rank"] = ranked["selection_rank_score"].rank(ascending=True, method="min").astype("Int64")

    ranked["rank_test_macro_f1"] = rank_column("test_macro_f1", False)
    ranked["rank_test_balanced_accuracy"] = rank_column("test_balanced_accuracy", False)
    ranked["rank_test_mcc"] = rank_column("test_mcc", False)
    ranked["test_rank_score"] = (
        0.50 * ranked["rank_test_macro_f1"]
        + 0.30 * ranked["rank_test_balanced_accuracy"]
        + 0.20 * ranked["rank_test_mcc"]
    )
    ranked["Test Performance Rank"] = ranked["test_rank_score"].rank(ascending=True, method="min").astype("Int64")

    ranked["rank_generalization_gap"] = rank_column("abs_train_cv_gap", True)
    ranked["rank_generalization_cv_sd"] = rank_column("cv_macro_f1_std", True)
    ranked["rank_generalization_split_gap"] = rank_column("abs_cv_val_gap", True)
    if ranked["abs_cv_val_gap"].notna().any():
        ranked["generalization_rank_score"] = (
            0.45 * ranked["rank_generalization_gap"]
            + 0.35 * ranked["rank_generalization_cv_sd"]
            + 0.20 * ranked["rank_generalization_split_gap"]
        )
    else:
        ranked["generalization_rank_score"] = (
            0.60 * ranked["rank_generalization_gap"]
            + 0.40 * ranked["rank_generalization_cv_sd"]
        )
    ranked["Generalization Rank"] = ranked["generalization_rank_score"].rank(ascending=True, method="min").astype("Int64")

    return ranked.sort_values(
        ["Selection Rank", "cv_macro_f1", "cv_macro_f1_std"],
        ascending=[True, False, True], na_position="last"
    ).reset_index(drop=True)


def run(data_dir: str, models_dir: str, reports_dir: str,
        model_file: str | None = None, params_path: str = "params.yaml") -> dict:
    data_path = Path(data_dir)
    models_path = Path(models_dir)
    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)
    params = _load_params(params_path)

    if model_file is None:
        model_file = auto_select_model_file(models_path, reports_path)
    model_path = models_path / model_file
    if not model_path.exists():
        raise FileNotFoundError(f"Classifier file does not exist: {model_path}")

    print(f"[EVAL_CLASS] Evaluating classifier: {model_path.name}")
    row, detail = evaluate_model_file(model_path=model_path, data_dir=data_path, params=params)

    stem = model_path.stem
    json_path = reports_path / f"eval_class_{stem}.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(detail, handle, indent=2, allow_nan=True, ensure_ascii=False, default=str)

    ranked_df = rank_evaluated_classifiers(pd.DataFrame([row]))
    outputs = {
        "evaluate_class_summary.csv": ranked_df,
        "evaluate_class_simple_summary.csv": ranked_df.reindex(columns=[
            "model", "feature_set", "target", "file", "Selection Rank",
            "Test Performance Rank", "Generalization Rank", "train_macro_f1",
            "cv_macro_f1", "cv_macro_f1_std", "val_macro_f1", "test_macro_f1",
            "test_mcc", "test_roc_auc_ovr_macro", "train_cv_gap", "cv_val_gap",
            "val_test_gap", "generalization_diagnosis",
        ]),
        "Results_classification_evaluation_summary.csv": ranked_df,
    }
    for filename, frame in outputs.items():
        frame.to_csv(reports_path / filename, index=False, float_format="%.6f", encoding="utf-8-sig")

    metadata_path = reports_path / "evaluate_class_metadata.json"
    metadata = {
        "selected_model_file": model_file,
        "params_file": params_path,
        "cross_validation_uses_training_partition_only": True,
        "test_metrics_used_for_selection": False,
        "probabilities_generated_for_roc_auc": True,
        "probability_strategy": "predict_proba preferred; decision_function softmax fallback",
        "outputs": list(outputs.keys()) + [json_path.name],
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    print(f"[EVAL_CLASS] Detailed report: {json_path}")
    print(f"[EVAL_CLASS] Summary: {reports_path / 'evaluate_class_summary.csv'}")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a classification model and export probability-aware ROC/AUC metrics."
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--models_dir", required=True)
    parser.add_argument("--reports_dir", required=True)
    parser.add_argument("--model_file", required=False, default=None)
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()
    run(
        data_dir=args.data,
        models_dir=args.models_dir,
        reports_dir=args.reports_dir,
        model_file=args.model_file,
        params_path=args.params,
    )


if __name__ == "__main__":
    main()
