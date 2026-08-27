#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

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
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

CLASS_NAMES = {0: "Inactive", 1: "Normal", 2: "Active"}
CLASS_LABELS = [0, 1, 2]


def continuous_to_three_classes(values, lower, upper):
    """Convert a continuous target to three classes using explicit limits.

    The prediction GUI is target-agnostic; boundaries must come from the
    active YAML profile or the saved model package rather than AC constants.
    """
    values = np.asarray(values, dtype=float).ravel()
    lower = float(lower)
    upper = float(upper)
    if not lower < upper:
        raise ValueError(f"Expected lower < upper, got {lower} and {upper}.")
    classes = np.ones(values.shape, dtype=int)
    classes[values < lower] = 0
    classes[values > upper] = 2
    return classes


def _load_params(path):
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _ensure_2d(X):
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return X


def load_classifier_package(model_path):
    loaded = joblib.load(model_path)
    if isinstance(loaded, dict):
        model = loaded.get("model")
        if model is None:
            for value in loaded.values():
                if hasattr(value, "predict"):
                    model = value
                    break
        if model is None:
            raise ValueError(f"No classifier found in {model_path}")
        return {
            "model": model,
            "model_type": loaded.get("model_type", type(model).__name__),
            "feature_set": loaded.get("feature_set"),
            "target": loaded.get("target"),
            "feature_names": loaded.get("feature_names", []),
            "class_names": loaded.get("class_names", CLASS_NAMES),
            "class_boundaries": loaded.get("class_boundaries", {}),
            "raw": loaded,
        }
    return {
        "model": loaded,
        "model_type": type(loaded).__name__,
        "feature_set": None,
        "target": None,
        "feature_names": [],
        "class_names": CLASS_NAMES,
        "class_boundaries": {},
        "raw": loaded,
    }


def infer_metadata_from_filename(filename):
    stem = Path(filename).stem
    stem = stem.removesuffix("_class")
    match = re.match(r"(.+?)_(fs\d+)_(.+?)$", stem, flags=re.IGNORECASE)
    if match:
        return match.group(1), match.group(2).lower(), match.group(3)
    return stem, None, None




def _normalise_feature_name(name):
    """Preserve a predictor name exactly for generic template use."""
    return str(name).strip()


def _clean_input_names(names):
    """Preserve predictor names/order while removing blanks and duplicates."""
    cleaned = []
    for name in names or []:
        value = _normalise_feature_name(name)
        if value and value not in cleaned:
            cleaned.append(value)
    return cleaned


def _extract_feature_list(value):
    """Extract a feature-name list from common params.yaml structures."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return _clean_input_names(value)
    if isinstance(value, str):
        return _clean_input_names([value])
    if isinstance(value, dict):
        for key in ("features", "feature_names", "columns", "inputs", "X"):
            if key in value:
                return _extract_feature_list(value[key])
    return []


def _params_feature_names(params, target, fs):
    """Resolve a feature set from params.yaml, preferring TARGETS[target][fs]."""
    fs = str(fs or "").lower()
    target = str(target or "")

    roots = []
    for key in ("TARGETS", "targets", "feature_sets", "FEATURE_SETS"):
        block = params.get(key)
        if isinstance(block, dict):
            roots.append(block)

    for root in roots:
        target_block = None
        for key, value in root.items():
            if str(key).lower() == target.lower():
                target_block = value
                break

        search_blocks = []
        if isinstance(target_block, dict):
            search_blocks.append(target_block)
        search_blocks.append(root)

        for block in search_blocks:
            for key, value in block.items():
                if str(key).lower() == fs:
                    names = _extract_feature_list(value)
                    if names:
                        return names
    return []


def _params_feature_note(params, name, target=None):
    """Return an optional per-feature GUI note from params.yaml.

    Supported generic schema:
        feature_notes:
          FeatureName: "note shown beside the input"

    Target-scoped notes are also supported:
        feature_notes:
          TargetName:
            FeatureName: "note"
    """
    params = params or {}
    notes = params.get("feature_notes", params.get("FEATURE_NOTES", {}))
    if not isinstance(notes, dict):
        return ""

    if target:
        target_block = next(
            (value for key, value in notes.items()
             if str(key).lower() == str(target).lower() and isinstance(value, dict)),
            None,
        )
        if isinstance(target_block, dict):
            for key, value in target_block.items():
                if str(key).lower() == str(name).lower():
                    return str(value).strip()

    for key, value in notes.items():
        if str(key).lower() == str(name).lower() and not isinstance(value, dict):
            return str(value).strip()
    return ""


def _model_expected_feature_count(model):
    """Find the number of raw input features expected by a model or pipeline."""
    candidates = [model]
    if hasattr(model, "named_steps"):
        candidates.extend(list(model.named_steps.values()))
    for obj in candidates:
        value = getattr(obj, "n_features_in_", None)
        if value is not None:
            try:
                return int(value)
            except Exception:
                pass
    return None


def _model_feature_names(package, fs=None, params=None, target=None):
    """
    Resolve raw feature names in training order.

    Priority:
      1. params.yaml feature-set definition
      2. saved package feature_names
      3. estimator feature_names_in_

    Predictor names are preserved exactly so the file can be reused as a
    template with arbitrary datasets and column names.
    """
    model = package["model"]
    fs = str(fs or "").lower()
    target = target or package.get("target") or ""

    params_names = _params_feature_names(params or {}, target, fs)

    raw_names = package.get("feature_names") or []
    if isinstance(raw_names, dict):
        raw_names = list(raw_names.values())
    package_names = _clean_input_names(raw_names)

    estimator_names = []
    candidates = [model]
    if hasattr(model, "named_steps"):
        candidates.extend(list(model.named_steps.values()))
    for obj in candidates:
        values = getattr(obj, "feature_names_in_", None)
        if values is not None:
            estimator_names = _clean_input_names(list(values))
            if estimator_names:
                break

    names = params_names or package_names or estimator_names
    expected = _model_expected_feature_count(model)

    if not names:
        raise ValueError(
            f"Feature set '{fs}' was not found in params.yaml and no valid feature metadata "
            "was stored in the classifier package."
        )

    if expected is not None and len(names) != expected:
        raise ValueError(
            f"Feature-count mismatch for {fs}: params.yaml defines {len(names)} predictor(s) "
            f"{names}, while the selected model expects {expected}. Ensure the selected model "
            "was trained with the same params.yaml feature-set definition."
        )

    return names

def _build_exact_feature_row(feature_names, values):
    """Build one numeric prediction row in the exact training feature order."""
    missing = [name for name in feature_names if values.get(name) is None]
    if missing:
        raise ValueError("Missing required feature(s): " + ", ".join(missing))
    return np.asarray([[float(values[name]) for name in feature_names]], dtype=float)


def get_class_scores(model, X):
    """Return a 3-column probability-like score matrix aligned to classes 0,1,2."""
    X = _ensure_2d(X)
    model_classes = np.asarray(getattr(model, "classes_", CLASS_LABELS), dtype=int)

    if hasattr(model, "predict_proba"):
        raw = np.asarray(model.predict_proba(X), dtype=float)
    elif hasattr(model, "decision_function"):
        decision = np.asarray(model.decision_function(X), dtype=float)
        if decision.ndim == 1:
            decision = np.column_stack([-decision, decision])
        decision -= np.max(decision, axis=1, keepdims=True)
        exp_scores = np.exp(decision)
        raw = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    else:
        predicted = np.asarray(model.predict(X), dtype=int)
        raw = np.zeros((len(predicted), len(model_classes)), dtype=float)
        for row, label in enumerate(predicted):
            locations = np.where(model_classes == label)[0]
            if locations.size:
                raw[row, locations[0]] = 1.0

    aligned = np.zeros((raw.shape[0], 3), dtype=float)
    for source_index, class_label in enumerate(model_classes):
        if int(class_label) in CLASS_LABELS and source_index < raw.shape[1]:
            aligned[:, int(class_label)] = raw[:, source_index]
    row_sums = aligned.sum(axis=1, keepdims=True)
    zero_rows = row_sums.ravel() <= 0
    aligned[~zero_rows] /= row_sums[~zero_rows]
    aligned[zero_rows] = 1.0 / 3.0
    return aligned


def calculate_classification_metrics(y_true, y_pred, probabilities=None):
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_pred = np.asarray(y_pred, dtype=int).ravel()
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=CLASS_LABELS).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=CLASS_LABELS,
            target_names=[CLASS_NAMES[i] for i in CLASS_LABELS],
            output_dict=True, zero_division=0,
        ),
    }
    if probabilities is not None:
        probabilities = np.asarray(probabilities, dtype=float)
        try:
            result["log_loss"] = float(log_loss(y_true, probabilities, labels=CLASS_LABELS))
        except Exception:
            result["log_loss"] = np.nan
        try:
            result["roc_auc_ovr_macro"] = float(
                roc_auc_score(y_true, probabilities, labels=CLASS_LABELS,
                              multi_class="ovr", average="macro")
            )
            result["roc_auc_ovr_weighted"] = float(
                roc_auc_score(y_true, probabilities, labels=CLASS_LABELS,
                              multi_class="ovr", average="weighted")
            )
        except Exception:
            result["roc_auc_ovr_macro"] = np.nan
            result["roc_auc_ovr_weighted"] = np.nan
    return result


def multiclass_roc_data(y_true, probabilities):
    """Compute one-vs-rest ROC curves and AUC values for the three activity classes."""
    y_true = np.asarray(y_true, dtype=int).ravel()
    probabilities = np.asarray(probabilities, dtype=float)

    if probabilities.ndim != 2 or probabilities.shape[1] != 3:
        raise ValueError(
            f"Expected probability matrix with shape (n_samples, 3); "
            f"received {probabilities.shape}."
        )
    if len(y_true) != probabilities.shape[0]:
        raise ValueError(
            "Number of y_true values does not match the probability rows."
        )

    binary = label_binarize(y_true, classes=CLASS_LABELS)
    curves = {}
    auc_values = {}

    for class_code in CLASS_LABELS:
        # ROC for a class requires both positive and negative examples.
        if len(np.unique(binary[:, class_code])) < 2:
            continue
        fpr, tpr, thresholds = roc_curve(
            binary[:, class_code],
            probabilities[:, class_code],
        )
        curves[class_code] = {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
        }
        auc_values[class_code] = float(auc(fpr, tpr))

    try:
        macro_auc = float(
            roc_auc_score(
                y_true,
                probabilities,
                labels=CLASS_LABELS,
                multi_class="ovr",
                average="macro",
            )
        )
    except Exception:
        macro_auc = np.nan

    try:
        weighted_auc = float(
            roc_auc_score(
                y_true,
                probabilities,
                labels=CLASS_LABELS,
                multi_class="ovr",
                average="weighted",
            )
        )
    except Exception:
        weighted_auc = np.nan

    # Micro-average ROC treats the binarized class decisions jointly.
    try:
        micro_fpr, micro_tpr, micro_thresholds = roc_curve(
            binary.ravel(), probabilities.ravel()
        )
        micro_auc = float(auc(micro_fpr, micro_tpr))
    except Exception:
        micro_fpr = micro_tpr = micro_thresholds = None
        micro_auc = np.nan

    return {
        "curves": curves,
        "auc_by_class": auc_values,
        "macro_auc": macro_auc,
        "weighted_auc": weighted_auc,
        "micro_curve": (
            None if micro_fpr is None else {
                "fpr": micro_fpr,
                "tpr": micro_tpr,
                "thresholds": micro_thresholds,
            }
        ),
        "micro_auc": micro_auc,
    }



def _classification_prediction_arrays(df):
    """Return y_true/y_pred from either GUI or paper-report column names."""
    true_col = "y_true_class" if "y_true_class" in df.columns else "y_true"
    pred_col = "y_pred_class" if "y_pred_class" in df.columns else "y_pred"
    missing = [c for c in (true_col, pred_col) if c not in df.columns]
    if missing:
        raise ValueError(
            "Classification data require y_true/y_pred or "
            "y_true_class/y_pred_class columns."
        )
    return df[true_col].to_numpy(int), df[pred_col].to_numpy(int)


def _classification_probabilities_from_df(df):
    """Return a validated 3-column probability matrix for ROC/AUC.

    The function accepts the canonical columns written by evaluate_all_class.py:
        prob_inactive, prob_normal, prob_active

    It also accepts case variants (for example prob_Inactive) and normalizes
    rows to sum to one. Rows with missing/non-finite class scores cause a
    clear ValueError instead of silently disabling ROC/AUC.
    """
    canonical = {
        "inactive": "prob_inactive",
        "normal": "prob_normal",
        "active": "prob_active",
    }

    # Case-insensitive lookup while preserving the real dataframe column names.
    lookup = {str(column).strip().lower(): column for column in df.columns}
    resolved = []
    missing = []
    for class_name, wanted in canonical.items():
        actual = lookup.get(wanted.lower())
        if actual is None:
            missing.append(wanted)
        else:
            resolved.append(actual)

    if missing:
        return None

    probabilities = (
        df[resolved]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )

    if probabilities.ndim != 2 or probabilities.shape[1] != 3:
        raise ValueError(
            "ROC/AUC requires exactly three probability columns: "
            "prob_inactive, prob_normal, prob_active."
        )

    bad_rows = ~np.isfinite(probabilities).all(axis=1)
    if np.any(bad_rows):
        raise ValueError(
            f"Probability columns exist, but {int(np.sum(bad_rows))} selected "
            "row(s) contain missing or non-finite values. Re-run "
            "evaluate_all_class.py to regenerate classification_predictions_long.csv."
        )

    # Protect against tiny numerical deviations and decision-score exports.
    probabilities = np.clip(probabilities, 0.0, None)
    row_sums = probabilities.sum(axis=1, keepdims=True)
    zero_rows = row_sums.ravel() <= 0
    if np.any(zero_rows):
        raise ValueError(
            f"{int(np.sum(zero_rows))} selected row(s) have zero total class "
            "probability and cannot be used for ROC/AUC."
        )
    probabilities = probabilities / row_sums
    return probabilities


def _hard_prediction_ovr_points(y_true, y_pred):
    """One-vs-rest sensitivity/FPR points computed from hard class predictions.

    These are deliberately not labelled as ROC curves or AUC because a single
    hard prediction per sample does not provide a threshold-varying score.
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_pred = np.asarray(y_pred, dtype=int).ravel()
    rows = []
    for class_code in CLASS_LABELS:
        actual_pos = y_true == class_code
        pred_pos = y_pred == class_code
        tp = int(np.sum(actual_pos & pred_pos))
        fn = int(np.sum(actual_pos & ~pred_pos))
        fp = int(np.sum(~actual_pos & pred_pos))
        tn = int(np.sum(~actual_pos & ~pred_pos))
        sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
        specificity = tn / (tn + fp) if (tn + fp) else np.nan
        fpr = 1.0 - specificity if np.isfinite(specificity) else np.nan
        rows.append((class_code, fpr, sensitivity, specificity))
    return rows


def _draw_classification_confusion(ax, y_true, y_pred, title="Three-class confusion matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)
    image = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(CLASS_LABELS)
    ax.set_yticks(CLASS_LABELS)
    ax.set_xticklabels([CLASS_NAMES[i] for i in CLASS_LABELS], rotation=25, ha="right")
    ax.set_yticklabels([CLASS_NAMES[i] for i in CLASS_LABELS])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title(title)
    threshold = cm.max() / 2 if cm.size else 0
    for i in range(3):
        for j in range(3):
            ax.text(
                j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black",
            )
    return image


def _draw_classification_roc_or_fallback(ax, y_true, y_pred, probabilities):
    """Draw genuine multiclass ROC curves when probability scores are available."""
    if probabilities is not None:
        roc_data = multiclass_roc_data(y_true, probabilities)

        for class_code, curve in roc_data["curves"].items():
            class_auc = roc_data["auc_by_class"].get(class_code, np.nan)
            ax.plot(
                curve["fpr"],
                curve["tpr"],
                linewidth=2,
                label=f"{CLASS_NAMES[class_code]} (AUC={class_auc:.3f})",
            )

        micro_curve = roc_data.get("micro_curve")
        if micro_curve is not None:
            ax.plot(
                micro_curve["fpr"],
                micro_curve["tpr"],
                linewidth=1.7,
                linestyle=":",
                label=f"Micro-average (AUC={roc_data['micro_auc']:.3f})",
            )

        ax.plot([0, 1], [0, 1], "--", linewidth=1, label="Random")
        ax.set_title(
            f"One-vs-rest ROC | Macro AUC={roc_data['macro_auc']:.3f}"
        )
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, alpha=0.25)
        return roc_data

    # Probability columns are not present. Keep the non-ROC diagnostic rather
    # than calculating an invalid AUC from hard class predictions.
    points = _hard_prediction_ovr_points(y_true, y_pred)
    ax.plot([0, 1], [0, 1], "--", linewidth=1, label="No-skill reference")
    for class_code, fpr, sensitivity, specificity in points:
        if np.isfinite(fpr) and np.isfinite(sensitivity):
            ax.scatter([fpr], [sensitivity], s=65, label=CLASS_NAMES[class_code])
            ax.annotate(
                CLASS_NAMES[class_code],
                (fpr, sensitivity),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("Sensitivity (recall)")
    ax.set_title(
        "ROC unavailable: probability columns were not found"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc="lower right")
    return None


def _draw_classification_classwise(ax, metrics):
    report = metrics["classification_report"]
    x = np.arange(3)
    width = 0.25
    precision_values = [report[CLASS_NAMES[i]]["precision"] for i in CLASS_LABELS]
    recall_values = [report[CLASS_NAMES[i]]["recall"] for i in CLASS_LABELS]
    f1_values = [report[CLASS_NAMES[i]]["f1-score"] for i in CLASS_LABELS]
    ax.bar(x - width, precision_values, width, label="Precision")
    ax.bar(x, recall_values, width, label="Recall")
    ax.bar(x + width, f1_values, width, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels([CLASS_NAMES[i] for i in CLASS_LABELS])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Per-class performance")
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", alpha=0.25)


def _draw_classification_confidence_or_fallback(ax, y_true, y_pred, probabilities):
    if probabilities is not None:
        confidence = np.max(probabilities, axis=1)
        correct = np.asarray(y_true) == np.asarray(y_pred)
        ax.hist(confidence[correct], bins=10, alpha=0.7, label="Correct")
        ax.hist(confidence[~correct], bins=10, alpha=0.7, label="Incorrect")
        ax.set_xlabel("Maximum predicted probability")
        ax.set_ylabel("Count")
        ax.set_title("Prediction confidence distribution")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)
        return

    # Fallback: actual vs predicted class composition, requiring no probabilities.
    actual_counts = np.array([np.sum(np.asarray(y_true) == c) for c in CLASS_LABELS], dtype=float)
    pred_counts = np.array([np.sum(np.asarray(y_pred) == c) for c in CLASS_LABELS], dtype=float)
    n = max(1, len(y_true))
    actual_pct = 100.0 * actual_counts / n
    pred_pct = 100.0 * pred_counts / n
    x = np.arange(3)
    width = 0.36
    ax.bar(x - width / 2, actual_pct, width, label="Actual")
    ax.bar(x + width / 2, pred_pct, width, label="Predicted")
    ax.set_xticks(x)
    ax.set_xticklabels([CLASS_NAMES[i] for i in CLASS_LABELS])
    ax.set_ylabel("Samples (%)")
    ax.set_ylim(0, max(100.0, float(np.nanmax([actual_pct, pred_pct])) * 1.15))
    ax.set_title("Actual vs predicted class distribution")
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", alpha=0.2)


def save_classification_dashboard_png(df, output_path, model_name="classifier", split_name="all", dpi=600):
    """Save the four classification diagnostics together in one 2x2 PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_true, y_pred = _classification_prediction_arrays(df)
    probabilities = _classification_probabilities_from_df(df)
    metrics = calculate_classification_metrics(y_true, y_pred, probabilities)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(2, 2, figsize=(13.5, 10.0))
    _draw_classification_confusion(axs[0, 0], y_true, y_pred)
    _draw_classification_roc_or_fallback(axs[0, 1], y_true, y_pred, probabilities)
    _draw_classification_classwise(axs[1, 0], metrics)
    _draw_classification_confidence_or_fallback(axs[1, 1], y_true, y_pred, probabilities)
    fig.suptitle(f"Classification diagnostics: {model_name} ({split_name})", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return output_path



def generate_paper_classification_report(df, output_dir, model_name="classifier", split_name="all"):
    """Create publication-ready classification tables, figures, and a brief summary.

    The function expects y_true_class and y_pred_class columns. Probability
    columns are optional but enable ROC/AUC and confidence analyses.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir) / "paper_classification_report" / _safe_path_component(model_name) / _safe_path_component(split_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_true, y_pred = _classification_prediction_arrays(df)
    probabilities = _classification_probabilities_from_df(df)
    metrics = calculate_classification_metrics(y_true, y_pred, probabilities)
    report = metrics["classification_report"]

    rows = []
    for class_code in CLASS_LABELS:
        class_name = CLASS_NAMES[class_code]
        class_report = report[class_name]
        rows.append({
            "Class": class_name,
            "Code": class_code,
            "Support": int(class_report["support"]),
            "Precision": float(class_report["precision"]),
            "Recall": float(class_report["recall"]),
            "F1-score": float(class_report["f1-score"]),
        })
    rows.extend([
        {
            "Class": "Macro average", "Code": "",
            "Support": int(report["macro avg"]["support"]),
            "Precision": float(report["macro avg"]["precision"]),
            "Recall": float(report["macro avg"]["recall"]),
            "F1-score": float(report["macro avg"]["f1-score"]),
        },
        {
            "Class": "Weighted average", "Code": "",
            "Support": int(report["weighted avg"]["support"]),
            "Precision": float(report["weighted avg"]["precision"]),
            "Recall": float(report["weighted avg"]["recall"]),
            "F1-score": float(report["weighted avg"]["f1-score"]),
        },
    ])
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "classification_performance_table.csv", index=False)
    try:
        latex_table = table.to_latex(
            index=False, float_format=lambda value: f"{value:.3f}",
            caption="Class-wise performance of the soil activity classifier.",
            label="tab:classification_performance", escape=True,
        )
        (output_dir / "classification_performance_table.tex").write_text(latex_table, encoding="utf-8")
    except Exception as exc:
        (output_dir / "classification_performance_table.tex").write_text(
            f"% LaTeX export failed: {exc}\n", encoding="utf-8"
        )

    overall = pd.DataFrame([{
        "Model": model_name,
        "Split": split_name,
        "N": len(df),
        "Accuracy": metrics["accuracy"],
        "Balanced accuracy": metrics["balanced_accuracy"],
        "Macro precision": metrics["macro_precision"],
        "Macro recall": metrics["macro_recall"],
        "Macro F1": metrics["macro_f1"],
        "Weighted F1": metrics["weighted_f1"],
        "MCC": metrics["mcc"],
        "Cohen kappa": metrics["cohen_kappa"],
        "Macro OVR AUC": metrics.get("roc_auc_ovr_macro", np.nan),
        "Weighted OVR AUC": metrics.get("roc_auc_ovr_weighted", np.nan),
        "Log loss": metrics.get("log_loss", np.nan),
    }])
    overall.to_csv(output_dir / "overall_classification_metrics.csv", index=False)

    # Confusion matrix figure
    cm = np.asarray(metrics["confusion_matrix"], dtype=int)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(CLASS_LABELS)
    ax.set_yticks(CLASS_LABELS)
    ax.set_xticklabels([CLASS_NAMES[i] for i in CLASS_LABELS])
    ax.set_yticklabels([CLASS_NAMES[i] for i in CLASS_LABELS])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title(f"Confusion matrix: {model_name} ({split_name})")
    threshold = cm.max() / 2 if cm.size else 0
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "paper_confusion_matrix.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    # Class-wise precision, recall, and F1 figure
    x = np.arange(3)
    width = 0.25
    precision_values = [report[CLASS_NAMES[i]]["precision"] for i in CLASS_LABELS]
    recall_values = [report[CLASS_NAMES[i]]["recall"] for i in CLASS_LABELS]
    f1_values = [report[CLASS_NAMES[i]]["f1-score"] for i in CLASS_LABELS]
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.bar(x - width, precision_values, width, label="Precision")
    ax.bar(x, recall_values, width, label="Recall")
    ax.bar(x + width, f1_values, width, label="F1-score")
    ax.set_xticks(x)
    ax.set_xticklabels([CLASS_NAMES[i] for i in CLASS_LABELS])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_xlabel("Activity class")
    ax.set_title(f"Class-wise classification performance: {model_name}")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "paper_classwise_metrics.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    roc_summary = {}
    if probabilities is not None:
        roc_data = multiclass_roc_data(y_true, probabilities)
        fig, ax = plt.subplots(figsize=(6.4, 5.2))
        for class_code, curve in roc_data["curves"].items():
            class_auc = roc_data["auc_by_class"].get(class_code, np.nan)
            ax.plot(curve["fpr"], curve["tpr"], linewidth=2,
                    label=f"{CLASS_NAMES[class_code]} (AUC={class_auc:.3f})")
        ax.plot([0, 1], [0, 1], "--", linewidth=1, label="Random")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(f"One-vs-rest ROC curves: {model_name}")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(output_dir / "paper_multiclass_roc.png", dpi=600, bbox_inches="tight")
        plt.close(fig)
        roc_summary = {
            "auc_by_class": {CLASS_NAMES[int(k)]: float(v) for k, v in roc_data["auc_by_class"].items()},
            "macro_auc": float(roc_data["macro_auc"]),
            "weighted_auc": float(roc_data["weighted_auc"]),
        }

        confidence = np.max(probabilities, axis=1)
        correct = y_true == y_pred
        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        ax.hist(confidence[correct], bins=10, alpha=0.75, label="Correct")
        ax.hist(confidence[~correct], bins=10, alpha=0.75, label="Incorrect")
        ax.set_xlabel("Maximum predicted probability")
        ax.set_ylabel("Number of samples")
        ax.set_title(f"Prediction-confidence distribution: {model_name}")
        ax.legend()
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_dir / "paper_confidence_distribution.png", dpi=600, bbox_inches="tight")
        plt.close(fig)

    dashboard_path = output_dir / "paper_classification_all_plots.png"
    save_classification_dashboard_png(
        df, dashboard_path, model_name=model_name, split_name=split_name, dpi=600
    )

    weakest_code = min(CLASS_LABELS, key=lambda code: report[CLASS_NAMES[code]]["f1-score"])
    strongest_code = max(CLASS_LABELS, key=lambda code: report[CLASS_NAMES[code]]["f1-score"])
    summary_lines = [
        "CLASSIFICATION PERFORMANCE SUMMARY",
        "=" * 72,
        f"Model: {model_name}",
        f"Split: {split_name}",
        f"Number of evaluated samples: {len(df)}",
        "",
        f"Accuracy: {metrics['accuracy']:.4f}",
        f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}",
        f"Macro F1: {metrics['macro_f1']:.4f}",
        f"Weighted F1: {metrics['weighted_f1']:.4f}",
        f"Matthews correlation coefficient: {metrics['mcc']:.4f}",
        f"Cohen's kappa: {metrics['cohen_kappa']:.4f}",
    ]
    if probabilities is not None:
        summary_lines.extend([
            f"Macro one-vs-rest AUC: {metrics.get('roc_auc_ovr_macro', np.nan):.4f}",
            f"Weighted one-vs-rest AUC: {metrics.get('roc_auc_ovr_weighted', np.nan):.4f}",
            f"Log loss: {metrics.get('log_loss', np.nan):.4f}",
        ])
    summary_lines.extend([
        "",
        "Class-wise interpretation",
        f"Strongest class by F1-score: {CLASS_NAMES[strongest_code]} "
        f"(F1={report[CLASS_NAMES[strongest_code]]['f1-score']:.4f}).",
        f"Weakest class by F1-score: {CLASS_NAMES[weakest_code]} "
        f"(F1={report[CLASS_NAMES[weakest_code]]['f1-score']:.4f}).",
        "",
        "Suggested paper wording",
        (
            f"The classifier achieved an accuracy of {metrics['accuracy']:.3f}, a balanced "
            f"accuracy of {metrics['balanced_accuracy']:.3f}, and a macro-F1 score of "
            f"{metrics['macro_f1']:.3f} on the {split_name} set. Class-wise performance "
            f"was strongest for the {CLASS_NAMES[strongest_code].lower()} class and weakest "
            f"for the {CLASS_NAMES[weakest_code].lower()} class. Matthews correlation "
            f"coefficient ({metrics['mcc']:.3f}) and Cohen's kappa "
            f"({metrics['cohen_kappa']:.3f}) were also reported to account for class imbalance."
        ),
        "",
        "Class definitions",
        "Class names and boundaries are defined by the trained model and params.yaml.",
    ])
    (output_dir / "classification_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    json_payload = {
        "model": model_name, "split": split_name, "n": len(df),
        "metrics": metrics, "roc": roc_summary,
        "strongest_class": CLASS_NAMES[strongest_code],
        "weakest_class": CLASS_NAMES[weakest_code],
    }
    with (output_dir / "classification_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(json_payload, handle, indent=2)

    return {
        "output_dir": str(output_dir),
        "metrics_table": str(output_dir / "classification_performance_table.csv"),
        "overall_metrics": str(output_dir / "overall_classification_metrics.csv"),
        "summary": str(output_dir / "classification_summary.txt"),
        "all_plots_png": str(dashboard_path),
    }


def _safe_path_component(value):
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "unknown")).strip("._")
    return text or "unknown"

def evaluate_prediction_file(prediction_csv, output_dir=None):
    df = pd.read_csv(prediction_csv)
    required = {"y_true_class", "y_pred_class"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in prediction file: {sorted(missing)}")
    probability_columns = ["prob_inactive", "prob_normal", "prob_active"]
    probabilities = df[probability_columns].to_numpy(float) if all(c in df.columns for c in probability_columns) else None
    metrics = calculate_classification_metrics(df["y_true_class"], df["y_pred_class"], probabilities)
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "prediction_class_report.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        model_name = "classifier"
        split_name = "all"
        if "file" in df.columns and df["file"].nunique() == 1:
            model_name = str(df["file"].iloc[0])
        if "split" in df.columns and df["split"].nunique() == 1:
            split_name = str(df["split"].iloc[0])
        generate_paper_classification_report(df, output_dir, model_name, split_name)
    return metrics



# ============================================================
# SCIENTIFIC CLASSIFICATION RANKING AND WEIGHT SENSITIVITY
# Aligned with predict_Reg.py Monte Carlo robustness logic
# ============================================================

# Classification analog of the regression ranking scenarios:
#   performance  -> CV Macro-F1
#   stability    -> CV Macro-F1 SD
#   generalization gap -> |Train Macro-F1 - CV Macro-F1|
# Test metrics are NEVER used for model selection or Monte Carlo robustness.
CLASSIFICATION_WEIGHT_SCENARIOS = {
    "equal": {
        "rank_cv_macro_f1": 1 / 3,
        "rank_cv_stability": 1 / 3,
        "rank_train_cv_gap": 1 / 3,
    },
    "performance_focused": {
        "rank_cv_macro_f1": 0.70,
        "rank_cv_stability": 0.15,
        "rank_train_cv_gap": 0.15,
    },
    "stability_focused": {
        "rank_cv_macro_f1": 0.40,
        "rank_cv_stability": 0.30,
        "rank_train_cv_gap": 0.30,
    },
}


def _classification_numeric(df, column):
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _normalise_classification_ranking_schema(df):
    """Normalize common classifier-evaluation column names."""
    df = df.copy()

    if "feature_set" not in df.columns and "fs" in df.columns:
        df["feature_set"] = df["fs"]
    if "fs" not in df.columns and "feature_set" in df.columns:
        df["fs"] = df["feature_set"]

    aliases = {
        "train_macro_f1": ["train_macro_f1", "macro_f1_train"],
        "cv_macro_f1": [
            "cv_macro_f1",
            "cv_train_macro_f1",
            "cv_macro_f1_mean",
        ],
        "cv_macro_f1_std": [
            "cv_macro_f1_std",
            "cv_train_macro_f1_std",
            "cv_macro_f1_sd",
        ],
        "val_macro_f1": ["val_macro_f1", "macro_f1_val"],
        "test_macro_f1": ["test_macro_f1", "macro_f1_test"],
        "val_balanced_accuracy": [
            "val_balanced_accuracy",
            "balanced_accuracy_val",
        ],
        "test_balanced_accuracy": [
            "test_balanced_accuracy",
            "balanced_accuracy_test",
        ],
        "cv_mcc": [
            "cv_mcc", "cv_mcc_mean", "cv_train_mcc", "mean_cv_mcc",
            "nested_cv_mcc", "outer_cv_mcc"
        ],
        "val_mcc": ["val_mcc", "mcc_val"],
        "test_mcc": ["test_mcc", "mcc_test"],
    }

    for canonical, candidates in aliases.items():
        if canonical not in df.columns:
            for candidate in candidates:
                if candidate in df.columns:
                    df[canonical] = df[candidate]
                    break
        if canonical not in df.columns:
            df[canonical] = np.nan
        df[canonical] = pd.to_numeric(df[canonical], errors="coerce")

    if "model" not in df.columns:
        raise ValueError("Classification ranking input requires a model column.")
    if "feature_set" not in df.columns:
        raise ValueError("Classification ranking input requires feature_set or fs.")
    if "target" not in df.columns:
        df["target"] = "AC"

    return df


def _classification_rank(series, ascending):
    return pd.to_numeric(series, errors="coerce").rank(
        ascending=ascending,
        method="average",
        na_option="bottom",
    )


def classification_statistical_ranking_system(df):
    """CV-only, gap-aware classification ranking aligned with regression.

    Primary model selection excludes validation/test holdout metrics and uses:
      60% mean group-aware CV Macro-F1 rank
      20% CV Macro-F1 SD rank
      20% absolute train-CV Macro-F1 gap rank

    Test metrics are retained only for a separate descriptive test rank.
    """
    if df is None or df.empty:
        return df

    df = _normalise_classification_ranking_schema(df)

    df["train_cv_gap"] = df["train_macro_f1"] - df["cv_macro_f1"]
    df["abs_train_cv_gap"] = df["train_cv_gap"].abs()

    # Optional diagnostics when validation/test columns exist. They do not
    # participate in the primary ranking or Monte Carlo robustness analysis.
    df["train_val_gap"] = df["train_macro_f1"] - df["val_macro_f1"]
    df["cv_val_gap"] = df["cv_macro_f1"] - df["val_macro_f1"]
    df["val_test_gap"] = df["val_macro_f1"] - df["test_macro_f1"]
    df["cv_test_gap"] = df["cv_macro_f1"] - df["test_macro_f1"]
    df["abs_cv_val_gap"] = df["cv_val_gap"].abs()
    df["abs_val_test_gap"] = df["val_test_gap"].abs()
    df["abs_cv_test_gap"] = df["cv_test_gap"].abs()

    df["possible_overfitting"] = df["train_cv_gap"] >= 0.15
    df["possible_underfitting"] = (
        (df["train_macro_f1"] < 0.55)
        & (df["cv_macro_f1"] < 0.50)
    )
    df["high_cv_instability"] = df["cv_macro_f1_std"] >= 0.15
    df["test_shift"] = df["abs_cv_test_gap"] >= 0.15
    df["stable_generalization"] = (
        (df["abs_train_cv_gap"] <= 0.08)
        & (df["cv_macro_f1"] >= 0.50)
        & (~df["high_cv_instability"])
    )

    diagnoses = []
    for _, row in df.iterrows():
        if bool(row["possible_underfitting"]):
            diagnosis = "Possible underfitting"
        elif bool(row["possible_overfitting"]) and bool(row["high_cv_instability"]):
            diagnosis = "Overfitting with high CV instability"
        elif bool(row["possible_overfitting"]):
            diagnosis = "Possible overfitting"
        elif bool(row["high_cv_instability"]):
            diagnosis = "High CV instability"
        elif bool(row["test_shift"]):
            diagnosis = "CV-test shift"
        elif bool(row["stable_generalization"]):
            diagnosis = "Stable generalization"
        else:
            diagnosis = "Moderate generalization"
        diagnoses.append(diagnosis)
    df["generalization_diagnosis"] = diagnoses

    # Primary development-only ranking. If a genuine cross-validated MCC
    # column is available, MCC participates in selection as stated in the
    # classification methodology. Test MCC is NEVER used for selection.
    df["rank_cv_macro_f1"] = _classification_rank(df["cv_macro_f1"], ascending=False)
    df["rank_cv_stability"] = _classification_rank(df["cv_macro_f1_std"], ascending=True)
    df["rank_train_cv_gap"] = _classification_rank(df["abs_train_cv_gap"], ascending=True)

    cv_mcc_available = "cv_mcc" in df.columns and df["cv_mcc"].notna().any()
    df["mcc_used_for_selection"] = bool(cv_mcc_available)
    if cv_mcc_available:
        df["rank_cv_mcc"] = _classification_rank(df["cv_mcc"], ascending=False)
        # Performance remains dominant: Macro-F1 + MCC = 65% of the score.
        df["selection_rank_score"] = (
            0.45 * df["rank_cv_macro_f1"]
            + 0.20 * df["rank_cv_mcc"]
            + 0.15 * df["rank_cv_stability"]
            + 0.20 * df["rank_train_cv_gap"]
        )
    else:
        df["rank_cv_mcc"] = np.nan
        # Transparent fallback when the evaluation CSV contains no CV MCC.
        df["selection_rank_score"] = (
            0.60 * df["rank_cv_macro_f1"]
            + 0.20 * df["rank_cv_stability"]
            + 0.20 * df["rank_train_cv_gap"]
        )
    df["statistical_rank"] = df["selection_rank_score"].rank(
        ascending=True, method="min"
    ).astype("Int64")
    df["selection_rank"] = df["statistical_rank"]

    score = df["selection_rank_score"]
    finite_score = score[np.isfinite(score)]
    if finite_score.empty or finite_score.max() == finite_score.min():
        df["statistical_score"] = 1.0
    else:
        df["statistical_score"] = 1.0 - (
            score - finite_score.min()
        ) / (finite_score.max() - finite_score.min())

    stability_columns = ["rank_cv_macro_f1", "rank_cv_stability", "rank_train_cv_gap"]
    if cv_mcc_available:
        stability_columns.insert(1, "rank_cv_mcc")
    rank_matrix = df[stability_columns].to_numpy(dtype=float)
    df["rank_stability"] = [
        1.0 if np.std(row) == 0 else 1.0 / (1.0 + np.std(row) / np.mean(row))
        for row in rank_matrix
    ]

    # Separate DESCRIPTIVE test ranking. It never feeds back into selection.
    if df["test_macro_f1"].notna().any():
        df["rank_test_macro_f1"] = _classification_rank(
            df["test_macro_f1"], ascending=False
        )
        df["rank_test_balanced_accuracy"] = _classification_rank(
            df["test_balanced_accuracy"], ascending=False
        )
        df["rank_test_mcc"] = _classification_rank(df["test_mcc"], ascending=False)
        df["test_rank_score"] = (
            0.50 * df["rank_test_macro_f1"]
            + 0.30 * df["rank_test_balanced_accuracy"]
            + 0.20 * df["rank_test_mcc"]
        )
        df["test_performance_rank"] = df["test_rank_score"].rank(
            ascending=True, method="min"
        ).astype("Int64")
    else:
        df["test_performance_rank"] = pd.Series(
            pd.NA, index=df.index, dtype="Int64"
        )

    # Same structure as regression: 60% train-CV gap + 40% CV variability.
    df["generalization_rank_score"] = (
        0.60 * _classification_rank(df["abs_train_cv_gap"], ascending=True)
        + 0.40 * _classification_rank(df["cv_macro_f1_std"], ascending=True)
    )
    df["generalization_rank"] = df["generalization_rank_score"].rank(
        ascending=True, method="min"
    ).astype("Int64")

    return df.sort_values(
        ["statistical_rank", "cv_macro_f1", "cv_macro_f1_std"],
        ascending=[True, False, True],
        na_position="last",
    ).reset_index(drop=True)


def _classification_sensitivity_diagnosis(first_place_rate, rank_range, rank_sd):
    """Same robustness interpretation thresholds as predict_Reg.py."""
    try:
        first_place_rate = float(first_place_rate)
        rank_range = float(rank_range)
        rank_sd = float(rank_sd)
    except (TypeError, ValueError):
        return "Insufficient information"

    if first_place_rate >= 0.80 and rank_range <= 1:
        return "Highly robust"
    if first_place_rate >= 0.60 and rank_range <= 2:
        return "Moderately robust"
    if first_place_rate >= 0.40 or rank_sd <= 1.50:
        return "Weight-sensitive"
    return "Unstable selection"


def classification_predefined_weight_sensitivity(df):
    """Run the same three predefined weighting scenarios as regression."""
    if df is None or df.empty:
        return df, pd.DataFrame()

    result = df.copy()
    rank_columns = []
    scenario_rows = []

    for scenario_name, weights in CLASSIFICATION_WEIGHT_SCENARIOS.items():
        score = sum(
            float(weight) * pd.to_numeric(result[criterion], errors="coerce")
            for criterion, weight in weights.items()
        )
        score_column = f"{scenario_name}_score"
        rank_column = f"{scenario_name}_rank"
        result[score_column] = score
        result[rank_column] = score.rank(
            ascending=True, method="min", na_option="bottom"
        ).astype("Int64")
        rank_columns.append(rank_column)
        scenario_rows.append({
            "scenario": scenario_name,
            **weights,
            "total_weight": sum(weights.values()),
        })

    ranks = result[rank_columns].astype(float)
    result["sensitivity_mean_rank"] = ranks.mean(axis=1)
    result["sensitivity_median_rank"] = ranks.median(axis=1)
    result["sensitivity_min_rank"] = ranks.min(axis=1)
    result["sensitivity_max_rank"] = ranks.max(axis=1)
    result["sensitivity_rank_range"] = (
        result["sensitivity_max_rank"] - result["sensitivity_min_rank"]
    )
    result["sensitivity_rank_sd"] = ranks.std(axis=1, ddof=1)
    result["first_place_count"] = (ranks == 1).sum(axis=1)
    result["first_place_rate"] = result["first_place_count"] / float(len(rank_columns))
    result["weight_sensitivity_diagnosis"] = [
        _classification_sensitivity_diagnosis(rate, span, sd)
        for rate, span, sd in zip(
            result["first_place_rate"],
            result["sensitivity_rank_range"],
            result["sensitivity_rank_sd"],
        )
    ]
    return result, pd.DataFrame(scenario_rows)


def classification_monte_carlo_weight_sensitivity(
    df,
    n_simulations=1000,
    random_state=42,
    min_performance_weight=0.50,
    max_single_weight=0.80,
):
    """Constrained Monte Carlo robustness analysis using development metrics only.

    If cross-validated MCC is available, it joins Macro-F1 as a performance
    criterion. Test MCC and all other independent-test metrics are excluded.
    """
    if df is None or df.empty:
        return df, pd.DataFrame(), np.empty((0, 0), dtype=int)

    use_mcc = "rank_cv_mcc" in df.columns and pd.to_numeric(
        df["rank_cv_mcc"], errors="coerce"
    ).notna().any()

    if use_mcc:
        criteria = [
            "rank_cv_macro_f1", "rank_cv_mcc",
            "rank_cv_stability", "rank_train_cv_gap",
        ]
        weight_names = [
            "cv_macro_f1_weight", "cv_mcc_weight",
            "cv_sd_weight", "train_cv_gap_weight",
        ]
        performance_indices = [0, 1]
    else:
        criteria = [
            "rank_cv_macro_f1", "rank_cv_stability", "rank_train_cv_gap"
        ]
        weight_names = [
            "cv_macro_f1_weight", "cv_sd_weight", "train_cv_gap_weight"
        ]
        performance_indices = [0]

    matrix = df[criteria].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    rng = np.random.default_rng(random_state)
    accepted_weights, simulated_ranks = [], []
    attempts = 0
    max_attempts = max(10000, int(n_simulations) * 200)

    while len(accepted_weights) < int(n_simulations) and attempts < max_attempts:
        attempts += 1
        w = rng.dirichlet(np.ones(len(criteria)))
        if w[performance_indices].sum() < float(min_performance_weight):
            continue
        if float(w.max()) > float(max_single_weight):
            continue
        ranks = pd.Series(matrix @ w).rank(
            ascending=True, method="min", na_option="bottom"
        ).astype(int).to_numpy()
        accepted_weights.append(w)
        simulated_ranks.append(ranks)

    if not accepted_weights:
        raise RuntimeError("No Monte Carlo classification weights satisfied the constraints.")

    wa = np.asarray(accepted_weights, dtype=float)
    ra = np.asarray(simulated_ranks, dtype=int).T
    result = df.copy()
    result["mc_mean_rank"] = ra.mean(axis=1)
    result["mc_median_rank"] = np.median(ra, axis=1)
    result["mc_min_rank"] = ra.min(axis=1)
    result["mc_max_rank"] = ra.max(axis=1)
    result["mc_rank_range"] = result["mc_max_rank"] - result["mc_min_rank"]
    result["mc_rank_sd"] = ra.std(axis=1, ddof=1)
    result["mc_first_place_count"] = (ra == 1).sum(axis=1)
    result["mc_first_place_rate"] = result["mc_first_place_count"] / ra.shape[1]
    result["mc_top3_count"] = (ra <= 3).sum(axis=1)
    result["mc_top3_rate"] = result["mc_top3_count"] / ra.shape[1]
    result["mc_weight_sensitivity_diagnosis"] = [
        _classification_sensitivity_diagnosis(a, b, c)
        for a, b, c in zip(
            result["mc_first_place_rate"], result["mc_rank_range"], result["mc_rank_sd"]
        )
    ]

    # Robustness rank: Monte Carlo behaviour only; no test metrics.
    rank_mc_median = pd.to_numeric(result["mc_median_rank"], errors="coerce").rank(
        ascending=True, method="average", na_option="bottom"
    )
    rank_mc_first = pd.to_numeric(result["mc_first_place_rate"], errors="coerce").rank(
        ascending=False, method="average", na_option="bottom"
    )
    rank_mc_top3 = pd.to_numeric(result["mc_top3_rate"], errors="coerce").rank(
        ascending=False, method="average", na_option="bottom"
    )
    rank_mc_sd = pd.to_numeric(result["mc_rank_sd"], errors="coerce").rank(
        ascending=True, method="average", na_option="bottom"
    )
    result["robustness_rank_score"] = (
        0.45 * rank_mc_median + 0.25 * rank_mc_first
        + 0.20 * rank_mc_top3 + 0.10 * rank_mc_sd
    )
    result["robustness_rank"] = result["robustness_rank_score"].rank(
        ascending=True, method="min"
    ).astype("Int64")

    # Recommended rank combines the predeclared selection rank with robustness.
    # It remains fully development-only and cannot be optimized toward test results.
    rank_selection = pd.to_numeric(result["selection_rank_score"], errors="coerce").rank(
        ascending=True, method="average", na_option="bottom"
    )
    rank_robust = pd.to_numeric(result["robustness_rank_score"], errors="coerce").rank(
        ascending=True, method="average", na_option="bottom"
    )
    result["recommended_rank_score"] = 0.60 * rank_selection + 0.40 * rank_robust
    result["recommended_rank"] = result["recommended_rank_score"].rank(
        ascending=True, method="min"
    ).astype("Int64")

    weights_df = pd.DataFrame(wa, columns=weight_names)
    weights_df.insert(0, "simulation", np.arange(1, len(weights_df) + 1))
    weights_df["performance_weight"] = weights_df[
        [weight_names[i] for i in performance_indices]
    ].sum(axis=1)
    weights_df["mcc_included"] = bool(use_mcc)
    return result, weights_df, ra

def save_classification_enhanced_ranking(
    df,
    reports_dir,
    filename="ClassificationEnhancedFinalRanking.csv",
    n_simulations=1000,
    random_state=42,
):
    """Save classifier ranking and Monte Carlo robustness outputs.

    Output structure mirrors predict_Reg.py while retaining Classification-
    prefixed filenames to avoid collisions with regression reports.
    """
    if df is None or df.empty:
        raise ValueError("Classification ranking dataframe is empty.")

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    ranked = classification_statistical_ranking_system(df)
    ranked, scenarios_df = classification_predefined_weight_sensitivity(ranked)
    ranked, weights_df, ranks_array = classification_monte_carlo_weight_sensitivity(
        ranked,
        n_simulations=n_simulations,
        random_state=random_state,
    )

    output_path = reports_dir / filename
    alias_path = reports_dir / "Classification_EnhancedFinalRanking.csv"
    scenarios_path = reports_dir / "ClassificationWeightScenarios.csv"
    sensitivity_path = reports_dir / "ClassificationWeightSensitivityAnalysis.csv"
    weights_path = reports_dir / "ClassificationMonteCarloWeightSensitivity.csv"
    rank_matrix_path = reports_dir / "ClassificationMonteCarloRankMatrix.csv"
    summary_path = reports_dir / "ClassificationWeightSensitivitySummary.txt"
    metadata_path = reports_dir / "ClassificationEnhancedFinalRanking_metadata.json"

    ranked.to_csv(output_path, index=False, float_format="%.6f", encoding="utf-8-sig")
    if alias_path.resolve() != output_path.resolve():
        ranked.to_csv(alias_path, index=False, float_format="%.6f", encoding="utf-8-sig")
    ranked.to_csv(sensitivity_path, index=False, float_format="%.6f", encoding="utf-8-sig")
    scenarios_df.to_csv(scenarios_path, index=False)
    weights_df.to_csv(weights_path, index=False, float_format="%.6f")

    labels = ranked["model"].astype(str) + "-" + ranked["feature_set"].astype(str)
    rank_matrix_df = pd.DataFrame(
        ranks_array,
        columns=[f"simulation_{i}" for i in range(1, ranks_array.shape[1] + 1)],
    )
    rank_matrix_df.insert(0, "model_feature_set", labels.to_numpy())
    rank_matrix_df.to_csv(rank_matrix_path, index=False)

    # Same robustness-oriented ordering used by regression.
    sort_cols = ["recommended_rank", "robustness_rank", "statistical_rank"]
    sort_cols = [c for c in sort_cols if c in ranked.columns]
    sorted_result = ranked.sort_values(sort_cols, ascending=True)
    best = sorted_result.iloc[0]

    lines = [
        "CLASSIFICATION WEIGHT-SENSITIVITY ANALYSIS",
        "=" * 76,
        "",
        "Primary CV-only ranking weights:",
        "- Mean group-aware CV Macro-F1 rank: 0.60",
        "- CV Macro-F1 standard-deviation rank: 0.20",
        "- Absolute train-CV Macro-F1 gap rank: 0.20",
        "",
        f"Predefined scenarios: {len(CLASSIFICATION_WEIGHT_SCENARIOS)}",
        f"Constrained Monte Carlo simulations: {len(weights_df)}",
        "- CV Macro-F1 receives at least 50% Monte Carlo weight.",
        "- No single criterion receives more than 80% Monte Carlo weight.",
        "",
        "Most robust first-place classifier:",
        f"- Model: {best.get('model', 'N/A')}",
        f"- Feature set: {best.get('feature_set', 'N/A')}",
        f"- Primary selection rank: {best.get('statistical_rank', 'N/A')}",
        f"- First in predefined scenarios: {int(best.get('first_place_count', 0))}/"
        f"{len(CLASSIFICATION_WEIGHT_SCENARIOS)} "
        f"({float(best.get('first_place_rate', np.nan)):.1%})",
        f"- Monte Carlo median rank: {float(best.get('mc_median_rank', np.nan)):.2f}",
        f"- Monte Carlo rank SD: {float(best.get('mc_rank_sd', np.nan)):.3f}",
        f"- Monte Carlo first-place rate: {float(best.get('mc_first_place_rate', np.nan)):.1%}",
        f"- Monte Carlo top-three rate: {float(best.get('mc_top3_rate', np.nan)):.1%}",
        f"- Monte Carlo rank range: {int(best.get('mc_min_rank', 0))}-"
        f"{int(best.get('mc_max_rank', 0))}",
        f"- Diagnosis: {best.get('mc_weight_sensitivity_diagnosis', 'N/A')}",
        "",
        "Interpretation:",
        "A classifier should be described as robust only when it remains highly "
        "ranked across both predefined and constrained Monte Carlo weighting "
        "scenarios. Validation/test metrics are excluded from model selection and "
        "robustness calculations.",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    metadata = {
        "primary_rank": "statistical_rank / selection_rank",
        "test_metrics_used_for_selection": False,
        "validation_metrics_used_for_selection": False,
        "selection_weights_without_cv_mcc": {
            "cv_macro_f1": 0.60,
            "cv_macro_f1_std": 0.20,
            "absolute_train_cv_gap": 0.20,
        },
        "selection_weights_with_cv_mcc": {
            "cv_macro_f1": 0.45,
            "cv_mcc": 0.20,
            "cv_macro_f1_std": 0.15,
            "absolute_train_cv_gap": 0.20,
        },
        "predefined_weight_scenarios": CLASSIFICATION_WEIGHT_SCENARIOS,
        "monte_carlo_simulations": int(len(weights_df)),
        "monte_carlo_constraints": {
            "minimum_cv_macro_f1_weight": 0.50,
            "maximum_single_criterion_weight": 0.80,
        },
        "class_definitions": "Configured by the trained model and params.yaml",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    # Create the same two robustness figures as predict_Reg.py, when possible.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_df = ranked.copy()
        plot_df["model_label"] = labels
        plot_df = plot_df.sort_values("mc_first_place_rate", ascending=False)
        fig, ax = plt.subplots(figsize=(max(8, len(plot_df) * 0.48), 5.5))
        ax.bar(np.arange(len(plot_df)), plot_df["mc_first_place_rate"].astype(float))
        ax.set_xticks(np.arange(len(plot_df)))
        ax.set_xticklabels(plot_df["model_label"], rotation=55, ha="right")
        ax.set_ylabel("First-place frequency")
        ax.set_ylim(0, 1)
        ax.set_title("Classifier first-place frequency under constrained random weights")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(reports_dir / "ClassificationMonteCarloFirstPlaceRate.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        rank_columns = [f"{name}_rank" for name in CLASSIFICATION_WEIGHT_SCENARIOS]
        scenario_plot = ranked.copy()
        scenario_plot["model_label"] = labels
        scenario_plot = scenario_plot.sort_values("sensitivity_mean_rank", ascending=True)
        fig, ax = plt.subplots(figsize=(max(8, len(scenario_plot) * 0.48), 5.8))
        positions = np.arange(len(scenario_plot))
        for rank_column in rank_columns:
            ax.plot(
                positions,
                scenario_plot[rank_column].astype(float),
                marker="o",
                linewidth=1,
                alpha=0.75,
                label=rank_column.replace("_rank", "").replace("_", " "),
            )
        ax.set_xticks(positions)
        ax.set_xticklabels(scenario_plot["model_label"], rotation=55, ha="right")
        ax.set_ylabel("Rank (lower is better)")
        ax.invert_yaxis()
        ax.set_title("Classifier rank sensitivity across predefined weighting scenarios")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(reports_dir / "ClassificationWeightSensitivityRankDistribution.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        print(f"[WARN] Classification robustness plots were not created: {exc}")

    return ranked


def process_classification_ranking_file(
    ranking_csv,
    reports_dir=None,
    n_simulations=1000,
    random_state=42,
):
    ranking_csv = Path(ranking_csv)
    if not ranking_csv.exists():
        raise FileNotFoundError(f"Ranking CSV not found: {ranking_csv}")

    output_dir = Path(reports_dir) if reports_dir else ranking_csv.parent
    df = pd.read_csv(ranking_csv)
    return save_classification_enhanced_ranking(
        df,
        output_dir,
        n_simulations=n_simulations,
        random_state=random_state,
    )


# ============================================================
# Integrated nested-CV grouping and permutation sensitivity
# ============================================================
_PERMUTATION_MODEL_RE = re.compile(
    r"^(?P<model>.+?)_permutation_sensitivity\.csv$", re.IGNORECASE
)


def _natural_sort_key(value):
    parts = re.split(r"(\d+)", str(value))
    return tuple(int(p) if p.isdigit() else p.lower() for p in parts)


def merge_nested_cv_reports(reports_dir, task_suffix=""):
    """Merge all nested-CV fold CSVs below reports_dir/nested_cv."""
    reports_dir = Path(reports_dir)
    nested_dir = reports_dir / "nested_cv"
    if not nested_dir.exists():
        raise FileNotFoundError(f"Nested-CV directory not found: {nested_dir}")

    files = sorted(nested_dir.rglob("*_nested_cv_folds.csv"), key=lambda p: _natural_sort_key(str(p)))
    if not files:
        raise FileNotFoundError(f"No '*_nested_cv_folds.csv' files found under {nested_dir}")

    frames = []
    for csv_file in files:
        frame = pd.read_csv(csv_file)
        model = csv_file.stem.replace("_nested_cv_folds", "")
        feature_set = csv_file.parent.name
        frame.insert(0, "Feature_Set", feature_set)
        frame.insert(0, "Model", model)
        frame.insert(2, "Source_File", csv_file.name)
        frames.append(frame)

    merged = pd.concat(frames, ignore_index=True, sort=False)
    suffix = str(task_suffix or "").strip().upper()
    suffix = f"_{suffix}" if suffix else ""
    output = nested_dir / f"CV_all_nested{suffix}.csv"
    merged.to_csv(output, index=False, encoding="utf-8-sig")
    return merged, output, files


def _clean_perm_column(name):
    text = str(name).strip()
    text = re.sub(r"[%(){}\[\]/\\\-]+", "_", text)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_").lower()


def _normalize_perm_columns(df):
    df = df.copy()
    df.columns = [_clean_perm_column(c) for c in df.columns]
    aliases = {
        "features": "feature", "feature_name": "feature", "variable": "feature",
        "predictor": "feature", "input_feature": "feature",
        "importance": "mean_importance", "importance_mean": "mean_importance",
        "mean_permutation_importance": "mean_importance",
        "permutation_importance_mean": "mean_importance",
        "mean_importance_score": "mean_importance",
        "std": "importance_std", "sd": "importance_std",
        "importance_sd": "importance_std", "importance_stddev": "importance_std",
        "std_importance": "importance_std", "permutation_importance_std": "importance_std",
        "standard_deviation": "importance_std",
        "relative_importance": "relative_contribution_pct",
        "relative_contribution": "relative_contribution_pct",
        "relative_contribution_percent": "relative_contribution_pct",
        "relative_contribution_percentage": "relative_contribution_pct",
        "contribution_pct": "relative_contribution_pct",
        "contribution_percent": "relative_contribution_pct",
    }
    return df.rename(columns={c: aliases[c] for c in df.columns if c in aliases})


def _detect_perm_importance_column(df):
    for col in ["mean_importance", "importance_mean", "importance", "score", "mean_score"]:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            return col
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in {"importance_std", "relative_contribution_pct", "rank"}]
    likely = [c for c in numeric if "importance" in c or "score" in c]
    return likely[0] if likely else (numeric[0] if numeric else None)


def _relative_perm_contribution(values):
    values = pd.to_numeric(values, errors="coerce")
    positive = values.clip(lower=0)
    denom = positive.sum(skipna=True)
    if denom > 0:
        return positive / denom * 100.0
    absolute = values.abs()
    denom = absolute.sum(skipna=True)
    if denom > 0:
        return absolute / denom * 100.0
    return pd.Series(np.nan, index=values.index, dtype=float)


def _infer_perm_feature_set(path):
    for parent in [path.parent, *path.parents]:
        if re.fullmatch(r"fs\d+", parent.name, flags=re.IGNORECASE):
            return parent.name.upper()
    return "UNKNOWN"


def _infer_perm_group(path):
    parents = list(path.parents)
    for i, parent in enumerate(parents):
        if re.fullmatch(r"fs\d+", parent.name, flags=re.IGNORECASE) and i + 1 < len(parents):
            return parents[i + 1].name
    return "UNKNOWN"


def extract_permutation_sensitivity_reports(
    reports_dir,
    task="regression",
    experiment_family="GLFS",
    output_dir=None,
):
    """Consolidate model permutation-sensitivity CSVs and create a feature summary."""
    reports_dir = Path(reports_dir)
    root = reports_dir / "permutation_sensitivity"
    if not root.exists():
        raise FileNotFoundError(f"Permutation-sensitivity directory not found: {root}")

    files = sorted(
        [p for p in root.rglob("*.csv") if _PERMUTATION_MODEL_RE.match(p.name)],
        key=lambda p: _natural_sort_key(str(p)),
    )
    if not files:
        raise FileNotFoundError(f"No '*_permutation_sensitivity.csv' files found under {root}")

    frames, warnings_list = [], []
    for csv_path in files:
        try:
            match = _PERMUTATION_MODEL_RE.match(csv_path.name)
            model = match.group("model").strip().upper()
            df = pd.read_csv(csv_path)
            if df.empty:
                warnings_list.append(f"Skipped empty file: {csv_path}")
                continue
            df = _normalize_perm_columns(df)
            if "feature" not in df.columns:
                object_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
                if object_cols:
                    df = df.rename(columns={object_cols[0]: "feature"})
                else:
                    df.insert(0, "feature", [f"row_{i+1}" for i in range(len(df))])

            importance_col = _detect_perm_importance_column(df)
            if importance_col and importance_col != "mean_importance":
                df = df.rename(columns={importance_col: "mean_importance"})
            if "mean_importance" in df.columns:
                df["mean_importance"] = pd.to_numeric(df["mean_importance"], errors="coerce")
            if "importance_std" in df.columns:
                df["importance_std"] = pd.to_numeric(df["importance_std"], errors="coerce")
            if "relative_contribution_pct" not in df.columns and "mean_importance" in df.columns:
                df["relative_contribution_pct"] = _relative_perm_contribution(df["mean_importance"])
            elif "relative_contribution_pct" in df.columns:
                df["relative_contribution_pct"] = pd.to_numeric(df["relative_contribution_pct"], errors="coerce")
                finite = df["relative_contribution_pct"].dropna()
                if not finite.empty and finite.abs().max() <= 1.0:
                    df["relative_contribution_pct"] *= 100.0
            if "mean_importance" in df.columns:
                df["importance_rank_within_model"] = df["mean_importance"].rank(method="dense", ascending=False).astype("Int64")

            metadata = {
                "task": task,
                "experiment_family": experiment_family,
                "ac_group": _infer_perm_group(csv_path),
                "feature_set": _infer_perm_feature_set(csv_path),
                "model": model,
                "source_file": csv_path.name,
                "source_path": str(csv_path),
            }
            for key, value in reversed(list(metadata.items())):
                df.insert(0, key, value)
            frames.append(df)
        except Exception as exc:
            warnings_list.append(f"Failed to read {csv_path}: {exc}")

    if not frames:
        raise RuntimeError("No valid permutation-sensitivity data could be consolidated.")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    sort_cols = [c for c in ["feature_set", "model", "importance_rank_within_model", "feature"] if c in combined.columns]
    if sort_cols:
        combined = combined.sort_values(sort_cols, kind="stable")

    summary = pd.DataFrame()
    required = {"task", "experiment_family", "feature_set", "feature", "mean_importance"}
    if required.issubset(combined.columns):
        group_cols = ["task", "experiment_family", "ac_group", "feature_set", "feature"]
        summary = (
            combined.groupby(group_cols, dropna=False)
            .agg(
                number_of_models=("model", "nunique"),
                number_of_records=("mean_importance", "count"),
                mean_importance=("mean_importance", "mean"),
                median_importance=("mean_importance", "median"),
                importance_std_across_models=("mean_importance", "std"),
                minimum_importance=("mean_importance", "min"),
                maximum_importance=("mean_importance", "max"),
            ).reset_index()
        )
        summary["relative_contribution_pct"] = summary.groupby(
            ["task", "experiment_family", "ac_group", "feature_set"], group_keys=False
        )["mean_importance"].transform(_relative_perm_contribution)
        summary["importance_rank"] = summary.groupby(
            ["task", "experiment_family", "ac_group", "feature_set"]
        )["mean_importance"].rank(method="dense", ascending=False).astype("Int64")
        summary = summary.sort_values(["feature_set", "importance_rank", "feature"], kind="stable")

    out_dir = Path(output_dir) if output_dir else reports_dir / "permutation_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    task_tag = "R" if str(task).lower().startswith("reg") else "C" if str(task).lower().startswith("class") else str(task)
    combined_path = out_dir / f"permutation_sensitivity_all_models_{task_tag}.csv"
    summary_path = out_dir / f"permutation_sensitivity_feature_summary_{task_tag}.csv"
    combined.to_csv(combined_path, index=False, encoding="utf-8-sig")
    if not summary.empty:
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    return combined, summary, combined_path, summary_path, warnings_list, files


def run_integrated_report_consolidation(reports_dir, task="regression", experiment_family="GLFS"):
    """Run nested-CV grouping and permutation consolidation, independently."""
    suffix = "R" if str(task).lower().startswith("reg") else "C" if str(task).lower().startswith("class") else ""
    result = {"nested_cv": None, "permutation": None, "warnings": []}
    try:
        merged, path, files = merge_nested_cv_reports(reports_dir, suffix)
        result["nested_cv"] = {"path": str(path), "rows": len(merged), "files": len(files)}
    except Exception as exc:
        result["warnings"].append(f"Nested CV: {exc}")
    try:
        combined, summary, path, summary_path, warns, files = extract_permutation_sensitivity_reports(
            reports_dir, task=task, experiment_family=experiment_family
        )
        result["permutation"] = {
            "path": str(path), "summary_path": str(summary_path),
            "rows": len(combined), "summary_rows": len(summary), "files": len(files)
        }
        result["warnings"].extend(warns)
    except Exception as exc:
        result["warnings"].append(f"Permutation sensitivity: {exc}")
    return result


def _close_classification_gui(window):
    """
    Close the classification GUI normally.

    This avoids abnormal/forced termination codes being propagated to a
    parent launcher. In particular, the parent GUI should not receive a
    termination status such as code 15 merely because the user closed this
    window.
    """
    try:
        window.quit()
    except Exception:
        pass

    try:
        window.destroy()
    except Exception:
        pass


def launch_gui(data_dir, models_dir, reports_dir, params_path="params.yaml"):
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

    data_dir, models_dir, reports_dir = map(Path, (data_dir, models_dir, reports_dir))
    reports_dir.mkdir(parents=True, exist_ok=True)
    params = _load_params(params_path)

    root = tk.Tk()
    root.title("Classification Prediction, ROC/AUC and Reports")
    root.geometry("1480x980")
    root.protocol(
        "WM_DELETE_WINDOW",
        lambda: _close_classification_gui(root)
    )

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=6, pady=6)

    # ---------------- Single prediction tab ----------------
    prediction_tab = ttk.Frame(notebook)
    notebook.add(prediction_tab, text="Single Classification Prediction")

    model_files = sorted(models_dir.glob("*.pkl"))
    model_var = tk.StringVar(value=model_files[0].name if model_files else "")
    model_info_var = tk.StringVar(value="Select a classifier to load its required features.")

    ttk.Label(prediction_tab, text="Classifier:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
    model_combo = ttk.Combobox(
        prediction_tab,
        textvariable=model_var,
        values=[p.name for p in model_files],
        width=48,
        state="readonly",
    )
    model_combo.grid(row=0, column=1, columnspan=3, sticky="w", padx=5, pady=5)
    ttk.Label(prediction_tab, textvariable=model_info_var).grid(
        row=1, column=0, columnspan=4, sticky="w", padx=12, pady=(2, 8)
    )

    input_frame = ttk.LabelFrame(prediction_tab, text="Required model inputs")
    input_frame.grid(row=2, column=0, columnspan=4, sticky="nw", padx=8, pady=5)

    # Input controls are generated dynamically from params.yaml/model metadata.
    # No soil-specific aliases or default feature sets are used.
    fields = {}
    field_widgets = {}  # feature -> (label, entry, optional_note_label)

    result_text = tk.Text(prediction_tab, height=20, width=88, font=("Courier New", 10))
    result_text.grid(row=2, column=4, rowspan=7, columnspan=5, sticky="nsew", padx=12, pady=5)

    single_state = {"package": None, "feature_names": [], "feature_set": None}

    def _parse_field(name):
        text = fields[name].get().strip()
        return None if not text else float(text)

    def update_single_model_features(event=None):
        """Load selected classifier and display only its related raw inputs."""
        try:
            if not model_var.get():
                return
            selected = models_dir / model_var.get()
            package = load_classifier_package(selected)
            _, filename_fs, _ = infer_metadata_from_filename(selected.name)
            fs = str(package.get("feature_set") or filename_fs or "").lower()
            feature_names = _model_feature_names(
                package, fs, params=params, target=package.get("target")
            )

            single_state.update(package=package, feature_names=feature_names, feature_set=fs)

            # Hide existing controls and display exactly the model's predictors.
            for label, entry, note_label in field_widgets.values():
                label.grid_remove()
                entry.grid_remove()
                note_label.grid_remove()

            for row, name in enumerate(feature_names):
                note_text = _params_feature_note(params, name, package.get("target"))
                if name not in fields:
                    variable = tk.StringVar()
                    label = ttk.Label(input_frame, text=f"{name}:")
                    entry = ttk.Entry(input_frame, textvariable=variable, width=24)
                    note_label = ttk.Label(input_frame, text="", foreground="gray")
                    fields[name] = variable
                    field_widgets[name] = (label, entry, note_label)
                label, entry, note_label = field_widgets[name]
                note_label.configure(text=f"Note: {note_text}" if note_text else "")
                label.grid(row=row, column=0, sticky="e", padx=5, pady=4)
                entry.grid(row=row, column=1, sticky="w", padx=5, pady=4)
                if note_text:
                    note_label.grid(row=row, column=2, sticky="w", padx=(8, 5), pady=4)
                else:
                    note_label.grid_remove()

            order_text = ", ".join(feature_names)
            count = _model_expected_feature_count(package["model"])
            model_info_var.set(
                f"Model: {package['model_type']}   |   Feature set: {fs or 'unknown'}   |   "
                f"Required order from params.yaml: {order_text}   |   Expected columns: {count or len(feature_names)}"
            )
        except Exception as exc:
            single_state.update(package=None, feature_names=[], feature_set=None)
            model_info_var.set(f"Could not inspect selected model: {exc}")
            messagebox.showerror("Model feature detection", str(exc))

    def _single_class_metadata(package):
        """Resolve class display names and optional numeric boundaries from package/YAML."""
        model = package["model"]
        labels = list(getattr(model, "classes_", []))
        target = str(package.get("target") or "target")

        cfg = {}
        raw_package = package.get("raw") if isinstance(package, dict) else None
        if isinstance(raw_package, dict):
            saved_values = raw_package.get("class_boundary_values")
            if isinstance(saved_values, dict):
                cfg = dict(saved_values)

        classification = params.get("classification", {}) if isinstance(params, dict) else {}
        boundaries_root = classification.get("class_boundaries", {}) if isinstance(classification, dict) else {}
        yaml_cfg = {}
        if isinstance(boundaries_root, dict):
            yaml_cfg = next((v for k, v in boundaries_root.items() if str(k).lower() == target.lower()), {}) or {}
        if isinstance(yaml_cfg, dict):
            # YAML may provide display names while the model package preserves
            # the exact numeric boundaries resolved during training.
            merged = dict(yaml_cfg)
            merged.update(cfg)
            cfg = merged

        names_source = package.get("class_names")
        # Old packages may carry the former AC defaults; prefer YAML when available.
        if isinstance(cfg, dict) and cfg.get("class_names"):
            names_source = cfg.get("class_names")

        display = {}
        if isinstance(names_source, dict):
            for label in labels:
                display[label] = str(names_source.get(label, names_source.get(str(label), label)))
        elif isinstance(names_source, (list, tuple)):
            for i, label in enumerate(labels):
                display[label] = str(names_source[i]) if i < len(names_source) else str(label)
        else:
            display = {label: str(label) for label in labels}
        return target, labels, display, cfg

    def _single_probabilities(model, X, labels):
        if hasattr(model, "predict_proba"):
            probs = np.asarray(model.predict_proba(X), dtype=float).reshape(1, -1)[0]
            model_labels = list(getattr(model, "classes_", labels))
            return {label: float(probs[i]) for i, label in enumerate(model_labels) if i < len(probs)}
        if hasattr(model, "decision_function"):
            scores = np.asarray(model.decision_function(X), dtype=float)
            if scores.ndim == 1:
                scores = np.column_stack([-scores, scores])
            scores = scores.reshape(1, -1)[0]
            scores = scores - np.max(scores)
            probs = np.exp(scores); probs = probs / probs.sum()
            model_labels = list(getattr(model, "classes_", labels))
            return {label: float(probs[i]) for i, label in enumerate(model_labels) if i < len(probs)}
        return {}

    def run_single_prediction():
        try:
            if single_state["package"] is None:
                update_single_model_features()
            package = single_state["package"]
            feature_names = single_state["feature_names"]
            if package is None or not feature_names:
                raise ValueError("Select a valid classification model first.")

            values = {name: _parse_field(name) for name in feature_names}
            X = _build_exact_feature_row(feature_names, values)
            model = package["model"]

            expected = _model_expected_feature_count(model)
            if expected is not None and X.shape[1] != expected:
                raise ValueError(
                    f"Prepared {X.shape[1]} features, but the selected model expects {expected}. "
                    f"Resolved order: {feature_names}"
                )

            predicted = np.asarray(model.predict(X)).ravel()[0]
            target_name, labels, class_names, boundary_cfg = _single_class_metadata(package)
            probabilities = _single_probabilities(model, X, labels)
            confidence = probabilities.get(predicted)

            input_lines = "\n".join(
                f"  {name:<24}: {X[0, index]:.6f}"
                for index, name in enumerate(feature_names)
            )
            class_display = class_names.get(predicted, str(predicted))
            result = (
                "SINGLE CLASSIFICATION PREDICTION\n"
                + "=" * 60 + "\n"
                f"Model file       : {model_var.get()}\n"
                f"Model type       : {package['model_type']}\n"
                f"Target           : {target_name}\n"
                f"Feature set      : {single_state['feature_set'] or 'unknown'}\n"
                f"Feature order    : {', '.join(feature_names)}\n\n"
                f"Model inputs:\n{input_lines}\n\n"
                f"Predicted class  : {class_display} ({predicted})\n"
            )
            if confidence is not None:
                result += f"Confidence       : {confidence:.4f}\n"
            if probabilities:
                result += "\nClass probabilities:\n"
                for label in labels:
                    if label in probabilities:
                        result += f"  {class_names.get(label, str(label))}: {probabilities[label]:.4f}\n"
            if isinstance(boundary_cfg, dict) and boundary_cfg:
                shown = {k: v for k, v in boundary_cfg.items() if k != "class_names"}
                if shown:
                    result += "\nConfigured class boundaries:\n"
                    for key, value in shown.items():
                        result += f"  {key}: {value}\n"

            result_text.delete("1.0", tk.END)
            result_text.insert(tk.END, result)
        except Exception as exc:
            messagebox.showerror("Single classification prediction", str(exc))

    ttk.Button(
        prediction_tab,
        text="Predict class",
        command=run_single_prediction,
    ).grid(row=8, column=0, columnspan=4, pady=12)

    model_combo.bind("<<ComboboxSelected>>", update_single_model_features)
    prediction_tab.columnconfigure(4, weight=1)
    prediction_tab.rowconfigure(7, weight=1)
    if model_files:
        root.after(100, update_single_model_features)

    # ---------------- Model Evaluation & Ranking tab ----------------
    ranking_tab = ttk.Frame(notebook)
    notebook.add(ranking_tab, text="Model Evaluation & Ranking")
    notebook.insert(0, ranking_tab)
    # Move Model Evaluation & Ranking to the first (leftmost) tab.
    notebook.insert(0, ranking_tab)

    ranking_tab.columnconfigure(0, weight=1)
    ranking_tab.rowconfigure(1, weight=3)
    ranking_tab.rowconfigure(3, weight=2)

    source_frame = ttk.LabelFrame(
        ranking_tab,
        text="Classification evaluation source",
        padding=8,
    )
    source_frame.grid(
        row=0,
        column=0,
        columnspan=2,
        sticky="ew",
        padx=6,
        pady=(6, 4),
    )
    source_frame.columnconfigure(1, weight=1)

    default_ranking_candidates = [
        reports_dir / "all_evaluations_class.csv",
        reports_dir / "classification_evaluation_ranking.csv",
        reports_dir / "all_class_evaluations.csv",
    ]
    default_ranking_path = next(
        (path for path in default_ranking_candidates if path.exists()),
        default_ranking_candidates[0],
    )
    ranking_path_var = tk.StringVar(value=str(default_ranking_path))
    ranking_status_var = tk.StringVar(
        value="Load the all-classifier evaluation report."
    )
    sensitivity_simulations_var = tk.IntVar(value=1000)
    sensitivity_seed_var = tk.IntVar(value=42)

    ttk.Label(source_frame, text="Evaluation CSV:").grid(
        row=0, column=0, sticky="e", padx=(0, 6), pady=3
    )
    ttk.Entry(
        source_frame,
        textvariable=ranking_path_var,
        width=95,
    ).grid(row=0, column=1, sticky="ew", pady=3)

    def browse_ranking_csv():
        selected = filedialog.askopenfilename(
            title="Select all-classifier evaluation CSV",
            initialdir=str(reports_dir),
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if selected:
            ranking_path_var.set(selected)

    ttk.Button(
        source_frame,
        text="Browse",
        command=browse_ranking_csv,
    ).grid(row=0, column=2, padx=(6, 0), pady=3)

    ttk.Label(source_frame, text="Monte Carlo simulations:").grid(
        row=1, column=0, sticky="e", padx=(0, 6), pady=3
    )
    ttk.Entry(
        source_frame,
        textvariable=sensitivity_simulations_var,
        width=12,
    ).grid(row=1, column=1, sticky="w", pady=3)

    ttk.Label(source_frame, text="Random seed:").grid(
        row=1, column=1, sticky="e", padx=(0, 95), pady=3
    )
    ttk.Entry(
        source_frame,
        textvariable=sensitivity_seed_var,
        width=12,
    ).grid(row=1, column=2, sticky="w", pady=3)

    ranking_columns = (
        "model",
        "feature_set",
        "statistical_rank",
        "test_performance_rank",
        "generalization_rank",
        "selection_rank_score",
        "train_macro_f1",
        "cv_macro_f1",
        "cv_macro_f1_std",
        "val_macro_f1",
        "test_macro_f1",
        "train_cv_gap",
        "cv_val_gap",
        "val_test_gap",
        "first_place_rate",
        "mc_first_place_rate",
        "mc_top3_rate",
        "generalization_diagnosis",
        "mc_weight_sensitivity_diagnosis",
        "file",
    )

    ranking_tree_frame = ttk.Frame(ranking_tab)
    ranking_tree_frame.grid(
        row=1,
        column=0,
        columnspan=2,
        sticky="nsew",
        padx=6,
        pady=4,
    )
    ranking_tree_frame.columnconfigure(0, weight=1)
    ranking_tree_frame.rowconfigure(0, weight=1)

    ranking_tree = ttk.Treeview(
        ranking_tree_frame,
        columns=ranking_columns,
        show="headings",
        height=14,
    )
    ranking_tree.grid(row=0, column=0, sticky="nsew")

    vertical_scroll = ttk.Scrollbar(
        ranking_tree_frame,
        orient="vertical",
        command=ranking_tree.yview,
    )
    vertical_scroll.grid(row=0, column=1, sticky="ns")
    horizontal_scroll = ttk.Scrollbar(
        ranking_tree_frame,
        orient="horizontal",
        command=ranking_tree.xview,
    )
    horizontal_scroll.grid(row=1, column=0, sticky="ew")
    ranking_tree.configure(
        yscrollcommand=vertical_scroll.set,
        xscrollcommand=horizontal_scroll.set,
    )

    ranking_widths = {
        "model": 105,
        "feature_set": 75,
        "statistical_rank": 95,
        "test_performance_rank": 105,
        "generalization_rank": 105,
        "selection_rank_score": 105,
        "train_macro_f1": 95,
        "cv_macro_f1": 95,
        "cv_macro_f1_std": 95,
        "val_macro_f1": 95,
        "test_macro_f1": 95,
        "train_cv_gap": 90,
        "cv_val_gap": 90,
        "val_test_gap": 90,
        "first_place_rate": 95,
        "mc_first_place_rate": 110,
        "mc_top3_rate": 90,
        "generalization_diagnosis": 185,
        "mc_weight_sensitivity_diagnosis": 175,
        "file": 220,
    }
    ranking_headings = {
        "statistical_rank": "SELECTION RANK",
        "test_performance_rank": "TEST RANK",
        "generalization_rank": "GENERALIZATION RANK",
        "selection_rank_score": "SELECTION SCORE",
        "cv_macro_f1": "CV MACRO-F1",
        "cv_macro_f1_std": "CV SD",
        "train_macro_f1": "TRAIN MACRO-F1",
        "val_macro_f1": "VAL MACRO-F1",
        "test_macro_f1": "TEST MACRO-F1",
        "first_place_rate": "SCENARIO FIRST RATE",
        "mc_first_place_rate": "MC FIRST RATE",
        "mc_top3_rate": "MC TOP-3 RATE",
        "mc_weight_sensitivity_diagnosis": "WEIGHT ROBUSTNESS",
    }

    for column in ranking_columns:
        ranking_tree.heading(
            column,
            text=ranking_headings.get(
                column,
                column.replace("_", " ").upper(),
            ),
        )
        ranking_tree.column(
            column,
            width=ranking_widths.get(column, 100),
            minwidth=65,
            anchor="center",
        )

    ranking_button_frame = ttk.Frame(ranking_tab)
    ranking_button_frame.grid(
        row=2,
        column=0,
        columnspan=2,
        sticky="ew",
        padx=6,
        pady=4,
    )

    ranking_summary_frame = ttk.LabelFrame(
        ranking_tab,
        text="CV-only classifier ranking & Monte Carlo robustness summary",
        padding=6,
    )
    ranking_summary_frame.grid(
        row=3,
        column=0,
        columnspan=2,
        sticky="nsew",
        padx=6,
        pady=(4, 6),
    )
    ranking_summary_frame.columnconfigure(0, weight=1)
    ranking_summary_frame.rowconfigure(0, weight=1)

    ranking_summary_text = tk.Text(
        ranking_summary_frame,
        wrap=tk.WORD,
        height=13,
        font=("Courier New", 9),
    )
    ranking_summary_text.grid(row=0, column=0, sticky="nsew")
    ranking_summary_scroll = ttk.Scrollbar(
        ranking_summary_frame,
        orient="vertical",
        command=ranking_summary_text.yview,
    )
    ranking_summary_scroll.grid(row=0, column=1, sticky="ns")
    ranking_summary_text.configure(
        yscrollcommand=ranking_summary_scroll.set
    )

    ranking_state = {"df": None}

    def _ranking_value(value, digits=4, percentage=False):
        if value is None or pd.isna(value):
            return ""
        try:
            numeric = float(value)
            if percentage:
                return f"{numeric:.1%}"
            return f"{numeric:.{digits}f}"
        except (TypeError, ValueError):
            return str(value)

    def _classification_ranking_summary(df):
        if df is None or df.empty:
            return "No classifier-ranking data are available."

        ordered = df.sort_values(
            ["statistical_rank", "cv_macro_f1", "cv_macro_f1_std"],
            ascending=[True, False, True],
            na_position="last",
        )
        best = ordered.iloc[0]

        lines = [
            "SCIENTIFIC CLASSIFIER SELECTION",
            "=" * 76,
            "",
            "Primary CV-only selection weights:",
            "  60% mean group-aware CV Macro-F1 rank",
            "  20% CV Macro-F1 standard-deviation rank",
            "  20% absolute train-CV Macro-F1 gap rank",
            "",
            "Validation/test metrics are excluded from selection and Monte Carlo robustness.",
            "",
            "Top selected classifier:",
            f"  Model: {best.get('model', 'N/A')}",
            f"  Feature set: {best.get('feature_set', 'N/A')}",
            f"  Selection rank: {best.get('statistical_rank', 'N/A')}",
            f"  Selection score: "
            f"{_ranking_value(best.get('selection_rank_score'), 4)}",
            f"  Train Macro-F1: "
            f"{_ranking_value(best.get('train_macro_f1'), 4)}",
            f"  CV Macro-F1: "
            f"{_ranking_value(best.get('cv_macro_f1'), 4)} "
            f"(SD={_ranking_value(best.get('cv_macro_f1_std'), 4)})",
            f"  Validation Macro-F1: "
            f"{_ranking_value(best.get('val_macro_f1'), 4)}",
            f"  Test Macro-F1: "
            f"{_ranking_value(best.get('test_macro_f1'), 4)}",
            f"  Train-CV gap: "
            f"{_ranking_value(best.get('train_cv_gap'), 4)}",
            f"  Generalization diagnosis: "
            f"{best.get('generalization_diagnosis', 'N/A')}",
            "",
            "WEIGHT-SENSITIVITY ANALYSIS",
            "=" * 76,
            f"  Predefined-scenario first-place rate: "
            f"{_ranking_value(best.get('first_place_rate'), percentage=True)}",
            f"  Monte Carlo first-place rate: "
            f"{_ranking_value(best.get('mc_first_place_rate'), percentage=True)}",
            f"  Monte Carlo top-three rate: "
            f"{_ranking_value(best.get('mc_top3_rate'), percentage=True)}",
            f"  Weight robustness: "
            f"{best.get('mc_weight_sensitivity_diagnosis', 'N/A')}",
            "",
            f"Classifiers compared: {len(df)}",
            "Class definitions are taken from the trained model/params.yaml.",
        ]
        return "\n".join(lines)

    def _populate_ranking_tree(df):
        ranking_tree.delete(*ranking_tree.get_children())

        ordered = df.sort_values(
            ["statistical_rank", "cv_macro_f1", "cv_macro_f1_std"],
            ascending=[True, False, True],
            na_position="last",
        ).reset_index(drop=True)

        for index, row in ordered.iterrows():
            values = (
                row.get("model", ""),
                row.get("feature_set", row.get("fs", "")),
                row.get("statistical_rank", ""),
                row.get("test_performance_rank", ""),
                row.get("generalization_rank", ""),
                _ranking_value(row.get("selection_rank_score"), 4),
                _ranking_value(row.get("train_macro_f1"), 4),
                _ranking_value(row.get("cv_macro_f1"), 4),
                _ranking_value(row.get("cv_macro_f1_std"), 4),
                _ranking_value(row.get("val_macro_f1"), 4),
                _ranking_value(row.get("test_macro_f1"), 4),
                _ranking_value(row.get("train_cv_gap"), 4),
                _ranking_value(row.get("cv_val_gap"), 4),
                _ranking_value(row.get("val_test_gap"), 4),
                _ranking_value(
                    row.get("first_place_rate"),
                    percentage=True,
                ),
                _ranking_value(
                    row.get("mc_first_place_rate"),
                    percentage=True,
                ),
                _ranking_value(
                    row.get("mc_top3_rate"),
                    percentage=True,
                ),
                row.get("generalization_diagnosis", ""),
                row.get("mc_weight_sensitivity_diagnosis", ""),
                row.get("file", ""),
            )
            tags = ("best_classifier",) if index == 0 else ()
            ranking_tree.insert("", "end", values=values, tags=tags)

        ranking_tree.tag_configure(
            "best_classifier",
            background="#d9f2d9",
            font=("TkDefaultFont", 9, "bold"),
        )

    def load_and_rank_classifiers():
        try:
            ranking_csv = Path(ranking_path_var.get())
            if not ranking_csv.exists():
                raise FileNotFoundError(
                    f"Evaluation CSV not found: {ranking_csv}"
                )

            simulations = int(sensitivity_simulations_var.get())
            if simulations < 1:
                raise ValueError(
                    "Monte Carlo simulations must be at least 1."
                )

            source_df = pd.read_csv(ranking_csv)
            ranked_df = classification_statistical_ranking_system(source_df)
            ranked_df, _ = classification_predefined_weight_sensitivity(
                ranked_df
            )
            ranked_df, _, _ = (
                classification_monte_carlo_weight_sensitivity(
                    ranked_df,
                    n_simulations=simulations,
                    random_state=int(sensitivity_seed_var.get()),
                )
            )

            ranking_state["df"] = ranked_df
            _populate_ranking_tree(ranked_df)
            ranking_summary_text.delete("1.0", tk.END)
            ranking_summary_text.insert(
                tk.END,
                _classification_ranking_summary(ranked_df),
            )
            ranking_status_var.set(
                f"Loaded and ranked {len(ranked_df)} classifiers from "
                f"{ranking_csv.name}."
            )
        except Exception as exc:
            ranking_status_var.set(f"Ranking error: {exc}")
            messagebox.showerror("Classifier ranking", str(exc))

    def save_enhanced_classifier_ranking():
        try:
            ranking_csv = Path(ranking_path_var.get())
            if not ranking_csv.exists():
                raise FileNotFoundError(
                    f"Evaluation CSV not found: {ranking_csv}"
                )

            ranked_df = process_classification_ranking_file(
                ranking_csv,
                reports_dir=reports_dir,
                n_simulations=int(sensitivity_simulations_var.get()),
                random_state=int(sensitivity_seed_var.get()),
            )
            ranking_state["df"] = ranked_df
            _populate_ranking_tree(ranked_df)
            ranking_summary_text.delete("1.0", tk.END)
            ranking_summary_text.insert(
                tk.END,
                _classification_ranking_summary(ranked_df),
            )
            output = reports_dir / "ClassificationEnhancedFinalRanking.csv"
            ranking_status_var.set(
                f"Enhanced ranking and sensitivity reports saved to "
                f"{reports_dir}."
            )
            messagebox.showinfo(
                "Classifier ranking saved",
                f"Enhanced classifier ranking saved to:\n{output}",
            )
        except Exception as exc:
            ranking_status_var.set(f"Save error: {exc}")
            messagebox.showerror("Save classifier ranking", str(exc))

    def export_visible_classifier_ranking():
        try:
            ranked_df = ranking_state.get("df")
            if ranked_df is None or ranked_df.empty:
                raise RuntimeError(
                    "Load or generate a classifier ranking first."
                )

            selected = filedialog.asksaveasfilename(
                title="Export classifier ranking",
                initialdir=str(reports_dir),
                initialfile="ClassificationEnhancedFinalRanking.csv",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            )
            if not selected:
                return

            ranked_df.to_csv(
                selected,
                index=False,
                float_format="%.6f",
                encoding="utf-8-sig",
            )
            ranking_status_var.set(
                f"Visible classifier ranking exported to "
                f"{Path(selected).name}."
            )
        except Exception as exc:
            messagebox.showerror("Export classifier ranking", str(exc))

    ttk.Button(
        ranking_button_frame,
        text="Load & Rank",
        command=load_and_rank_classifiers,
    ).pack(side="left", padx=(0, 8))
    ttk.Button(
        ranking_button_frame,
        text="Generate Enhanced Reports",
        command=save_enhanced_classifier_ranking,
    ).pack(side="left", padx=(0, 8))
    ttk.Button(
        ranking_button_frame,
        text="Export Visible Ranking",
        command=export_visible_classifier_ranking,
    ).pack(side="left", padx=(0, 8))
    ttk.Label(
        ranking_button_frame,
        textvariable=ranking_status_var,
    ).pack(side="left", padx=(12, 0))

    if default_ranking_path.exists():
        root.after(250, load_and_rank_classifiers)

    # Open the GUI on Model Evaluation & Ranking by default.
    notebook.select(ranking_tab)

    # ---------------- Evaluation tab ----------------
    eval_tab = ttk.Frame(notebook)
    notebook.add(eval_tab, text="Classification Evaluation")
    notebook.insert(1, eval_tab)
    path_var = tk.StringVar(value=str(reports_dir / "classification_predictions_long.csv"))
    ttk.Label(eval_tab, text="Prediction file:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
    ttk.Entry(eval_tab, textvariable=path_var, width=90).grid(row=0, column=1, columnspan=6, sticky="ew", padx=5)

    def browse_predictions():
        selected = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        if selected:
            path_var.set(selected)

    ttk.Button(eval_tab, text="Browse", command=browse_predictions).grid(row=0, column=7, padx=5)

    file_var = tk.StringVar()
    split_var = tk.StringVar(value="test")
    ttk.Label(eval_tab, text="Model file:").grid(row=1, column=0, sticky="e", padx=5)
    file_combo = ttk.Combobox(eval_tab, textvariable=file_var, width=45, state="readonly")
    file_combo.grid(row=1, column=1, columnspan=3, sticky="w", padx=5)
    ttk.Label(eval_tab, text="Split:").grid(row=1, column=4, sticky="e", padx=5)
    split_combo = ttk.Combobox(eval_tab, textvariable=split_var,
                               values=["train", "val", "test", "all"], width=10, state="readonly")
    split_combo.grid(row=1, column=5, sticky="w", padx=5)

    state = {"df": None}
    figures = []
    axes = []
    canvases = []
    for index, (row, col) in enumerate([(3, 0), (3, 4), (4, 0), (4, 4)]):
        frame = ttk.Frame(eval_tab)
        frame.grid(row=row, column=col, columnspan=4, sticky="nsew", padx=5, pady=5)
        fig = Figure(figsize=(6.1, 3.5), dpi=100)
        ax = fig.add_subplot(111)
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(canvas, frame).update()
        figures.append(fig); axes.append(ax); canvases.append(canvas)

    report_text = tk.Text(eval_tab, height=12, font=("Courier New", 9))
    report_text.grid(row=5, column=0, columnspan=8, sticky="nsew", padx=5, pady=5)

    for col in range(8):
        eval_tab.columnconfigure(col, weight=1)
    for row in [3, 4, 5]:
        eval_tab.rowconfigure(row, weight=1)

    def filtered_data():
        df = state["df"]
        if df is None:
            raise RuntimeError("Load classification_predictions_long.csv first.")
        selected_file = file_var.get()
        if selected_file and "file" in df.columns:
            df = df[df["file"].astype(str) == selected_file]
        if split_var.get() != "all" and "split" in df.columns:
            df = df[df["split"].astype(str) == split_var.get()]
        if df.empty:
            raise RuntimeError("No rows match the selected classifier and split.")
        return df

    def refresh_plots(*_):
        try:
            df = filtered_data()
            y_true = df["y_true"].to_numpy(int)
            y_pred = df["y_pred"].to_numpy(int)
            probabilities = _classification_probabilities_from_df(df)
            metrics = calculate_classification_metrics(y_true, y_pred, probabilities)

            # Four diagnostics. Probability-dependent plots automatically
            # fall back to hard-prediction diagnostics instead of going blank.
            for ax in axes:
                ax.clear()
            _draw_classification_confusion(axes[0], y_true, y_pred)
            _draw_classification_roc_or_fallback(axes[1], y_true, y_pred, probabilities)
            _draw_classification_classwise(axes[2], metrics)
            _draw_classification_confidence_or_fallback(axes[3], y_true, y_pred, probabilities)

            for fig, canvas in zip(figures, canvases):
                fig.tight_layout(); canvas.draw()

            probability_note = (
                "ROC source: per-class probabilities from classification_predictions_long.csv"
                if probabilities is not None
                else "ROC source: unavailable (probability columns missing)"
            )
            summary = (
                f"Classifier: {file_var.get()} | Split: {split_var.get()} | N={len(df)}\n"
                f"{probability_note}\n"
                f"Accuracy={metrics['accuracy']:.4f} | Balanced accuracy={metrics['balanced_accuracy']:.4f}\n"
                f"Macro precision={metrics['macro_precision']:.4f} | Macro recall={metrics['macro_recall']:.4f}\n"
                f"Macro F1={metrics['macro_f1']:.4f} | Weighted F1={metrics['weighted_f1']:.4f}\n"
                f"MCC={metrics['mcc']:.4f} | Cohen kappa={metrics['cohen_kappa']:.4f}\n"
                f"Macro OVR AUC={metrics.get('roc_auc_ovr_macro', np.nan):.4f} | "
                f"Weighted OVR AUC={metrics.get('roc_auc_ovr_weighted', np.nan):.4f}\n"
                f"Log loss={metrics.get('log_loss', np.nan):.4f}\n\n"
                "Class definitions are taken from the trained model/params.yaml."
            )
            report_text.delete("1.0", tk.END); report_text.insert(tk.END, summary)
        except Exception as exc:
            report_text.delete("1.0", tk.END); report_text.insert(tk.END, f"Evaluation error: {exc}")

    def load_prediction_file():
        try:
            df = pd.read_csv(path_var.get())
            required = {"y_true", "y_pred"}
            if not required.issubset(df.columns):
                raise ValueError(f"Missing required columns: {sorted(required - set(df.columns))}")
            state["df"] = df
            files = sorted(df["file"].astype(str).unique()) if "file" in df.columns else [""]
            file_combo["values"] = files
            if files:
                file_var.set(files[0])

            # Verify ROC/AUC inputs at load time.
            lookup = {str(c).strip().lower() for c in df.columns}
            required_probs = {"prob_inactive", "prob_normal", "prob_active"}
            missing_probs = sorted(required_probs - lookup)
            if missing_probs:
                report_text.delete("1.0", tk.END)
                report_text.insert(
                    tk.END,
                    "Prediction file loaded, but genuine ROC/AUC cannot be plotted yet.\n"
                    f"Missing probability columns: {', '.join(missing_probs)}\n\n"
                    "Re-run evaluate_all_class.py with the corrected evaluate_class.py "
                    "to regenerate classification_predictions_long.csv."
                )

            refresh_plots()
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))

    def save_paper_report():
        try:
            df = filtered_data()
            model_name = file_var.get() or "classifier"
            split_name = split_var.get() or "all"
            result = generate_paper_classification_report(
                df, reports_dir, model_name=model_name, split_name=split_name
            )
            messagebox.showinfo(
                "Paper report created",
                f"Publication-ready classification report saved to:\n{result['output_dir']}\n\n"
                f"Combined 2x2 PNG:\n{result['all_plots_png']}"
            )
        except Exception as exc:
            messagebox.showerror("Paper report", str(exc))

    def save_all_plots_png():
        try:
            df = filtered_data()
            model_name = file_var.get() or "classifier"
            split_name = split_var.get() or "all"
            selected = filedialog.asksaveasfilename(
                title="Save all classification plots as one PNG",
                initialdir=str(reports_dir),
                initialfile=f"{_safe_path_component(model_name)}_{_safe_path_component(split_name)}_all_plots.png",
                defaultextension=".png",
                filetypes=[("PNG image", "*.png"), ("All files", "*.*")],
            )
            if not selected:
                return
            output = save_classification_dashboard_png(
                df, selected, model_name=model_name, split_name=split_name, dpi=600
            )
            messagebox.showinfo("Plots saved", f"All four plots saved in one PNG:\n{output}")
        except Exception as exc:
            messagebox.showerror("Save all plots", str(exc))

    ttk.Button(eval_tab, text="Load and plot", command=load_prediction_file).grid(row=1, column=6, padx=5)
    ttk.Button(eval_tab, text="Save paper report", command=save_paper_report).grid(row=1, column=7, padx=5)
    ttk.Button(eval_tab, text="Save all plots PNG", command=save_all_plots_png).grid(row=2, column=7, padx=5, pady=(0, 4), sticky="e")
    file_combo.bind("<<ComboboxSelected>>", refresh_plots)
    split_combo.bind("<<ComboboxSelected>>", refresh_plots)


    # ---------------- Consolidated analysis outputs tab ----------------
    consolidation_tab = ttk.Frame(notebook)
    notebook.add(consolidation_tab, text="Nested CV & Permutation")
    consolidation_tab.columnconfigure(0, weight=1)
    consolidation_tab.rowconfigure(2, weight=1)
    ttk.Label(
        consolidation_tab,
        text="Build consolidated nested-CV and permutation-sensitivity CSV reports from the current reports directory.",
        wraplength=1100,
    ).grid(row=0, column=0, sticky="w", padx=12, pady=(12, 8))
    consolidation_status = tk.StringVar(value=f"Reports directory: {reports_dir}")
    ttk.Label(consolidation_tab, textvariable=consolidation_status).grid(row=1, column=0, sticky="w", padx=12)
    consolidation_text = tk.Text(consolidation_tab, height=22, wrap="word", font=("Courier New", 9))
    consolidation_text.grid(row=2, column=0, sticky="nsew", padx=12, pady=8)

    def _show_consolidation_result(result):
        lines = ["INTEGRATED REPORT CONSOLIDATION", "=" * 72]
        nested = result.get("nested_cv")
        if nested:
            lines += ["", "Nested CV grouping:", f"  Files merged: {nested['files']}", f"  Rows: {nested['rows']}", f"  Output: {nested['path']}"]
        perm = result.get("permutation")
        if perm:
            lines += ["", "Permutation sensitivity:", f"  Files merged: {perm['files']}", f"  Rows: {perm['rows']}", f"  Summary rows: {perm['summary_rows']}", f"  Combined: {perm['path']}", f"  Summary: {perm['summary_path']}"]
        if result.get("warnings"):
            lines += ["", "Warnings:"] + [f"  - {w}" for w in result["warnings"]]
        consolidation_text.delete("1.0", tk.END)
        consolidation_text.insert(tk.END, "\n".join(lines))
        consolidation_status.set("Consolidation completed." if nested or perm else "No outputs were generated.")

    def _run_all_consolidation():
        result = run_integrated_report_consolidation(reports_dir, task="classification", experiment_family="GLFS")
        _show_consolidation_result(result)

    def _run_nested_only():
        try:
            merged, output, files = merge_nested_cv_reports(reports_dir, "C")
            _show_consolidation_result({"nested_cv": {"path": str(output), "rows": len(merged), "files": len(files)}, "permutation": None, "warnings": []})
        except Exception as exc:
            messagebox.showerror("Nested CV grouping", str(exc))

    def _run_permutation_only():
        try:
            combined, summary, output, summary_output, warns, files = extract_permutation_sensitivity_reports(
                reports_dir, task="classification", experiment_family="GLFS"
            )
            _show_consolidation_result({"nested_cv": None, "permutation": {"path": str(output), "summary_path": str(summary_output), "rows": len(combined), "summary_rows": len(summary), "files": len(files)}, "warnings": warns})
        except Exception as exc:
            messagebox.showerror("Permutation sensitivity", str(exc))

    consolidation_buttons = ttk.Frame(consolidation_tab)
    consolidation_buttons.grid(row=3, column=0, sticky="w", padx=12, pady=(0, 12))
    ttk.Button(consolidation_buttons, text="Merge Nested CV", command=_run_nested_only).pack(side="left", padx=(0, 8))
    ttk.Button(consolidation_buttons, text="Extract Permutation Sensitivity", command=_run_permutation_only).pack(side="left", padx=(0, 8))
    ttk.Button(consolidation_buttons, text="Run Both", command=_run_all_consolidation).pack(side="left")

    root.mainloop()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Predict, evaluate, rank, and test weight sensitivity for "
            "classification models."
        )
    )
    parser.add_argument("--gui", action="store_true", help="Launch classification GUI")
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--models_dir", default="models_class")
    parser.add_argument("--reports_dir", default="reports_class")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument(
        "--prediction_csv",
        help="Evaluate a predictions_long_class.csv file without GUI",
    )
    parser.add_argument(
        "--ranking_csv",
        help=(
            "Classification evaluation/ranking CSV. Generates "
            "ClassificationEnhancedFinalRanking.csv and sensitivity reports."
        ),
    )
    parser.add_argument(
        "--sensitivity_simulations",
        type=int,
        default=1000,
        help="Number of constrained Monte Carlo weight simulations.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for weight-sensitivity analysis.",
    )
    parser.add_argument("--merge-nested-cv", action="store_true", help="Merge reports_dir/nested_cv/*_nested_cv_folds.csv and exit")
    parser.add_argument("--extract-permutation", action="store_true", help="Consolidate permutation-sensitivity CSV files and exit")
    parser.add_argument("--consolidate-analysis", action="store_true", help="Run both nested-CV grouping and permutation-sensitivity consolidation and exit")
    args = parser.parse_args()

    if args.merge_nested_cv:
        merged, output, files = merge_nested_cv_reports(args.reports_dir, "C")
        print(f"Merged {len(files)} nested-CV files ({len(merged)} rows) -> {output}")
        return
    if args.extract_permutation:
        combined, summary, output, summary_output, warns, files = extract_permutation_sensitivity_reports(args.reports_dir, task="classification", experiment_family="GLFS")
        print(f"Merged {len(files)} permutation files ({len(combined)} rows) -> {output}")
        print(f"Feature summary: {len(summary)} rows -> {summary_output}")
        for warning in warns: print(f"[WARN] {warning}")
        return
    if args.consolidate_analysis:
        print(json.dumps(run_integrated_report_consolidation(args.reports_dir, task="classification", experiment_family="GLFS"), indent=2))
        return

    if args.gui:
        launch_gui(args.data, args.models_dir, args.reports_dir, args.params)
        return
    if args.prediction_csv:
        metrics = evaluate_prediction_file(args.prediction_csv, args.reports_dir)
        print(json.dumps(metrics, indent=2))
        return
    if args.ranking_csv:
        ranked = process_classification_ranking_file(
            args.ranking_csv,
            reports_dir=args.reports_dir,
            n_simulations=args.sensitivity_simulations,
            random_state=args.random_state,
        )
        print(
            f"Saved classification ranking for {len(ranked)} classifiers to "
            f"{Path(args.reports_dir) / 'ClassificationEnhancedFinalRanking.csv'}"
        )
        return
    print(
        "Use --gui, --prediction_csv, or --ranking_csv. "
        "For scientific classification ranking, provide the CSV generated by "
        "evaluate_all_class.py."
    )


if __name__ == "__main__":
    main()
