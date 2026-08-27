from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, StratifiedKFold, RepeatedStratifiedKFold,
    GroupKFold, StratifiedGroupKFold, cross_validate, learning_curve
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

# Reuse exactly the same preprocessing builder used by regression training.
try:
    from src.train import build_preprocessor as create_preprocessing_pipeline
except ImportError:
    from train import build_preprocessor as create_preprocessing_pipeline


# Support both ``python src/trainclass4.py`` and ``python -m src.trainclass4``.
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
for _path in (SRC_DIR, PROJECT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# Import directly from ``registry`` so this works even when Windows/Python
# treats ``train_model_code_class`` as a namespace package because an
# ``__init__.py`` file was not copied correctly.
try:
    from train_model_code_class.registry import (
        available_classifiers,
        load_classifier_spec,
        validate_requested_classifiers,
    )
except ImportError:
    from src.train_model_code_class.registry import (
        available_classifiers,
        load_classifier_spec,
        validate_requested_classifiers,
    )


@dataclass
class CVPlan:
    tuning_cv: Any
    stability_cv: Any
    groups: np.ndarray | None
    mode: str
    folds: int
    repeats: int


@dataclass
class OutputPaths:
    data_dir: Path
    models_dir: Path
    reports_dir: Path


CLASS_NAMES = {0: "Inactive", 1: "Normal", 2: "Active"}
CLASS_LABELS = [0, 1, 2]


class ClassificationBoundaryCancelled(Exception):
    """Raised when the user declines automatic class-boundary inference."""


def _choose_boundary_action(
    target: str,
    reason: str,
    *,
    allow_user_limits: bool,
    lower_value: float | None = None,
    upper_value: float | None = None,
) -> str:
    """Return one of: 'user', 'automatic', or 'exit'.

    'user' is available only when the configured limits are numeric, finite,
    and ordered (lower < upper), even if their resulting training-class counts
    are not considered CV-safe.
    """
    message = (
        f"Class limits for target '{target}' need attention.\n\n"
        f"Reason: {reason}\n\n"
    )

    if allow_user_limits:
        message += (
            f"Your configured limits: lower={lower_value:g}, upper={upper_value:g}\n\n"
            "Choose how to continue:\n"
            "• Use My Limits: keep exactly the limits from params.yaml.\n"
            "• Use Automatic Limits: calculate limits from TRAINING target values only.\n"
            "• Exit: stop classification training cleanly.\n\n"
        )
    else:
        message += (
            "The configured limits cannot be used as written because they are missing, \n"
            "non-numeric, non-finite, or lower is not smaller than upper.\n\n"
            "Choose how to continue:\n"
            "• Use Automatic Limits: calculate limits from TRAINING target values only.\n"
            "• Exit: stop classification training cleanly.\n\n"
        )

    message += (
        "Automatic mode briefly: uses configured quantiles (default 1/3 and 2/3); \n"
        "if ties make them unusable, it chooses CV-safe cut points between ordered \n"
        "unique training values. Validation and test data are never used."
    )

    try:
        import tkinter as tk
        from tkinter import ttk

        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass

        dialog = tk.Toplevel(root)
        dialog.title("Classification Limits Required")
        dialog.resizable(False, False)
        try:
            dialog.attributes("-topmost", True)
        except Exception:
            pass
        dialog.grab_set()

        frame = ttk.Frame(dialog, padding=14)
        frame.pack(fill="both", expand=True)
        ttk.Label(
            frame,
            text=message,
            justify="left",
            wraplength=620,
        ).pack(anchor="w")

        result = {"choice": "exit"}

        buttons = ttk.Frame(frame)
        buttons.pack(fill="x", pady=(14, 0))

        def choose(value: str) -> None:
            result["choice"] = value
            dialog.destroy()

        ttk.Button(
            buttons,
            text="Exit",
            command=lambda: choose("exit"),
        ).pack(side="right", padx=(6, 0))

        ttk.Button(
            buttons,
            text="Use Automatic Limits",
            command=lambda: choose("automatic"),
        ).pack(side="right", padx=(6, 0))

        if allow_user_limits:
            ttk.Button(
                buttons,
                text="Use My Limits",
                command=lambda: choose("user"),
            ).pack(side="right", padx=(6, 0))

        dialog.protocol("WM_DELETE_WINDOW", lambda: choose("exit"))
        dialog.update_idletasks()
        x = max(20, (dialog.winfo_screenwidth() - dialog.winfo_width()) // 2)
        y = max(20, (dialog.winfo_screenheight() - dialog.winfo_height()) // 3)
        dialog.geometry(f"+{x}+{y}")
        root.wait_window(dialog)
        root.destroy()
        return result["choice"]

    except Exception as error:
        print(
            "[CLASS][WARN] Could not open the class-boundary choice window: "
            f"{error}"
        )
        print(
            "[CLASS][WARN] Classification will stop rather than change or "
            "override limits without user consent."
        )
        return "exit"



def _show_classification_cv_error(target: str, error: Exception, counts: list[int] | None = None) -> None:
    """Show a blocking warning for an unusable classification/CV setup.

    The warning is informational only. After the user acknowledges it, the
    classification stage terminates cleanly instead of raising a traceback.
    """
    count_text = f"\n\nTraining class counts: {counts}" if counts is not None else ""
    message = (
        f"Classification cannot continue for target '{target}'.\n\n"
        f"Reason: {error}{count_text}\n\n"
        "The selected class limits do not provide enough observations for "
        "the required stratified cross-validation. Classification training "
        "will now terminate cleanly.\n\n"
        "To continue later, adjust lower/upper limits in params.yaml so that "
        "all three classes contain sufficient training samples, or choose "
        "automatic limits when the earlier boundary warning is shown."
    )
    print(f"[CLASS][WARN] {message.replace(chr(10), ' ')}")
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        messagebox.showwarning(
            "Classification Cannot Continue",
            message,
            parent=root,
        )
        root.destroy()
    except Exception as gui_error:
        print(
            "[CLASS][WARN] Could not open CV warning window: "
            f"{gui_error}"
        )


def _normalize_class_names(value: Any, target: str) -> dict[int, str]:
    """Return display names for the currently active encoded class labels."""
    labels = list(CLASS_LABELS)
    if isinstance(value, dict):
        names: dict[int, str] = {}
        for index in labels:
            candidate = value.get(index, value.get(str(index)))
            if candidate is not None and str(candidate).strip():
                names[index] = str(candidate).strip()
        if len(names) == len(labels):
            return names

    if isinstance(value, (list, tuple)) and len(value) == len(labels):
        names = {index: str(name).strip() for index, name in zip(labels, value)}
        if all(names.values()):
            return names

    if labels == [0, 1, 2] and str(target).strip().upper() == "AC":
        return {0: "Inactive", 1: "Normal", 2: "Active"}
    return {index: f"Class {index}" for index in labels}


def _classification_target_mode(params: dict[str, Any], target: str, values: Any) -> str:
    """Resolve threshold-derived versus already-categorical classification."""
    cfg = params.get("classification", {}) or {}
    modes = cfg.get("target_modes", {}) or {}
    raw = modes.get(target, modes.get(str(target), "auto")) if isinstance(modes, dict) else "auto"
    mode = str(raw or "auto").strip().lower().replace("-", "_")
    if mode in {"categorical", "category", "labels", "label", "direct", "existing_classes", "existing"}:
        return "categorical"
    if mode in {"threshold", "thresholds", "continuous", "continuous_threshold", "three_class"}:
        return "threshold"
    boundaries = cfg.get("class_boundaries", {}) or {}
    candidate = boundaries.get(target, {}) if isinstance(boundaries, dict) else {}
    if isinstance(candidate, dict) and candidate.get("lower") is not None and candidate.get("upper") is not None:
        return "threshold"
    series = pd.Series(_ensure_1d(values)).dropna()
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        return "categorical"
    nunique, n = int(numeric.nunique()), int(len(numeric))
    if nunique <= max(20, int(np.ceil(np.sqrt(max(n, 1))))) and nunique / max(n, 1) <= 0.20:
        return "categorical"
    return "threshold"


def _encode_existing_classes(values: Any, class_values: list[str] | None = None) -> tuple[np.ndarray, dict[int, str], list[str]]:
    series = pd.Series(_ensure_1d(values))
    if series.isna().any():
        raise ValueError("Classification labels contain missing values after cleaning.")
    text = series.astype(str)
    values_sorted = list(class_values) if class_values is not None else sorted(text.unique().tolist())
    if len(values_sorted) < 2:
        raise ValueError("Classification requires at least two distinct target classes.")
    mapping = {value: i for i, value in enumerate(values_sorted)}
    unknown = sorted(set(text.unique()) - set(mapping))
    if unknown:
        raise ValueError(f"Encountered class labels not seen during training: {unknown}")
    encoded = text.map(mapping).to_numpy(dtype=np.int64)
    names = {i: str(value) for i, value in enumerate(values_sorted)}
    return encoded, names, values_sorted


def _encode_target_values(values: Any, info: dict[str, Any]) -> np.ndarray:
    if info.get("mode") == "categorical":
        encoded, _, _ = _encode_existing_classes(values, list(info.get("class_values", [])))
        return encoded
    return _continuous_to_class(values, info)


def _resolve_class_boundaries(
    params: dict[str, Any],
    target: str,
    y_train_raw: Any,
) -> dict[str, Any]:
    """Resolve three-class boundaries from params.yaml.

    Valid predefined limits are used directly. Missing, malformed, reversed,
    non-finite, or CV-unusable limits are never replaced silently. The user is
    first asked whether automatic inference may be used. Declining stops
    classification cleanly.
    """
    class_config = params.get("classification", {}) or {}
    all_boundaries = class_config.get("class_boundaries", {}) or {}

    target_config: dict[str, Any] = {}
    if isinstance(all_boundaries, dict):
        candidate = all_boundaries.get(target)
        if candidate is None:
            candidate = all_boundaries.get(str(target))
        if candidate is None:
            candidate = all_boundaries.get("default")
        if isinstance(candidate, dict):
            target_config = candidate

    lower = target_config.get("lower")
    upper = target_config.get("upper")
    class_names = _normalize_class_names(
        target_config.get("class_names"), target
    )

    values = _ensure_1d(y_train_raw)
    values = values[np.isfinite(values)]

    # Determine whether the YAML limits are legal and usable.
    reason = ""
    if lower is None or upper is None:
        reason = (
            "'lower' and/or 'upper' is not defined in "
            "classification.class_boundaries for this target."
        )
        configured_valid = False
        lower_value = np.nan
        upper_value = np.nan
    else:
        try:
            lower_value = float(lower)
            upper_value = float(upper)
            configured_valid = (
                np.isfinite(lower_value)
                and np.isfinite(upper_value)
                and lower_value < upper_value
            )
            if not configured_valid:
                reason = (
                    f"Illegal limits: lower={lower!r}, upper={upper!r}. "
                    "Both must be finite numbers and lower must be smaller than upper."
                )
        except (TypeError, ValueError):
            configured_valid = False
            lower_value = np.nan
            upper_value = np.nan
            reason = (
                f"Non-numeric limits: lower={lower!r}, upper={upper!r}."
            )

    if configured_valid:
        if not values.size:
            reason = (
                "No finite training target values are available, so the "
                "configured limits cannot be validated."
            )
            configured_valid = False
        else:
            configured_labels = _continuous_to_class(
                values, {"lower": lower_value, "upper": upper_value}
            )
            configured_counts = np.bincount(configured_labels, minlength=3)
            if np.all(configured_counts >= 2):
                return {
                    "lower": lower_value,
                    "upper": upper_value,
                    "class_names": class_names,
                    "source": "params.yaml",
                    "method": "predefined",
                    "training_class_counts": configured_counts.tolist(),
                }
            reason = (
                f"The configured limits lower={lower_value:.6g}, "
                f"upper={upper_value:.6g} produce training class counts "
                f"{configured_counts.tolist()}. At least two observations "
                "are required in each class for stratified CV."
            )

    print(f"[CLASS][WARN] {target}: {reason}")

    # If the values themselves are legal but merely CV-questionable, let the
    # user explicitly keep them. Missing/malformed/reversed limits cannot be
    # forced because they do not define valid numeric class intervals.
    allow_user_limits = bool(
        np.isfinite(lower_value)
        and np.isfinite(upper_value)
        and lower_value < upper_value
    )
    action = _choose_boundary_action(
        target,
        reason,
        allow_user_limits=allow_user_limits,
        lower_value=(float(lower_value) if allow_user_limits else None),
        upper_value=(float(upper_value) if allow_user_limits else None),
    )

    if action == "exit":
        raise ClassificationBoundaryCancelled(
            f"Class-boundary selection cancelled for target '{target}'."
        )

    if action == "user":
        forced_labels = _continuous_to_class(
            values,
            {"lower": lower_value, "upper": upper_value},
        ) if values.size else np.asarray([], dtype=int)
        forced_counts = (
            np.bincount(forced_labels, minlength=3).tolist()
            if forced_labels.size
            else [0, 0, 0]
        )
        print(
            f"[CLASS][WARN] User chose to keep configured limits for {target}: "
            f"lower={lower_value:.6g}, upper={upper_value:.6g}, "
            f"training counts={forced_counts}."
        )
        return {
            "lower": float(lower_value),
            "upper": float(upper_value),
            "class_names": class_names,
            "source": "params.yaml_user_forced",
            "method": "predefined_user_forced",
            "training_class_counts": forced_counts,
        }

    # From here onward, automatic inference is explicitly user-approved.
    if values.size < 3:
        raise ValueError(
            f"Cannot infer class boundaries for target '{target}': "
            "fewer than three finite training target values are available."
        )

    unique_values = np.unique(values)
    if unique_values.size < 3:
        raise ValueError(
            f"Cannot infer three class boundaries for target '{target}': "
            f"only {unique_values.size} unique training target value(s) are available."
        )

    automatic_method = str(
        class_config.get("automatic_boundary_method", "quantile")
    ).strip().lower()
    if automatic_method not in {"quantile", "quantiles", "tertile", "tertiles"}:
        print(
            f"[CLASS][WARN] Unsupported automatic boundary method "
            f"'{automatic_method}'; using training quantiles."
        )

    raw_quantiles = class_config.get(
        "automatic_boundary_quantiles", [1.0 / 3.0, 2.0 / 3.0]
    )
    try:
        q_lower, q_upper = [float(v) for v in raw_quantiles]
    except Exception:
        q_lower, q_upper = 1.0 / 3.0, 2.0 / 3.0

    if not (0.0 < q_lower < q_upper < 1.0):
        print(
            "[CLASS][WARN] Invalid classification.automatic_boundary_quantiles; "
            "using [0.333333, 0.666667]."
        )
        q_lower, q_upper = 1.0 / 3.0, 2.0 / 3.0

    lower_value, upper_value = np.quantile(values, [q_lower, q_upper])
    lower_value = float(lower_value)
    upper_value = float(upper_value)

    provisional = {"lower": lower_value, "upper": upper_value}
    provisional_labels = _continuous_to_class(values, provisional)
    provisional_counts = np.bincount(provisional_labels, minlength=3)

    if not lower_value < upper_value or np.any(provisional_counts < 2):
        best = None
        total = float(values.size)

        for first_index in range(1, len(unique_values) - 1):
            lower_candidate = float(
                (unique_values[first_index - 1] + unique_values[first_index]) / 2.0
            )
            for second_index in range(first_index + 1, len(unique_values)):
                upper_candidate = float(
                    (unique_values[second_index - 1] + unique_values[second_index]) / 2.0
                )
                candidate_labels = _continuous_to_class(
                    values,
                    {"lower": lower_candidate, "upper": upper_candidate},
                )
                candidate_counts = np.bincount(candidate_labels, minlength=3)
                if np.any(candidate_counts < 2):
                    continue
                p0 = candidate_counts[0] / total
                p01 = (candidate_counts[0] + candidate_counts[1]) / total
                score = abs(p0 - q_lower) + abs(p01 - q_upper)
                candidate = (
                    score, lower_candidate, upper_candidate, candidate_counts
                )
                if best is None or candidate[0] < best[0]:
                    best = candidate

        if best is None:
            raise ValueError(
                f"Could not infer three CV-usable classes for target '{target}'. "
                "At least two training observations are required in each class."
            )

        _, lower_value, upper_value, automatic_counts = best
        method = "unique_value_cv_safe_fallback"
    else:
        automatic_counts = provisional_counts
        method = "training_quantiles"

    print(
        f"[CLASS][INFO] User approved automatic boundaries for {target}: "
        f"lower={lower_value:.6g}, upper={upper_value:.6g}, "
        f"method={method}, counts={automatic_counts.tolist()}"
    )

    return {
        "lower": lower_value,
        "upper": upper_value,
        "class_names": class_names,
        "source": "automatic_user_approved",
        "method": method,
        "quantiles": [q_lower, q_upper],
        "training_class_counts": automatic_counts.tolist(),
    }

def _continuous_to_class(values: Any, boundary_info: dict[str, Any]) -> np.ndarray:
    """Convert a continuous target to classes using resolved boundary limits."""
    values = _ensure_1d(values)
    lower = float(boundary_info["lower"])
    upper = float(boundary_info["upper"])
    classes = np.ones(values.shape, dtype=np.int64)
    classes[values < lower] = 0
    classes[values > upper] = 2
    return classes


def _boundary_description(target: str, boundary_info: dict[str, Any]) -> dict[str, str]:
    lower = float(boundary_info["lower"])
    upper = float(boundary_info["upper"])
    names = _normalize_class_names(boundary_info.get("class_names"), target)
    return {
        names[0]: f"{target} < {lower:g}",
        names[1]: f"{lower:g} <= {target} <= {upper:g}",
        names[2]: f"{target} > {upper:g}",
    }


def _score_macro_f1(estimator: Any, X: np.ndarray, y: np.ndarray) -> float:
    """Prediction-based scorer that avoids estimator-tag compatibility issues."""
    return float(f1_score(y, estimator.predict(X), average="macro", zero_division=0))


def _score_balanced_accuracy(estimator: Any, X: np.ndarray, y: np.ndarray) -> float:
    return float(balanced_accuracy_score(y, estimator.predict(X)))


def _score_accuracy(estimator: Any, X: np.ndarray, y: np.ndarray) -> float:
    return float(accuracy_score(y, estimator.predict(X)))


CLASSIFICATION_SCORERS = {
    "macro_f1": _score_macro_f1,
    "balanced_accuracy": _score_balanced_accuracy,
    "accuracy": _score_accuracy,
}


def _load_params(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parameter file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _ensure_2d(values: Any) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    return array


def _ensure_1d(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=float).ravel()



def _load_aligned_train_metadata(
    data_dir: Path,
    target: str,
    valid_mask: np.ndarray,
) -> pd.DataFrame | None:
    """Load optional row-aligned metadata for grouped fold diagnostics."""
    candidates = [
        data_dir / f"metadata_train_{target}.csv",
        data_dir / "metadata_train.csv",
        data_dir / "train_metadata.csv",
        data_dir / f"metadata_train_{target}.joblib",
        data_dir / "metadata_train.joblib",
        data_dir / "train_metadata.joblib",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            if candidate.suffix.lower() == ".csv":
                frame = pd.read_csv(candidate)
            else:
                obj = joblib.load(candidate)
                frame = obj.copy() if isinstance(obj, pd.DataFrame) else pd.DataFrame(obj)
        except Exception as error:
            print(f"[CLASS][WARN] Could not read optional metadata {candidate.name}: {error}")
            continue

        if len(frame) != len(valid_mask):
            print(
                f"[CLASS][WARN] Ignoring {candidate.name}: metadata rows={len(frame)} "
                f"but training rows={len(valid_mask)}."
            )
            continue

        frame = frame.loc[np.asarray(valid_mask, dtype=bool)].reset_index(drop=True)
        print(f"[CLASS][GROUP DIAGNOSTICS] Using aligned metadata: {candidate.name}")
        return frame
    return None


def _first_existing_column(
    frame: pd.DataFrame | None,
    names: list[str],
) -> str | None:
    if frame is None:
        return None
    lower = {str(column).lower(): str(column) for column in frame.columns}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def _build_outer_group_classification_diagnostics(
    *,
    fold_index: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_values: np.ndarray | None,
    metadata: pd.DataFrame | None,
    validation_idx: np.ndarray,
    class_names: dict[int, str],
) -> list[dict[str, Any]]:
    """Return one diagnostic row per held-out group/location in an outer fold."""
    if group_values is None:
        return []

    y_true = np.asarray(y_true, dtype=int).ravel()
    y_pred = np.asarray(y_pred, dtype=int).ravel()
    validation_groups = np.asarray(group_values)[validation_idx]
    validation_meta = (
        metadata.iloc[validation_idx].reset_index(drop=True)
        if metadata is not None else None
    )
    project_col = _first_existing_column(
        validation_meta,
        ["Project", "project", "Project_Name", "project_name"],
    )
    location_col = _first_existing_column(
        validation_meta,
        ["Location_No", "location_no", "Location", "location"],
    )

    rows: list[dict[str, Any]] = []
    for group in np.unique(validation_groups):
        mask = validation_groups == group
        yt = y_true[mask]
        yp = y_pred[mask]
        actual_counts = np.bincount(yt, minlength=len(CLASS_LABELS))
        predicted_counts = np.bincount(yp, minlength=len(CLASS_LABELS))

        row: dict[str, Any] = {
            "outer_fold": int(fold_index),
            "group": str(group),
            "n_samples": int(mask.sum()),
            "macro_f1": float(
                f1_score(
                    yt, yp,
                    labels=CLASS_LABELS,
                    average="macro",
                    zero_division=0,
                )
            ),
            "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
            "accuracy": float(accuracy_score(yt, yp)),
        }

        for label in CLASS_LABELS:
            safe_label = _safe_name(class_names[label]).lower()
            row[f"actual_{safe_label}_n"] = int(actual_counts[label])
            row[f"predicted_{safe_label}_n"] = int(predicted_counts[label])
            row[f"actual_class_{label}_n"] = int(actual_counts[label])
            row[f"predicted_class_{label}_n"] = int(predicted_counts[label])

        if validation_meta is not None:
            meta_group = validation_meta.loc[mask]
            if project_col is not None:
                projects = sorted(
                    {str(v) for v in meta_group[project_col].dropna().unique()}
                )
                row["project"] = " | ".join(projects)
            if location_col is not None:
                locations = sorted(
                    {str(v) for v in meta_group[location_col].dropna().unique()}
                )
                row["location_no"] = " | ".join(locations)

        rows.append(row)
    return rows


def _class_output_dir(base_dir: str | Path) -> Path:
    """Use the classification output directory exactly as configured."""
    return Path(base_dir)


def ac_to_activity_class(
    values: Any,
    boundary_info: dict[str, Any] | None = None,
) -> np.ndarray:
    """Backward-compatible three-class converter.

    New training code supplies ``boundary_info`` resolved from params.yaml or
    inferred from training data.  The historical AC limits are retained only
    for callers that invoke this helper directly without boundary metadata.
    """
    if boundary_info is None:
        boundary_info = {"lower": 0.75, "upper": 1.25}
    return _continuous_to_class(values, boundary_info)


def _model_factories() -> dict[str, Any]:
    """Compatibility wrapper around the modular classifier registry."""
    return {
        name: (lambda name=name: load_classifier_spec(name).build_estimator(42))
        for name in available_classifiers()
    }


def _parameter_grids() -> dict[str, dict[str, list[Any]]]:
    """Return pipeline-prefixed default search spaces from classifier modules."""
    return {
        name: {f"classifier__{key}": value for key, value in load_classifier_spec(name).default_grid.items()}
        for name in available_classifiers()
    }


def _build_pipeline(model: Any, preprocessing_config: dict[str, Any]) -> Pipeline:
    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    preprocessing = create_preprocessing_pipeline(preprocessing_config)
    if preprocessing is not None:
        steps.append(("preprocessing", preprocessing))
    elif isinstance(model, (LogisticRegression, SVC, KNeighborsClassifier, MLPClassifier)):
        # Scale distance/gradient-based models when no shared preprocessing is configured.
        steps.append(("scaler", StandardScaler()))
    steps.append(("classifier", model))
    return Pipeline(steps)


def _evaluate(
    model: Any,
    X: np.ndarray,
    y_true: np.ndarray,
    class_names: dict[int, str] | None = None,
) -> dict[str, Any]:
    class_names = _normalize_class_names(class_names, "AC")
    y_pred = model.predict(X)
    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=CLASS_LABELS).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=CLASS_LABELS,
            target_names=[class_names[i] for i in CLASS_LABELS],
            output_dict=True,
            zero_division=0,
        ),
        "actual": y_true.tolist(),
        "predictions": y_pred.tolist(),
    }
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(X)
            result["log_loss"] = float(log_loss(y_true, probabilities, labels=CLASS_LABELS))
            result["probabilities"] = probabilities.tolist()
        except Exception:
            result["log_loss"] = None
    return result


def _classification_diagnostics(
    train_cv_macro_f1: float,
    cv_macro_f1: float,
    cv_macro_f1_std: float,
    val_macro_f1: float = np.nan,
    test_macro_f1: float = np.nan,
) -> dict[str, Any]:
    """Describe grouped generalization without equating a CV gap with overfitting.

    With location-grouped validation, a large train--CV gap can reflect
    conventional model overfitting, genuine between-location heterogeneity,
    or both. The primary diagnosis therefore uses neutral wording.
    """
    train_cv_gap = train_cv_macro_f1 - cv_macro_f1
    train_val_gap = (
        train_cv_macro_f1 - val_macro_f1
        if np.isfinite(val_macro_f1) else np.nan
    )
    cv_val_gap = (
        cv_macro_f1 - val_macro_f1
        if np.isfinite(val_macro_f1) else np.nan
    )
    val_test_gap = (
        val_macro_f1 - test_macro_f1
        if np.isfinite(val_macro_f1) and np.isfinite(test_macro_f1)
        else np.nan
    )

    large_gap = bool(np.isfinite(train_cv_gap) and train_cv_gap >= 0.10)
    low_performance = bool(
        np.isfinite(train_cv_macro_f1)
        and np.isfinite(cv_macro_f1)
        and train_cv_macro_f1 < 0.55
        and cv_macro_f1 < 0.50
    )
    high_between_group_variability = bool(
        np.isfinite(cv_macro_f1_std) and cv_macro_f1_std >= 0.15
    )
    stable = bool(
        np.isfinite(train_cv_gap)
        and abs(train_cv_gap) <= 0.05
        and np.isfinite(cv_macro_f1)
        and cv_macro_f1 >= 0.50
        and np.isfinite(cv_macro_f1_std)
        and cv_macro_f1_std < 0.15
    )

    if low_performance:
        diagnosis = "Low train and grouped-CV performance"
    elif large_gap and high_between_group_variability:
        diagnosis = (
            "Large train-CV generalization gap with high between-group variability"
        )
    elif large_gap:
        diagnosis = "Large train-CV generalization gap"
    elif high_between_group_variability:
        diagnosis = "High between-group CV variability"
    elif stable:
        diagnosis = "Stable grouped generalization"
    else:
        diagnosis = "Moderate grouped generalization"

    return {
        "train_cv_gap": train_cv_gap,
        "train_val_gap": train_val_gap,
        "cv_val_gap": cv_val_gap,
        "val_test_gap": val_test_gap,
        "abs_train_cv_gap": abs(train_cv_gap) if np.isfinite(train_cv_gap) else np.nan,
        "abs_cv_val_gap": abs(cv_val_gap) if np.isfinite(cv_val_gap) else np.nan,
        # Compatibility flags retained for downstream readers. These are
        # screening indicators only, not proof of causal overfitting.
        "possible_overfitting": large_gap,
        "possible_underfitting": low_performance,
        "large_generalization_gap": large_gap,
        "high_cv_instability": high_between_group_variability,
        "high_between_group_variability": high_between_group_variability,
        "stable_generalization": stable,
        "split_sensitive": bool(
            np.isfinite(cv_val_gap) and abs(cv_val_gap) >= 0.15
        ),
        "generalization_diagnosis": diagnosis,
    }

def _flatten_results(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in results:
        val = item.get("validation") or {}
        test = item.get("test") or {}
        train_cv = float(item.get("cv_train_macro_f1", np.nan))
        cv_mean = float(item.get("cv_macro_f1", np.nan))
        cv_std = float(item.get("cv_macro_f1_std", np.nan))
        val_f1 = float(val.get("macro_f1", np.nan))
        test_f1 = float(test.get("macro_f1", np.nan))
        diagnostic = _classification_diagnostics(train_cv, cv_mean, cv_std, val_f1, test_f1)

        row = {
            "model": item["model"],
            "feature_set": item["feature_set"],
            "target": item["target"],
            "file": item["file"],
            "cv_train_macro_f1": train_cv,
            "cv_train_macro_f1_std": item.get("cv_train_macro_f1_std", np.nan),
            "cv_macro_f1": cv_mean,
            "cv_macro_f1_std": cv_std,
            "val_accuracy": val.get("accuracy", np.nan),
            "val_balanced_accuracy": val.get("balanced_accuracy", np.nan),
            "val_macro_precision": val.get("macro_precision", np.nan),
            "val_macro_recall": val.get("macro_recall", np.nan),
            "val_macro_f1": val_f1,
            "val_weighted_f1": val.get("weighted_f1", np.nan),
            "test_accuracy": test.get("accuracy", np.nan),
            "test_balanced_accuracy": test.get("balanced_accuracy", np.nan),
            "test_macro_precision": test.get("macro_precision", np.nan),
            "test_macro_recall": test.get("macro_recall", np.nan),
            "test_macro_f1": test_f1,
            "test_weighted_f1": test.get("weighted_f1", np.nan),
        }
        row.update(diagnostic)
        # Backward-compatible alias retained for older report readers.
        row["cv_generalization_gap"] = diagnostic["train_cv_gap"]
        rows.append(row)
    return pd.DataFrame(rows)




def _generate_results_classification_summary(summary_df: pd.DataFrame, reports_dir: Path) -> None:
    """Generate a manuscript-ready model-performance table across feature sets.

    The requested misspelled filename is retained for compatibility, while a
    correctly spelled alias and raw numeric companion are generated as well.
    Models are displayed in model/feature-set order; ``Overall Rank`` is based
    on holdout-validation Macro-F1, followed by CV Macro-F1, lower CV SD, and
    a smaller absolute train-CV gap. Test scores are reported but never used
    for ranking.
    """
    if summary_df.empty:
        return

    required = {
        "model", "feature_set", "target", "cv_train_macro_f1",
        "cv_train_macro_f1_std", "cv_macro_f1", "cv_macro_f1_std",
        "val_macro_f1", "test_macro_f1", "generalization_diagnosis",
    }
    missing = sorted(required.difference(summary_df.columns))
    if missing:
        print(f"[CLASS][WARN] Cannot generate manuscript summary; missing columns: {missing}")
        return

    work = summary_df.copy()
    work["abs_train_cv_gap"] = work.get(
        "abs_train_cv_gap",
        (work["cv_train_macro_f1"] - work["cv_macro_f1"]).abs(),
    )

    ranked = work.sort_values(
        ["val_macro_f1", "cv_macro_f1", "cv_macro_f1_std", "abs_train_cv_gap"],
        ascending=[False, False, True, True],
        na_position="last",
    ).copy()
    ranked["Overall Rank"] = np.arange(1, len(ranked) + 1)
    rank_map = {
        (
            str(row["model"]),
            str(row["feature_set"]),
            str(row["target"]),
        ): int(row["Overall Rank"])
        for _, row in ranked[
            ["model", "feature_set", "target", "Overall Rank"]
        ].iterrows()
    }

    def fmt_mean_sd(mean: Any, sd: Any) -> str:
        if not np.isfinite(float(mean)):
            return "NA"
        if not np.isfinite(float(sd)):
            return f"{float(mean):.3f}"
        return f"{float(mean):.3f} ± {float(sd):.3f}"

    def fmt_value(value: Any) -> str:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return "NA"
        return f"{value:.3f}" if np.isfinite(value) else "NA"

    # Natural model order and feature-set order make the table easy to read in a paper.
    display = work.sort_values(["model", "feature_set", "target"], kind="stable").copy()
    rows: list[dict[str, Any]] = []
    for _, row in display.iterrows():
        key = (str(row["model"]), str(row["feature_set"]), str(row["target"]))
        rows.append({
            "Overall Rank": rank_map[key],
            "Target": row["target"],
            "Model": row["model"],
            "Feature Set": row["feature_set"],
            "CV Train Macro F1": fmt_mean_sd(
                row["cv_train_macro_f1"], row["cv_train_macro_f1_std"]
            ),
            "CV Macro F1": fmt_mean_sd(row["cv_macro_f1"], row["cv_macro_f1_std"]),
            "Validation Macro F1": fmt_value(row["val_macro_f1"]),
            "Test Macro F1": fmt_value(row["test_macro_f1"]),
            "Train-CV Gap": fmt_value(row.get("train_cv_gap", np.nan)),
            "CV-Validation Gap": fmt_value(row.get("cv_val_gap", np.nan)),
            "CV SD": fmt_value(row["cv_macro_f1_std"]),
            "Validation Balanced Accuracy": fmt_value(row.get("val_balanced_accuracy", np.nan)),
            "Test Balanced Accuracy": fmt_value(row.get("test_balanced_accuracy", np.nan)),
            "Generalization Diagnosis": row["generalization_diagnosis"],
        })

    paper = pd.DataFrame(rows)
    requested_path = reports_dir / "Results_classification_summery.csv"
    corrected_path = reports_dir / "Results_classification_summary.csv"
    paper.to_csv(requested_path, index=False, encoding="utf-8-sig")
    paper.to_csv(corrected_path, index=False, encoding="utf-8-sig")

    # Raw numeric companion supports re-analysis without parsing formatted cells.
    numeric_columns = [
        "target", "model", "feature_set", "cv_train_macro_f1",
        "cv_train_macro_f1_std", "cv_macro_f1", "cv_macro_f1_std",
        "val_macro_f1", "test_macro_f1", "train_cv_gap", "cv_val_gap",
        "val_test_gap", "val_balanced_accuracy", "test_balanced_accuracy",
        "generalization_diagnosis",
    ]
    numeric = display[[c for c in numeric_columns if c in display.columns]].copy()
    numeric.insert(0, "overall_rank", [
        rank_map[(str(r.model), str(r.feature_set), str(r.target))]
        for r in display[["model", "feature_set", "target"]].itertuples(index=False)
    ])
    numeric.to_csv(reports_dir / "Results_classification_summery_numeric.csv", index=False)

    # Overleaf-ready version. Formatted mean ± SD cells remain intact.
    try:
        latex = paper.to_latex(
            index=False,
            escape=True,
            caption=(
                "Model performance comparison across feature sets for the AC "
                "activity-classification target."
            ),
            label="tab:classification_model_comparison",
            longtable=True,
        )
        (reports_dir / "Results_classification_summery.tex").write_text(latex, encoding="utf-8")
    except Exception as error:
        (reports_dir / "Results_classification_summery.tex").write_text(
            f"% LaTeX export failed: {error}\n", encoding="utf-8"
        )

    # Brief results-section aid identifying the strongest and most stable candidates.
    best = ranked.iloc[0]
    stable = ranked[ranked["stable_generalization"] == True] if "stable_generalization" in ranked else pd.DataFrame()
    lines = [
        "TABLE 1: MODEL PERFORMANCE COMPARISON ACROSS FEATURE SETS FOR AC TARGET",
        "=" * 78,
        "",
        f"Best validation model: {best['model']} with {best['feature_set']} "
        f"(validation Macro-F1={best['val_macro_f1']:.3f}, "
        f"test Macro-F1={best['test_macro_f1']:.3f}, "
        f"CV Macro-F1={best['cv_macro_f1']:.3f} ± {best['cv_macro_f1_std']:.3f}).",
        f"Generalization diagnosis: {best['generalization_diagnosis']}.",
        "",
        "Reporting note: models were ranked using validation and cross-validation "
        "performance and stability; test Macro-F1 was excluded from ranking and "
        "is reported only as an independent final assessment.",
    ]
    if not stable.empty:
        stable_names = ", ".join(
            f"{r.model}-{r.feature_set}" for r in stable[["model", "feature_set"]].itertuples(index=False)
        )
        lines.extend(["", f"Stable-generalization combinations: {stable_names}."])
    (reports_dir / "Results_classification_summery.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print(f"[CLASS] Manuscript table saved: {requested_path}")

def _safe_name(value: Any) -> str:
    """Return a file-system-safe label."""
    text = str(value).strip().replace("classifier__", "")
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _generate_hyperparameter_analysis(
    search: Any,
    reports_dir: Path,
    target: str,
    feature_set: str,
    model_name: str,
) -> dict[str, Any]:
    """Create post-search CV validation curves and generalization-gap plots.

    Hyperparameters are selected objectively by GridSearchCV/RandomizedSearchCV
    using mean stratified-CV Macro-F1. These plots are generated only after the
    search and never override ``best_params_``. For multi-parameter searches,
    values for one parameter are averaged across combinations of the remaining
    parameters; therefore, the plots show marginal relationships rather than
    isolated one-dimensional experiments.
    """
    if not hasattr(search, "cv_results_"):
        return {}

    cv_results = search.cv_results_
    if "mean_train_score" not in cv_results or "mean_test_score" not in cv_results:
        return {}

    output_dir = (
        reports_dir / "hyperparameter_analysis" / _safe_name(target)
        / _safe_name(feature_set) / _safe_name(model_name)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(cv_results)
    parameter_columns = [c for c in frame.columns if c.startswith("param_")]
    generated: dict[str, Any] = {"figures": [], "tables": [], "parameters": {}}

    for parameter_column in parameter_columns:
        parameter = parameter_column.removeprefix("param_").removeprefix("classifier__")
        work = frame[[parameter_column, "mean_train_score", "std_train_score",
                      "mean_test_score", "std_test_score"]].copy()
        work["parameter_value"] = work[parameter_column].map(str)
        grouped = (
            work.groupby("parameter_value", dropna=False, sort=False)
            .agg(
                train_macro_f1=("mean_train_score", "mean"),
                train_macro_f1_sd=("std_train_score", "mean"),
                cv_macro_f1=("mean_test_score", "mean"),
                cv_macro_f1_sd=("std_test_score", "mean"),
            )
            .reset_index()
        )
        grouped["generalization_gap"] = grouped["train_macro_f1"] - grouped["cv_macro_f1"]
        grouped["parameter"] = parameter

        table_path = output_dir / f"{_safe_name(parameter)}_validation_curve.csv"
        grouped.to_csv(table_path, index=False)

        x = np.arange(len(grouped))
        labels = grouped["parameter_value"].tolist()

        fig, ax = plt.subplots(figsize=(8.2, 5.2))
        ax.plot(x, grouped["train_macro_f1"], marker="o", label="Training CV Macro-F1")
        ax.plot(x, grouped["cv_macro_f1"], marker="o", label="Validation CV Macro-F1")
        ax.fill_between(
            x,
            grouped["cv_macro_f1"] - grouped["cv_macro_f1_sd"],
            grouped["cv_macro_f1"] + grouped["cv_macro_f1_sd"],
            alpha=0.15,
        )
        ax.set_xticks(x, labels, rotation=35, ha="right")
        ax.set_xlabel(parameter)
        ax.set_ylabel("Macro-F1")
        ax.set_title(f"{model_name} ({feature_set}): {parameter} CV validation curve")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        curve_path = output_dir / f"{_safe_name(parameter)}_validation_curve.png"
        fig.savefig(curve_path, dpi=600, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        ax.bar(x, grouped["generalization_gap"])
        ax.axhline(0.10, linestyle="--", linewidth=1.2, label="Review threshold (0.10)")
        ax.axhline(0.00, linewidth=0.9)
        ax.set_xticks(x, labels, rotation=35, ha="right")
        ax.set_xlabel(parameter)
        ax.set_ylabel("Train CV Macro-F1 − validation CV Macro-F1")
        ax.set_title(f"{model_name} ({feature_set}): {parameter} generalization gap")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        gap_path = output_dir / f"{_safe_name(parameter)}_overfitting_gap.png"
        fig.savefig(gap_path, dpi=600, bbox_inches="tight")
        plt.close(fig)

        best_idx = grouped["cv_macro_f1"].idxmax()
        best_row = grouped.loc[best_idx]
        generated["figures"].extend([str(curve_path), str(gap_path)])
        generated["tables"].append(str(table_path))
        generated["parameters"][parameter] = {
            "best_marginal_value": str(best_row["parameter_value"]),
            "train_macro_f1": float(best_row["train_macro_f1"]),
            "cv_macro_f1": float(best_row["cv_macro_f1"]),
            "cv_macro_f1_sd": float(best_row["cv_macro_f1_sd"]),
            "generalization_gap": float(best_row["generalization_gap"]),
        }

    summary_path = output_dir / "hyperparameter_gap_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(generated["parameters"], file, indent=2, default=_json_default)
    generated["tables"].append(str(summary_path))
    return generated




def _generate_permutation_sensitivity_classification(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    reports_dir: Path,
    target: str,
    feature_set: str,
    model_name: str,
    n_repeats: int = 30,
    random_state: int = 42,
    n_jobs: int = -1,
) -> dict[str, Any]:
    """Compute raw-feature permutation sensitivity with Macro-F1 decrease."""
    output_dir = (
        reports_dir / "permutation_sensitivity" / _safe_name(target)
        / _safe_name(feature_set)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result = permutation_importance(
        estimator, X, y, scoring=_score_macro_f1,
        n_repeats=max(1, int(n_repeats)), random_state=random_state,
        n_jobs=n_jobs,
    )
    names = list(feature_names)
    if len(names) != X.shape[1]:
        names = [f"feature_{i + 1}" for i in range(X.shape[1])]
    means = np.asarray(result.importances_mean, dtype=float)
    stds = np.asarray(result.importances_std, dtype=float)
    absolute = np.abs(means)
    denominator = float(np.sum(absolute))
    relative = (100.0 * absolute / denominator) if denominator > 0 else np.zeros_like(absolute)
    frame = pd.DataFrame({
        "feature": names,
        "mean_importance": means,
        "std_importance": stds,
        "absolute_importance": absolute,
        "relative_contribution_percent": relative,
        "scoring": "macro_f1",
        "n_repeats": int(n_repeats),
    }).sort_values("absolute_importance", ascending=False).reset_index(drop=True)
    path = output_dir / f"{_safe_name(model_name)}_permutation_sensitivity.csv"
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return {
        "performed": True,
        "scoring": "macro_f1",
        "n_repeats": int(n_repeats),
        "file": str(path),
        "features": frame.to_dict(orient="records"),
    }


def _grouped_classification_learning_curve_scores(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv: Any,
    groups: np.ndarray,
    train_sizes: Any,
    scoring_callable: Any,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a learning curve by adding complete sampling groups.

    Standard sklearn ``learning_curve`` applies fractional train sizes to row
    counts after CV splitting. For grouped observations that can produce
    confusing effective sample sizes. This routine selects complete groups at
    each fraction and never divides one borehole/location group across the
    learning subset and the held-out validation fold.
    """
    X = _ensure_2d(X)
    y = np.asarray(y, dtype=int).ravel()
    groups = np.asarray(groups).ravel()
    fractions = np.asarray(train_sizes, dtype=float)

    if not (len(X) == len(y) == len(groups)):
        raise ValueError(
            "X, y, and groups must have identical row counts for a grouped "
            "classification learning curve."
        )
    if np.any(fractions <= 0) or np.any(fractions > 1):
        raise ValueError(
            "Grouped learning-curve train sizes must be fractions in (0, 1]."
        )

    splits = list(cv.split(X, y, groups))
    n_sizes = len(fractions)
    n_folds = len(splits)

    train_scores = np.full((n_sizes, n_folds), np.nan, dtype=float)
    cv_scores = np.full((n_sizes, n_folds), np.nan, dtype=float)
    effective_sizes = np.zeros((n_sizes, n_folds), dtype=int)

    for fold_index, (train_idx, validation_idx) in enumerate(splits):
        fold_groups = groups[train_idx]
        unique_groups = np.unique(fold_groups)

        rng = np.random.default_rng(random_state + fold_index)
        ordered_groups = unique_groups.copy()
        rng.shuffle(ordered_groups)

        for size_index, fraction in enumerate(fractions):
            n_groups = max(
                1,
                min(
                    len(ordered_groups),
                    int(np.ceil(float(fraction) * len(ordered_groups))),
                ),
            )
            selected_groups = set(ordered_groups[:n_groups].tolist())
            selected_mask = np.array(
                [g in selected_groups for g in fold_groups],
                dtype=bool,
            )
            subset_idx = train_idx[selected_mask]

            if len(subset_idx) < 2:
                continue

            # Small grouped subsets can omit one or more classes. Some
            # classifiers cannot fit such a subset, so skip it rather than
            # turning a diagnostic curve into a training failure.
            if len(np.unique(y[subset_idx])) < 2:
                continue

            fitted = clone(estimator)
            try:
                fitted.fit(X[subset_idx], y[subset_idx])
                train_scores[size_index, fold_index] = scoring_callable(
                    fitted, X[subset_idx], y[subset_idx]
                )
                cv_scores[size_index, fold_index] = scoring_callable(
                    fitted, X[validation_idx], y[validation_idx]
                )
                effective_sizes[size_index, fold_index] = len(subset_idx)
            except Exception:
                # Learning curves are descriptive diagnostics. A failure at an
                # unusually small grouped subset is recorded as NaN.
                continue

    effective = np.where(effective_sizes > 0, effective_sizes, np.nan)
    mean_sizes = np.rint(np.nanmean(effective, axis=1))
    min_sizes = np.nanmin(effective, axis=1)

    mean_sizes = np.where(np.isfinite(mean_sizes), mean_sizes, 0).astype(int)
    min_sizes = np.where(np.isfinite(min_sizes), min_sizes, 0).astype(int)

    return mean_sizes, min_sizes, train_scores, cv_scores


def _generate_learning_curve_report(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv: Any,
    reports_dir: Path,
    target: str,
    feature_set: str,
    model_name: str,
    scoring: str = "f1_macro",
    train_sizes: Any = None,
    groups: np.ndarray | None = None,
) -> dict[str, Any]:
    """Generate a training-only, leakage-safe diagnostic learning curve."""
    output = {
        "figure": None,
        "table": None,
        "role": "post-selection diagnostic only",
    }
    if train_sizes is None:
        train_sizes = [0.2, 0.4, 0.6, 0.8, 1.0]

    try:
        fractions = np.asarray(train_sizes, dtype=float)
        scoring_key = str(scoring).strip().lower()
        scoring_callable = {
            "f1_macro": _score_macro_f1,
            "macro_f1": _score_macro_f1,
            "balanced_accuracy": _score_balanced_accuracy,
            "accuracy": _score_accuracy,
        }.get(scoring_key, _score_macro_f1)

        if groups is not None:
            (
                sizes_mean,
                sizes_min,
                train_scores,
                validation_scores,
            ) = _grouped_classification_learning_curve_scores(
                estimator=estimator,
                X=X,
                y=y,
                cv=cv,
                groups=np.asarray(groups).ravel(),
                train_sizes=fractions,
                scoring_callable=scoring_callable,
                random_state=42,
            )
            learning_curve_mode = "whole_group_subsampling"
        else:
            sizes, train_scores, validation_scores = learning_curve(
                estimator,
                X,
                y,
                groups=None,
                cv=cv,
                scoring=scoring_callable,
                train_sizes=fractions,
                n_jobs=-1,
                shuffle=True,
                random_state=42,
                error_score=np.nan,
            )
            sizes_mean = np.asarray(sizes, dtype=int)
            sizes_min = np.asarray(sizes, dtype=int)
            learning_curve_mode = "row_subsampling"

        frame = pd.DataFrame({
            "train_fraction": fractions,
            "train_size": sizes_mean,
            "train_size_min_across_folds": sizes_min,
            "train_score_mean": np.nanmean(train_scores, axis=1),
            "train_score_sd": np.nanstd(train_scores, axis=1, ddof=1),
            "cv_score_mean": np.nanmean(validation_scores, axis=1),
            "cv_score_sd": np.nanstd(validation_scores, axis=1, ddof=1),
            "learning_curve_mode": learning_curve_mode,
            "group_aware": groups is not None,
        })

        folder = (
            reports_dir
            / "learning_curves"
            / _safe_name(target)
            / _safe_name(feature_set)
        )
        folder.mkdir(parents=True, exist_ok=True)

        table = folder / f"{_safe_name(model_name)}_learning_curve.csv"
        figure = folder / f"{_safe_name(model_name)}_learning_curve.png"
        frame.to_csv(table, index=False, encoding="utf-8-sig")

        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        ax.errorbar(
            frame["train_size"],
            frame["train_score_mean"],
            yerr=frame["train_score_sd"],
            marker="o",
            capsize=3,
            label="Training Macro-F1",
        )
        ax.errorbar(
            frame["train_size"],
            frame["cv_score_mean"],
            yerr=frame["cv_score_sd"],
            marker="o",
            capsize=3,
            label="Validation CV Macro-F1",
        )
        ax.set_xlabel("Effective number of training samples")
        ax.set_ylabel(scoring)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(
            f"Diagnostic learning curve: {model_name} | "
            f"{feature_set} | {target}"
        )
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figure, dpi=600, bbox_inches="tight")
        plt.close(fig)

        output.update({
            "figure": str(figure),
            "table": str(table),
            "group_aware": groups is not None,
            "learning_curve_mode": learning_curve_mode,
        })
    except Exception as error:
        output["error"] = str(error)
        print(
            f"[CLASS][WARN] Learning curve skipped for "
            f"{model_name}/{feature_set}/{target}: {error}"
        )

    return output


def _summarize_classification_learning_curves(reports_dir: Path) -> pd.DataFrame:
    """Create per-feature-set and overall learning-curve summaries for classifiers."""
    root = Path(reports_dir) / "learning_curves"
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return pd.DataFrame()
    for target_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for feature_dir in sorted(p for p in target_dir.iterdir() if p.is_dir()):
            feature_rows: list[dict[str, Any]] = []
            for csv_path in sorted(feature_dir.glob("*_learning_curve.csv")):
                try:
                    frame = pd.read_csv(csv_path).sort_values("train_size")
                except Exception as error:
                    print(f"[CLASS][WARN] Could not read learning curve {csv_path}: {error}")
                    continue
                required = {"train_size", "train_score_mean", "train_score_sd", "cv_score_mean", "cv_score_sd"}
                if frame.empty or not required.issubset(frame.columns):
                    continue
                first, last = frame.iloc[0], frame.iloc[-1]
                gaps = frame["train_score_mean"] - frame["cv_score_mean"]
                cv_gain = float(last["cv_score_mean"] - first["cv_score_mean"])
                final_gap = float(last["train_score_mean"] - last["cv_score_mean"])
                if len(frame) >= 2:
                    recent_cv_gain = float(
                        frame.iloc[-1]["cv_score_mean"]
                        - frame.iloc[-2]["cv_score_mean"]
                    )
                else:
                    recent_cv_gain = 0.0

                # A large train-CV gap is an overfitting warning, but does not
                # mean that additional independent groups cannot help.
                more_data_signal = bool(
                    cv_gain >= 0.03
                    and recent_cv_gain >= -0.01
                )

                gap_label = "small" if abs(final_gap) <= 0.05 else ("moderate" if abs(final_gap) <= 0.10 else "large")
                trend = "improving" if cv_gain >= 0.02 else ("declining" if cv_gain <= -0.02 else "near plateau")
                model = csv_path.name.removesuffix("_learning_curve.csv")
                item = {
                    "target": target_dir.name,
                    "feature_set": feature_dir.name,
                    "model": model,
                    "initial_train_size": int(first["train_size"]),
                    "final_train_size": int(last["train_size"]),
                    "initial_train_macro_f1": float(first["train_score_mean"]),
                    "final_train_macro_f1": float(last["train_score_mean"]),
                    "initial_cv_macro_f1": float(first["cv_score_mean"]),
                    "final_cv_macro_f1": float(last["cv_score_mean"]),
                    "cv_macro_f1_gain": cv_gain,
                    "recent_cv_macro_f1_gain": recent_cv_gain,
                    "final_train_cv_gap": final_gap,
                    "max_abs_train_cv_gap": float(np.nanmax(np.abs(gaps))),
                    "initial_train_sd": float(first["train_score_sd"]),
                    "final_train_sd": float(last["train_score_sd"]),
                    "train_sd_change": float(last["train_score_sd"] - first["train_score_sd"]),
                    "initial_cv_sd": float(first["cv_score_sd"]),
                    "final_cv_sd": float(last["cv_score_sd"]),
                    "cv_sd_change": float(last["cv_score_sd"] - first["cv_score_sd"]),
                    "validation_trend": trend,
                    "final_gap_level": gap_label,
                    "more_data_likely_helpful": more_data_signal,
                }
                feature_rows.append(item); rows.append(item)
            if not feature_rows:
                continue
            summary = pd.DataFrame(feature_rows).sort_values(
                ["final_cv_macro_f1", "final_cv_sd", "max_abs_train_cv_gap"],
                ascending=[False, True, True],
            )
            summary.to_csv(feature_dir / "learning_curve_model_summary.csv", index=False, encoding="utf-8-sig")
            best = summary.iloc[0]
            improving = int((summary["validation_trend"] == "improving").sum())
            small_gap = int((summary["final_gap_level"] == "small").sum())
            lower_cv_sd = int((summary["cv_sd_change"] < 0).sum())
            mean_gain = float(summary["cv_macro_f1_gain"].mean())
            median_gap = float(summary["final_train_cv_gap"].abs().median())
            likely = int(summary["more_data_likely_helpful"].sum())
            lines = [
                f"CLASSIFICATION LEARNING-CURVE INTERPRETATION: {target_dir.name} / {feature_dir.name}",
                "=" * 72,
                f"Models summarized: {len(summary)}",
                f"Best final CV Macro-F1: {best['model']} ({best['final_cv_macro_f1']:.3f}; final gap={best['final_train_cv_gap']:.3f}).",
                f"Mean CV Macro-F1 change: {mean_gain:+.3f}.",
                f"Improving models (gain >= 0.02): {improving}/{len(summary)}.",
                f"Small final train-CV gap (|gap| <= 0.05): {small_gap}/{len(summary)}.",
                f"Reduced CV variability at largest size: {lower_cv_sd}/{len(summary)}.",
                f"Median absolute final train-CV gap: {median_gap:.3f}.",
                f"Models showing a continued more-data signal: {likely}/{len(summary)}.",
                "",
            ]
            if improving >= max(1, len(summary) // 2):
                lines.append("Overall interpretation: Macro-F1 generally improved as additional independent observations/groups were added. This is descriptive evidence that more independent sampling groups may improve classification generalization; the train-CV gap should be interpreted separately.")
            elif median_gap > 0.10:
                lines.append("Overall interpretation: validation gains were limited while the final train-CV gap remained comparatively large. Under grouped validation this may reflect model overfitting, between-location heterogeneity, or both; additional independent groups and model regularization should be considered separately.")
            else:
                lines.append("Overall interpretation: the learning curves were broadly stable and close to a plateau, without strong evidence of severe overfitting at the largest training size.")
            lines.extend(["", "These are descriptive diagnostics, not formal hypothesis tests."])
            (feature_dir / "learning_curve_interpretation.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    combined = pd.DataFrame(rows)
    if combined.empty:
        return combined
    combined.to_csv(root / "learning_curve_all_models_summary.csv", index=False, encoding="utf-8-sig")
    fs = (
        combined.groupby(["target", "feature_set"], as_index=False)
        .agg(
            models=("model", "count"),
            mean_cv_macro_f1_gain=("cv_macro_f1_gain", "mean"),
            median_final_cv_macro_f1=("final_cv_macro_f1", "median"),
            median_abs_final_gap=("final_train_cv_gap", lambda s: float(np.median(np.abs(s)))),
            mean_final_cv_sd=("final_cv_sd", "mean"),
            models_likely_helped_by_more_data=("more_data_likely_helpful", "sum"),
        )
        .sort_values(["target", "median_final_cv_macro_f1"], ascending=[True, False])
    )
    fs.to_csv(root / "learning_curve_feature_set_summary.csv", index=False, encoding="utf-8-sig")
    overview = ["CLASSIFICATION LEARNING-CURVE OVERVIEW BY FEATURE SET", "=" * 72, ""]
    for row in fs.itertuples(index=False):
        overview.append(
            f"{row.target}/{row.feature_set}: median final CV Macro-F1={row.median_final_cv_macro_f1:.3f}, "
            f"mean gain={row.mean_cv_macro_f1_gain:+.3f}, median |gap|={row.median_abs_final_gap:.3f}, "
            f"more-data signal={int(row.models_likely_helped_by_more_data)}/{int(row.models)} models."
        )
    overview.extend(["", "Interpretations are descriptive and should not be presented as formal significance tests."])
    (root / "learning_curve_feature_set_interpretation.txt").write_text("\n".join(overview) + "\n", encoding="utf-8")
    print(f"[CLASS][LEARNING CURVES] Summaries saved under: {root}")
    return combined


def _diagnose_classifier(row: pd.Series) -> str:
    """Rule-based screening label; it is not a formal statistical test."""
    train_score = row.get("cv_train_macro_f1", np.nan)
    cv_score = row.get("cv_macro_f1", np.nan)
    cv_sd = row.get("cv_macro_f1_std", np.nan)
    val_score = row.get("val_macro_f1", np.nan)
    gap = row.get("cv_generalization_gap", np.nan)

    labels: list[str] = []
    if np.isfinite(gap) and gap >= 0.10:
        labels.append("possible overfitting")
    if np.isfinite(train_score) and np.isfinite(cv_score) and train_score < 0.55 and cv_score < 0.50:
        labels.append("possible underfitting")
    if np.isfinite(cv_sd) and cv_sd >= 0.15:
        labels.append("high CV instability")
    if np.isfinite(cv_score) and np.isfinite(val_score) and abs(cv_score - val_score) >= 0.15:
        labels.append("split sensitivity")
    return "; ".join(labels) if labels else "acceptable generalization"


def _generate_paper_classification_report(results: list[dict[str, Any]], reports_dir: Path) -> None:
    """Generate paper-ready classification tables, figures, and a brief summary."""
    if not results:
        return

    output_dir = reports_dir / "paper_over_underfitting"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for item in results:
        val = item.get("validation") or {}
        test = item.get("test") or {}
        rows.append({
            "model": item["model"],
            "feature_set": item["feature_set"],
            "target": item["target"],
            "cv_train_macro_f1": item.get("cv_train_macro_f1", np.nan),
            "cv_macro_f1": item.get("cv_macro_f1", np.nan),
            "cv_macro_f1_std": item.get("cv_macro_f1_std", np.nan),
            "cv_generalization_gap": item.get("cv_generalization_gap", np.nan),
            "val_macro_f1": val.get("macro_f1", np.nan),
            "val_balanced_accuracy": val.get("balanced_accuracy", np.nan),
            "test_macro_f1": test.get("macro_f1", np.nan),
            "test_balanced_accuracy": test.get("balanced_accuracy", np.nan),
        })

    table = pd.DataFrame(rows)
    diagnostics = table.apply(
        lambda row: pd.Series(_classification_diagnostics(
            float(row.get("cv_train_macro_f1", np.nan)),
            float(row.get("cv_macro_f1", np.nan)),
            float(row.get("cv_macro_f1_std", np.nan)),
            float(row.get("val_macro_f1", np.nan)),
            float(row.get("test_macro_f1", np.nan)),
        )),
        axis=1,
    )
    for column in diagnostics.columns:
        table[column] = diagnostics[column]
    table["cv_generalization_gap"] = table["train_cv_gap"]
    table["cv_to_validation_shift"] = table["cv_val_gap"]
    table["test_to_validation_shift"] = -table["val_test_gap"]
    table["diagnosis"] = table["generalization_diagnosis"]
    table = table.sort_values(["val_macro_f1", "val_balanced_accuracy"], ascending=False)

    csv_path = output_dir / "classification_over_underfitting_table.csv"
    table.to_csv(csv_path, index=False)

    paper_table = table.rename(columns={
        "model": "Model",
        "feature_set": "Feature set",
        "cv_train_macro_f1": "Train CV Macro-F1",
        "cv_macro_f1": "Validation CV Macro-F1",
        "cv_macro_f1_std": "CV SD",
        "cv_generalization_gap": "Train-CV gap",
        "cv_val_gap": "CV-holdout gap",
        "val_test_gap": "Validation-test gap",
        "val_macro_f1": "Holdout validation Macro-F1",
        "test_macro_f1": "Test Macro-F1",
        "diagnosis": "Interpretation",
    })[["Model", "Feature set", "Train CV Macro-F1", "Validation CV Macro-F1",
        "CV SD", "Train-CV gap", "CV-holdout gap", "Validation-test gap",
        "Holdout validation Macro-F1", "Test Macro-F1", "Interpretation"]]
    latex_path = output_dir / "classification_over_underfitting_table.tex"
    try:
        latex_path.write_text(
            paper_table.to_latex(index=False, float_format=lambda x: f"{x:.3f}", escape=True),
            encoding="utf-8",
        )
    except Exception as error:
        latex_path.write_text(f"% LaTeX export failed: {error}\n", encoding="utf-8")

    labels = [f"{m}\n({fs})" for m, fs in zip(table["model"], table["feature_set"])]
    x = np.arange(len(table))
    width = 0.20
    fig_width = max(10.0, len(table) * 0.72)
    fig, ax = plt.subplots(figsize=(fig_width, 6.0))
    ax.bar(x - 1.5 * width, table["cv_train_macro_f1"], width, label="Training CV")
    ax.bar(x - 0.5 * width, table["cv_macro_f1"], width, label="Validation CV")
    ax.bar(x + 0.5 * width, table["val_macro_f1"], width, label="Holdout validation")
    ax.bar(x + 1.5 * width, table["test_macro_f1"], width, label="Test")
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 1.05)
    ax.set_title("Classifier performance across training, validation, and test stages")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "paper_split_performance_macro_f1.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(fig_width, 5.4))
    ax.bar(x, table["cv_generalization_gap"])
    ax.errorbar(x, table["cv_generalization_gap"], yerr=table["cv_macro_f1_std"],
                fmt="none", capsize=3, label="CV SD")
    ax.axhline(0.10, linestyle="--", linewidth=1.2, label="Review threshold (0.10)")
    ax.axhline(0.00, linewidth=0.9)
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_ylabel("Training CV Macro-F1 − validation CV Macro-F1")
    ax.set_title("Generalization-gap screening for classification models")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "paper_generalization_gap_macro_f1.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    best = table.iloc[0]
    flagged = table[table["diagnosis"] != "acceptable generalization"]
    lines = [
        "CLASSIFICATION OVERFITTING AND UNDERFITTING ANALYSIS",
        "=" * 72,
        "",
        "Assessment basis",
        "----------------",
        "Training and validation Macro-F1 values are taken from the internal",
        "stratified cross-validation used for hyperparameter selection. The",
        "holdout validation set is used for model comparison, while the test set",
        "is reported only as a final independent check.",
        "",
        "Best holdout-validation model",
        "-----------------------------",
        f"Model: {best['model']}",
        f"Feature set: {best['feature_set']}",
        f"Training CV Macro-F1: {best['cv_train_macro_f1']:.4f}",
        f"Validation CV Macro-F1: {best['cv_macro_f1']:.4f} ± {best['cv_macro_f1_std']:.4f}",
        f"Holdout validation Macro-F1: {best['val_macro_f1']:.4f}",
        f"Test Macro-F1: {best['test_macro_f1']:.4f}",
        f"Train-CV generalization gap: {best['cv_generalization_gap']:.4f}",
        f"Interpretation: {best['diagnosis']}",
        "",
        "Rule-based screening criteria",
        "-----------------------------",
        "Large train-CV generalization gap: train-CV Macro-F1 gap >= 0.10.",
        "Possible underfitting: training CV Macro-F1 < 0.55 and validation CV Macro-F1 < 0.50.",
        "High instability: CV standard deviation >= 0.15.",
        "Split sensitivity: |CV Macro-F1 - holdout validation Macro-F1| >= 0.15.",
        "These thresholds are descriptive screening rules, not formal hypothesis tests.",
        "",
        "Paper-ready summary",
        "-------------------",
        "Classifier generalization was examined using training and validation",
        "Macro-F1 scores obtained within stratified cross-validation, together",
        "with performance on separate validation and test sets. The difference",
        "between mean training and validation CV Macro-F1 was used as a",
        "generalization-gap indicator, while the cross-validation standard",
        "deviation was used to assess stability across folds. Models with large",
        "positive gaps were treated as potentially overfitted, whereas models",
        "with consistently low training and validation scores were treated as",
        "potentially underfitted.",
        "",
        f"Flagged model-feature combinations: {len(flagged)} of {len(table)}",
    ]
    if not flagged.empty:
        lines.append("")
        lines.append("Flagged combinations")
        lines.append("--------------------")
        for _, row in flagged.iterrows():
            lines.append(
                f"- {row['model']} / {row['feature_set']}: {row['diagnosis']} "
                f"(gap={row['cv_generalization_gap']:.3f}, CV SD={row['cv_macro_f1_std']:.3f})"
            )

    (output_dir / "over_underfitting.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (output_dir / "over_underfitting_summary.json").open("w", encoding="utf-8") as file:
        json.dump({
            "best_model": best.to_dict(),
            "n_models": int(len(table)),
            "n_flagged": int(len(flagged)),
            "screening_thresholds": {
                "possible_overfitting_gap": 0.10,
                "possible_underfitting_train_macro_f1": 0.55,
                "possible_underfitting_cv_macro_f1": 0.50,
                "high_cv_instability_sd": 0.15,
                "split_sensitivity_absolute_shift": 0.15,
            },
        }, file, indent=2, default=_json_default)


# ============================================================
# GAP-AWARE CLASSIFICATION RANKING AND WEIGHT SENSITIVITY
# ============================================================

CLASSIFICATION_WEIGHT_SCENARIOS = {
    "equal": {
        "rank_cv_macro_f1": 0.20,
        "rank_val_macro_f1": 0.20,
        "rank_cv_stability": 0.20,
        "rank_train_cv_gap": 0.20,
        "rank_val_balanced_error": 0.20,
    },
    "performance_focused": {
        "rank_cv_macro_f1": 0.35,
        "rank_val_macro_f1": 0.35,
        "rank_cv_stability": 0.10,
        "rank_train_cv_gap": 0.10,
        "rank_val_balanced_error": 0.10,
    },
    "cv_focused": {
        "rank_cv_macro_f1": 0.40,
        "rank_val_macro_f1": 0.25,
        "rank_cv_stability": 0.15,
        "rank_train_cv_gap": 0.15,
        "rank_val_balanced_error": 0.05,
    },
    "stability_focused": {
        "rank_cv_macro_f1": 0.25,
        "rank_val_macro_f1": 0.20,
        "rank_cv_stability": 0.20,
        "rank_train_cv_gap": 0.25,
        "rank_val_balanced_error": 0.10,
    },
    "validation_focused": {
        "rank_cv_macro_f1": 0.20,
        "rank_val_macro_f1": 0.45,
        "rank_cv_stability": 0.10,
        "rank_train_cv_gap": 0.15,
        "rank_val_balanced_error": 0.10,
    },
}


def _classification_rank(series: pd.Series, ascending: bool) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").rank(
        ascending=ascending,
        method="average",
        na_option="bottom",
    )


def _classification_weight_diagnosis(
    first_place_rate: float,
    rank_range: float,
    rank_sd: float,
) -> str:
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


def _add_classification_scientific_ranks(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Create selection, test-performance, and generalization ranks.

    The primary Selection Rank excludes all test metrics. It uses:
      * 40% mean CV Macro-F1 rank;
      * 25% holdout validation Macro-F1 rank;
      * 15% CV stability rank;
      * 15% absolute train-CV gap rank;
      * 5% validation balanced-error rank.
    """
    if summary_df is None or summary_df.empty:
        return summary_df

    df = summary_df.copy()
    numeric_columns = [
        "cv_train_macro_f1", "cv_train_macro_f1_std",
        "cv_macro_f1", "cv_macro_f1_std",
        "val_macro_f1", "val_balanced_accuracy",
        "test_macro_f1", "test_balanced_accuracy",
        "train_cv_gap", "train_val_gap", "cv_val_gap", "val_test_gap",
        "abs_train_cv_gap", "abs_cv_val_gap",
    ]
    for column in numeric_columns:
        if column not in df.columns:
            df[column] = np.nan
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Complement of balanced accuracy is an interpretable validation error.
    df["val_balanced_error"] = 1.0 - df["val_balanced_accuracy"]
    df["test_balanced_error"] = 1.0 - df["test_balanced_accuracy"]
    df["abs_val_test_gap"] = df["val_test_gap"].abs()

    df["rank_cv_macro_f1"] = _classification_rank(df["cv_macro_f1"], False)
    df["rank_val_macro_f1"] = _classification_rank(df["val_macro_f1"], False)
    df["rank_cv_stability"] = _classification_rank(df["cv_macro_f1_std"], True)
    df["rank_train_cv_gap"] = _classification_rank(df["abs_train_cv_gap"], True)
    df["rank_val_balanced_error"] = _classification_rank(
        df["val_balanced_error"], True
    )

    df["selection_rank_score"] = (
        0.40 * df["rank_cv_macro_f1"]
        + 0.25 * df["rank_val_macro_f1"]
        + 0.15 * df["rank_cv_stability"]
        + 0.15 * df["rank_train_cv_gap"]
        + 0.05 * df["rank_val_balanced_error"]
    )
    df["Selection Rank"] = df["selection_rank_score"].rank(
        ascending=True, method="min"
    ).astype("Int64")

    # Test rank is descriptive only and never used to select a classifier.
    if df["test_macro_f1"].notna().any():
        df["rank_test_macro_f1"] = _classification_rank(
            df["test_macro_f1"], False
        )
        df["rank_test_balanced_accuracy"] = _classification_rank(
            df["test_balanced_accuracy"], False
        )
        df["test_rank_score"] = (
            0.70 * df["rank_test_macro_f1"]
            + 0.30 * df["rank_test_balanced_accuracy"]
        )
        df["Test Performance Rank"] = df["test_rank_score"].rank(
            ascending=True, method="min"
        ).astype("Int64")
    else:
        df["Test Performance Rank"] = pd.Series(
            pd.NA, index=df.index, dtype="Int64"
        )

    df["rank_generalization_gap"] = _classification_rank(
        df["abs_train_cv_gap"], True
    )
    df["rank_generalization_cv_sd"] = _classification_rank(
        df["cv_macro_f1_std"], True
    )
    df["rank_generalization_split_gap"] = _classification_rank(
        df["abs_cv_val_gap"], True
    )
    df["generalization_rank_score"] = (
        0.45 * df["rank_generalization_gap"]
        + 0.35 * df["rank_generalization_cv_sd"]
        + 0.20 * df["rank_generalization_split_gap"]
    )
    df["Generalization Rank"] = df["generalization_rank_score"].rank(
        ascending=True, method="min"
    ).astype("Int64")

    criterion_ranks = df[[
        "rank_cv_macro_f1", "rank_val_macro_f1", "rank_cv_stability",
        "rank_train_cv_gap", "rank_val_balanced_error",
    ]].to_numpy(dtype=float)
    stability = []
    for row in criterion_ranks:
        mean_rank = np.nanmean(row)
        if not np.isfinite(mean_rank) or mean_rank == 0:
            stability.append(np.nan)
        else:
            stability.append(1.0 / (1.0 + np.nanstd(row) / mean_rank))
    df["selection_rank_stability"] = stability

    return df


def _classification_predefined_sensitivity(
    ranked_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = ranked_df.copy()
    rank_columns: list[str] = []
    scenario_rows: list[dict[str, Any]] = []

    for name, weights in CLASSIFICATION_WEIGHT_SCENARIOS.items():
        score_column = f"{name}_score"
        rank_column = f"{name}_rank"
        score = pd.Series(0.0, index=result.index, dtype=float)
        for criterion, weight in weights.items():
            score = score + float(weight) * pd.to_numeric(
                result[criterion], errors="coerce"
            )
        result[score_column] = score
        result[rank_column] = score.rank(
            ascending=True, method="min", na_option="bottom"
        ).astype("Int64")
        rank_columns.append(rank_column)
        scenario_rows.append({
            "scenario": name,
            "cv_macro_f1_weight": weights["rank_cv_macro_f1"],
            "validation_macro_f1_weight": weights["rank_val_macro_f1"],
            "cv_sd_weight": weights["rank_cv_stability"],
            "train_cv_gap_weight": weights["rank_train_cv_gap"],
            "validation_balanced_error_weight": weights[
                "rank_val_balanced_error"
            ],
            "total_weight": sum(weights.values()),
        })

    frame = result[rank_columns].astype(float)
    result["sensitivity_mean_rank"] = frame.mean(axis=1)
    result["sensitivity_median_rank"] = frame.median(axis=1)
    result["sensitivity_min_rank"] = frame.min(axis=1)
    result["sensitivity_max_rank"] = frame.max(axis=1)
    result["sensitivity_rank_range"] = (
        result["sensitivity_max_rank"] - result["sensitivity_min_rank"]
    )
    result["sensitivity_rank_sd"] = frame.std(axis=1, ddof=1)
    result["first_place_count"] = (frame == 1).sum(axis=1)
    result["first_place_rate"] = (
        result["first_place_count"] / float(len(rank_columns))
    )
    result["weight_sensitivity_diagnosis"] = [
        _classification_weight_diagnosis(rate, span, sd)
        for rate, span, sd in zip(
            result["first_place_rate"],
            result["sensitivity_rank_range"],
            result["sensitivity_rank_sd"],
        )
    ]
    return result, pd.DataFrame(scenario_rows)


def _classification_monte_carlo_sensitivity(
    ranked_df: pd.DataFrame,
    n_simulations: int = 1000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    criteria = [
        "rank_cv_macro_f1", "rank_val_macro_f1", "rank_cv_stability",
        "rank_train_cv_gap", "rank_val_balanced_error",
    ]
    matrix = ranked_df[criteria].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)
    rng = np.random.default_rng(random_state)
    weights_list: list[np.ndarray] = []
    ranks_list: list[np.ndarray] = []
    attempts = 0
    maximum_attempts = max(10000, n_simulations * 100)

    while len(weights_list) < n_simulations and attempts < maximum_attempts:
        attempts += 1
        weights = rng.dirichlet(np.ones(len(criteria)))
        if weights[0] + weights[1] < 0.50:
            continue
        if float(np.max(weights)) > 0.50:
            continue
        scores = matrix @ weights
        ranks = pd.Series(scores).rank(
            ascending=True, method="min", na_option="bottom"
        ).astype(int).to_numpy()
        weights_list.append(weights)
        ranks_list.append(ranks)

    if not weights_list:
        raise RuntimeError("No valid Monte Carlo classification weights generated")

    weights_array = np.asarray(weights_list, dtype=float)
    ranks_array = np.asarray(ranks_list, dtype=int).T
    result = ranked_df.copy()
    result["mc_mean_rank"] = ranks_array.mean(axis=1)
    result["mc_median_rank"] = np.median(ranks_array, axis=1)
    result["mc_min_rank"] = ranks_array.min(axis=1)
    result["mc_max_rank"] = ranks_array.max(axis=1)
    result["mc_rank_range"] = result["mc_max_rank"] - result["mc_min_rank"]
    result["mc_rank_sd"] = ranks_array.std(axis=1, ddof=1)
    result["mc_first_place_count"] = (ranks_array == 1).sum(axis=1)
    result["mc_first_place_rate"] = (
        result["mc_first_place_count"] / float(ranks_array.shape[1])
    )
    result["mc_top3_count"] = (ranks_array <= 3).sum(axis=1)
    result["mc_top3_rate"] = (
        result["mc_top3_count"] / float(ranks_array.shape[1])
    )
    result["mc_weight_sensitivity_diagnosis"] = [
        _classification_weight_diagnosis(rate, span, sd)
        for rate, span, sd in zip(
            result["mc_first_place_rate"],
            result["mc_rank_range"],
            result["mc_rank_sd"],
        )
    ]

    weights_df = pd.DataFrame(weights_array, columns=[
        "cv_macro_f1_weight", "validation_macro_f1_weight", "cv_sd_weight",
        "train_cv_gap_weight", "validation_balanced_error_weight",
    ])
    weights_df.insert(0, "simulation", np.arange(1, len(weights_df) + 1))
    weights_df["performance_weight"] = (
        weights_df["cv_macro_f1_weight"]
        + weights_df["validation_macro_f1_weight"]
    )
    return result, weights_df, ranks_array


def _write_classification_scientific_reports(
    summary_df: pd.DataFrame,
    reports_dir: Path,
    n_simulations: int = 1000,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Write classification counterparts of train7 regression reports."""
    outputs: dict[str, str] = {}
    if summary_df is None or summary_df.empty:
        return summary_df, outputs

    reports_dir.mkdir(parents=True, exist_ok=True)
    ranked = _add_classification_scientific_ranks(summary_df)
    ranked, scenarios = _classification_predefined_sensitivity(ranked)
    ranked, mc_weights, mc_ranks = _classification_monte_carlo_sensitivity(
        ranked, n_simulations=n_simulations, random_state=42
    )
    ranked = ranked.sort_values(
        ["Selection Rank", "cv_macro_f1", "val_macro_f1"],
        ascending=[True, False, False],
        na_position="last",
    ).reset_index(drop=True)

    paths = {
        "ranking": reports_dir / "classification_ranking.csv",
        "enhanced": reports_dir / "ClassificationEnhancedFinalRanking.csv",
        "paper": reports_dir / "Results_classification_scientific_ranking.csv",
        "scenarios": reports_dir / "ClassificationWeightScenarios.csv",
        "sensitivity": reports_dir / "ClassificationWeightSensitivityAnalysis.csv",
        "mc_weights": reports_dir / "ClassificationMonteCarloWeightSensitivity.csv",
        "mc_matrix": reports_dir / "ClassificationMonteCarloRankMatrix.csv",
        "summary": reports_dir / "ClassificationWeightSensitivitySummary.txt",
    }
    ranked.to_csv(paths["ranking"], index=False)
    ranked.to_csv(paths["enhanced"], index=False, encoding="utf-8-sig")
    scenarios.to_csv(paths["scenarios"], index=False)
    ranked.to_csv(paths["sensitivity"], index=False, encoding="utf-8-sig")
    mc_weights.to_csv(paths["mc_weights"], index=False)

    labels = (
        ranked["model"].astype(str) + "-" + ranked["feature_set"].astype(str)
    )
    matrix_df = pd.DataFrame(
        mc_ranks,
        columns=[f"simulation_{i}" for i in range(1, mc_ranks.shape[1] + 1)],
    )
    matrix_df.insert(0, "model_feature_set", labels.to_numpy())
    matrix_df.to_csv(paths["mc_matrix"], index=False)

    paper_columns = [
        "Selection Rank", "Test Performance Rank", "Generalization Rank",
        "target", "model", "feature_set", "cv_train_macro_f1",
        "cv_macro_f1", "cv_macro_f1_std", "val_macro_f1", "test_macro_f1",
        "train_cv_gap", "cv_val_gap", "val_test_gap", "val_balanced_accuracy",
        "test_balanced_accuracy", "generalization_diagnosis",
        "first_place_rate", "mc_first_place_rate", "mc_top3_rate",
        "mc_weight_sensitivity_diagnosis", "file",
    ]
    ranked[[c for c in paper_columns if c in ranked.columns]].to_csv(
        paths["paper"], index=False, encoding="utf-8-sig"
    )

    first_place_plot = reports_dir / "ClassificationMonteCarloFirstPlaceRate.png"
    scenario_plot = reports_dir / "ClassificationWeightSensitivityRankDistribution.png"
    plot_df = ranked.copy()
    plot_df["label"] = labels
    plot_df = plot_df.sort_values("mc_first_place_rate", ascending=False)
    fig, ax = plt.subplots(figsize=(max(8, len(plot_df) * 0.48), 5.5))
    ax.bar(np.arange(len(plot_df)), plot_df["mc_first_place_rate"].astype(float))
    ax.set_xticks(np.arange(len(plot_df)))
    ax.set_xticklabels(plot_df["label"], rotation=55, ha="right")
    ax.set_ylabel("First-place frequency")
    ax.set_ylim(0, 1)
    ax.set_title("Classifier first-place frequency under constrained random weights")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(first_place_plot, dpi=300, bbox_inches="tight")
    plt.close(fig)

    scenario_rank_columns = [
        f"{name}_rank" for name in CLASSIFICATION_WEIGHT_SCENARIOS
    ]
    scenario_df = ranked.sort_values("sensitivity_mean_rank").copy()
    x = np.arange(len(scenario_df))
    fig, ax = plt.subplots(figsize=(max(8, len(scenario_df) * 0.48), 5.8))
    for column in scenario_rank_columns:
        ax.plot(
            x, scenario_df[column].astype(float), marker="o", linewidth=1,
            label=column.replace("_rank", "").replace("_", " "),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(
        scenario_df["model"].astype(str) + "-" + scenario_df["feature_set"].astype(str),
        rotation=55, ha="right",
    )
    ax.set_ylabel("Rank (lower is better)")
    ax.invert_yaxis()
    ax.set_title("Classification rank sensitivity across weighting scenarios")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(scenario_plot, dpi=300, bbox_inches="tight")
    plt.close(fig)

    best = ranked.iloc[0]
    robust = ranked.sort_values(
        ["mc_first_place_rate", "sensitivity_mean_rank"],
        ascending=[False, True],
    ).iloc[0]
    lines = [
        "CLASSIFICATION WEIGHT-SENSITIVITY ANALYSIS",
        "=" * 76,
        "",
        "Primary selection weights:",
        "- Mean CV Macro-F1 rank: 0.40",
        "- Validation Macro-F1 rank: 0.25",
        "- CV standard-deviation rank: 0.15",
        "- Absolute train-CV gap rank: 0.15",
        "- Validation balanced-error rank: 0.05",
        "",
        "Test metrics were excluded from classifier selection.",
        f"Predefined scenarios: {len(CLASSIFICATION_WEIGHT_SCENARIOS)}",
        f"Accepted Monte Carlo weight vectors: {len(mc_weights)}",
        "",
        "Primary selected classifier:",
        f"- Model: {best['model']}",
        f"- Feature set: {best['feature_set']}",
        f"- CV Macro-F1: {best['cv_macro_f1']:.4f} ± {best['cv_macro_f1_std']:.4f}",
        f"- Validation Macro-F1: {best['val_macro_f1']:.4f}",
        f"- Test Macro-F1: {best['test_macro_f1']:.4f}",
        f"- Train-CV gap: {best['train_cv_gap']:.4f}",
        f"- Diagnosis: {best['generalization_diagnosis']}",
        "",
        "Most robust first-place classifier:",
        f"- Model: {robust['model']}",
        f"- Feature set: {robust['feature_set']}",
        f"- Predefined-scenario first-place rate: {robust['first_place_rate']:.1%}",
        f"- Monte Carlo first-place rate: {robust['mc_first_place_rate']:.1%}",
        f"- Monte Carlo top-three rate: {robust['mc_top3_rate']:.1%}",
        f"- Robustness diagnosis: {robust['mc_weight_sensitivity_diagnosis']}",
    ]
    paths["summary"].write_text("\n".join(lines) + "\n", encoding="utf-8")

    outputs.update({name: str(path) for name, path in paths.items()})
    outputs["first_place_plot"] = str(first_place_plot)
    outputs["scenario_plot"] = str(scenario_plot)
    return ranked, outputs


def _build_cv_plan(
    y_train: np.ndarray,
    groups: np.ndarray | None,
    params: dict[str, Any],
    random_state: int,
) -> CVPlan:
    """Build leakage-safe classification CV without touching final test data.

    With grouping information, StratifiedGroupKFold is preferred so that:
      * no sampling group appears in both training and validation folds; and
      * class proportions are preserved as well as the group constraints allow.

    If the grouped class structure makes StratifiedGroupKFold unusable,
    GroupKFold is used as a transparent fallback.
    """
    class_config = params.get("classification", {}) or {}
    folds_requested = int(
        class_config.get("cv_splits", params.get("cv_splits", 5))
    )
    repeats = int(class_config.get("cv_repeats", 3))

    y_train = np.asarray(y_train, dtype=int).ravel()
    counts = np.bincount(y_train, minlength=len(CLASS_LABELS))
    positive = counts[counts > 0]

    if len(positive) != len(CLASS_LABELS):
        raise ValueError(
            f"All configured classes are required for CV; counts={counts.tolist()}"
        )

    folds = min(folds_requested, int(positive.min()))
    if folds < 2:
        raise ValueError(
            "At least two observations per class are required for CV."
        )

    if groups is not None:
        groups = np.asarray(groups).ravel()
        unique_groups = np.unique(groups)
        folds = min(folds, len(unique_groups))
        if folds < 2:
            raise ValueError(
                "At least two distinct groups are required for grouped CV."
            )

        # A class must occur in enough distinct groups for meaningful
        # stratified grouped CV. Reduce folds when necessary.
        class_group_counts = []
        for label in CLASS_LABELS:
            n_groups = len(np.unique(groups[y_train == label]))
            class_group_counts.append(n_groups)

        minimum_class_groups = min(class_group_counts)
        if minimum_class_groups >= 2:
            stratified_folds = min(folds, minimum_class_groups)
            cv = StratifiedGroupKFold(
                n_splits=stratified_folds,
                shuffle=True,
                random_state=random_state,
            )
            return CVPlan(
                cv,
                cv,
                groups,
                "stratified_group_kfold",
                stratified_folds,
                1,
            )

        # Fallback retains leakage protection even when a minority class is
        # concentrated in too few groups to support stratification.
        cv = GroupKFold(n_splits=folds)
        return CVPlan(
            cv,
            cv,
            groups,
            "group_kfold_fallback",
            folds,
            1,
        )

    tuning = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=random_state,
    )
    stability = RepeatedStratifiedKFold(
        n_splits=folds,
        n_repeats=max(1, repeats),
        random_state=random_state,
    )
    return CVPlan(
        tuning,
        stability,
        None,
        "repeated_stratified",
        folds,
        max(1, repeats),
    )


def _prefix_classifier_grid(
    grid: dict[str, list[Any]],
) -> tuple[dict[str, list[Any]], int, int]:
    """Sanitize YAML control keys and prefix estimator parameters.

    ``_cv_`` and ``_n_jobs_`` configure the search itself and must not be sent
    to the classifier. ANN hidden-layer definitions loaded from YAML are lists,
    while scikit-learn expects tuples.
    """
    raw = dict(grid or {})
    model_cv = int(raw.pop("_cv_", 5))
    search_n_jobs = int(raw.pop("_n_jobs_", -1))
    clean: dict[str, list[Any]] = {}

    for key, candidates in raw.items():
        values = candidates if isinstance(candidates, list) else [candidates]
        plain_key = key.removeprefix("classifier__")
        if plain_key == "hidden_layer_sizes":
            values = [
                tuple(value) if isinstance(value, list) else value
                for value in values
            ]
        prefixed_key = (
            key if key.startswith("classifier__") else f"classifier__{key}"
        )
        clean[prefixed_key] = values
    return clean, model_cv, search_n_jobs


def _make_search(
    model_name: str,
    params: dict[str, Any],
    preprocessing_config: dict[str, Any],
    cv_plan: CVPlan,
    random_state: int,
):
    spec = load_classifier_spec(model_name)
    estimator = _build_pipeline(spec.build_estimator(random_state), preprocessing_config)
    configured = (
        ((params.get("classification", {}) or {}).get("param_grids", {}) or {})
        .get(model_name)
    )
    if configured is None:
        configured = (params.get("classification_param_grids", {}) or {}).get(model_name)
    if configured is None:
        configured = (params.get("param_grids_classification", {}) or {}).get(model_name)
    grid, configured_cv, configured_n_jobs = _prefix_classifier_grid(
        configured if configured is not None else spec.default_grid
    )
    combinations = int(np.prod([len(v) for v in grid.values()])) if grid else 1
    class_config = params.get("classification", {}) or {}
    threshold = int(class_config.get("random_search_threshold", 100))
    n_iter = int(class_config.get("n_iter", 40))
    common = dict(
        scoring=_score_macro_f1,
        cv=cv_plan.tuning_cv,
        n_jobs=configured_n_jobs,
        refit=True,
        error_score="raise",
        return_train_score=True,
    )
    if combinations > threshold:
        return RandomizedSearchCV(
            estimator, param_distributions=grid, n_iter=min(n_iter, combinations),
            random_state=random_state, **common
        ), spec.explanation
    return GridSearchCV(estimator, param_grid=grid, **common), spec.explanation



def _nested_cv_classification(
    model_name: str,
    params: dict[str, Any],
    preprocessing_config: dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None,
    random_state: int,
    metadata: pd.DataFrame | None = None,
    class_names: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Run inner hyperparameter search inside unbiased outer CV folds."""
    outer_params = dict(params)
    outer_class_config = dict(params.get("classification", {}) or {})
    outer_class_config["cv_repeats"] = int(
        outer_class_config.get("nested_cv_outer_repeats", 1)
    )
    outer_params["classification"] = outer_class_config
    outer_plan = _build_cv_plan(y, groups, outer_params, random_state)
    split_args = (X, y, groups) if groups is not None else (X, y)
    outer_splits = list(outer_plan.stability_cv.split(*split_args))

    train_macro_scores: list[float] = []
    outer_macro_scores: list[float] = []
    outer_balanced_scores: list[float] = []
    outer_accuracy_scores: list[float] = []
    fold_rows: list[dict[str, Any]] = []
    group_diagnostic_rows: list[dict[str, Any]] = []
    params_per_fold: list[dict[str, Any]] = []
    class_names = _normalize_class_names(class_names, "AC")

    for fold_index, (train_idx, test_idx) in enumerate(outer_splits, start=1):
        X_inner, y_inner = X[train_idx], y[train_idx]
        X_outer, y_outer = X[test_idx], y[test_idx]
        groups_inner = groups[train_idx] if groups is not None else None

        inner_plan = _build_cv_plan(
            y_inner, groups_inner, params, random_state + fold_index
        )
        search, _ = _make_search(
            model_name,
            params,
            preprocessing_config,
            inner_plan,
            random_state + fold_index,
        )
        fit_kwargs = {"groups": groups_inner} if groups_inner is not None else {}
        search.fit(X_inner, y_inner, **fit_kwargs)
        estimator = search.best_estimator_

        train_pred = estimator.predict(X_inner)
        outer_pred = estimator.predict(X_outer)
        train_macro = float(
            f1_score(y_inner, train_pred, average="macro", zero_division=0)
        )
        outer_macro = float(
            f1_score(y_outer, outer_pred, average="macro", zero_division=0)
        )
        outer_balanced = float(balanced_accuracy_score(y_outer, outer_pred))
        outer_accuracy = float(accuracy_score(y_outer, outer_pred))

        train_macro_scores.append(train_macro)
        outer_macro_scores.append(outer_macro)
        outer_balanced_scores.append(outer_balanced)
        outer_accuracy_scores.append(outer_accuracy)
        params_per_fold.append(search.best_params_)
        fold_row = {
            "outer_fold": fold_index,
            "n_inner_train": len(train_idx),
            "n_outer_validation": len(test_idx),
            "train_macro_f1": train_macro,
            "outer_macro_f1": outer_macro,
            "outer_balanced_accuracy": outer_balanced,
            "outer_accuracy": outer_accuracy,
            "train_outer_gap": train_macro - outer_macro,
            "inner_class_counts": json.dumps(
                np.bincount(y_inner, minlength=len(CLASS_LABELS)).tolist()
            ),
            "outer_class_counts": json.dumps(
                np.bincount(y_outer, minlength=len(CLASS_LABELS)).tolist()
            ),
            "best_params": json.dumps(
                search.best_params_, default=_json_default
            ),
        }

        if groups is not None:
            train_group_values = set(
                np.asarray(groups)[train_idx].tolist()
            )
            validation_group_values = set(
                np.asarray(groups)[test_idx].tolist()
            )
            overlap = train_group_values.intersection(
                validation_group_values
            )
            if overlap:
                raise RuntimeError(
                    f"Group leakage detected in classification outer fold "
                    f"{fold_index}: {sorted(map(str, overlap))}"
                )

            fold_row.update({
                "n_train_groups": len(train_group_values),
                "n_validation_groups": len(validation_group_values),
                "group_overlap_count": 0,
                "train_groups": json.dumps(
                    sorted(map(str, train_group_values))
                ),
                "validation_groups": json.dumps(
                    sorted(map(str, validation_group_values))
                ),
            })

        group_diagnostic_rows.extend(
            _build_outer_group_classification_diagnostics(
                fold_index=fold_index,
                y_true=y_outer,
                y_pred=outer_pred,
                group_values=groups,
                metadata=metadata,
                validation_idx=test_idx,
                class_names=class_names,
            )
        )
        fold_rows.append(fold_row)

    train_values = np.asarray(train_macro_scores, dtype=float)
    macro_values = np.asarray(outer_macro_scores, dtype=float)
    balanced_values = np.asarray(outer_balanced_scores, dtype=float)
    accuracy_values = np.asarray(outer_accuracy_scores, dtype=float)
    return {
        "train_macro_scores": train_values,
        "outer_macro_scores": macro_values,
        "outer_balanced_scores": balanced_values,
        "outer_accuracy_scores": accuracy_values,
        "train_macro_mean": float(np.mean(train_values)),
        "train_macro_std": float(np.std(train_values, ddof=1)) if len(train_values) > 1 else 0.0,
        "outer_macro_mean": float(np.mean(macro_values)),
        "outer_macro_std": float(np.std(macro_values, ddof=1)) if len(macro_values) > 1 else 0.0,
        "outer_balanced_mean": float(np.mean(balanced_values)),
        "outer_balanced_std": float(np.std(balanced_values, ddof=1)) if len(balanced_values) > 1 else 0.0,
        "outer_accuracy_mean": float(np.mean(accuracy_values)),
        "outer_accuracy_std": float(np.std(accuracy_values, ddof=1)) if len(accuracy_values) > 1 else 0.0,
        "fold_rows": fold_rows,
        "group_diagnostic_rows": group_diagnostic_rows,
        "best_params_per_fold": params_per_fold,
        "outer_mode": outer_plan.mode,
        "outer_folds": outer_plan.folds,
        "outer_repeats": outer_plan.repeats,
    }


def _cv_only_rank(summary: pd.DataFrame) -> pd.DataFrame:
    """Authoritative classifier selection rank; validation and test are excluded."""
    frame = summary.copy()
    for column in [
        "cv_macro_f1", "cv_macro_f1_std", "cv_balanced_accuracy",
        "cv_accuracy", "abs_train_cv_gap",
    ]:
        if column not in frame:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["rank_cv_macro_f1"] = _classification_rank(frame["cv_macro_f1"], False)
    frame["rank_cv_balanced_accuracy"] = _classification_rank(frame["cv_balanced_accuracy"], False)
    frame["rank_cv_accuracy"] = _classification_rank(frame["cv_accuracy"], False)
    frame["rank_cv_stability"] = _classification_rank(frame["cv_macro_f1_std"], True)
    frame["rank_train_cv_gap"] = _classification_rank(frame["abs_train_cv_gap"], True)
    frame["selection_rank_score"] = (
        0.40 * frame["rank_cv_macro_f1"]
        + 0.20 * frame["rank_cv_balanced_accuracy"]
        + 0.10 * frame["rank_cv_accuracy"]
        + 0.15 * frame["rank_cv_stability"]
        + 0.15 * frame["rank_train_cv_gap"]
    )
    frame["Selection Rank"] = frame["selection_rank_score"].rank(
        ascending=True, method="min"
    ).astype("Int64")
    frame["selection_policy"] = "CV only: Macro-F1 40%, balanced accuracy 20%, accuracy 10%, stability 15%, train-CV gap 15%"
    frame["test_used_for_selection"] = False
    return frame.sort_values(["Selection Rank", "cv_macro_f1"], ascending=[True, False])


def _select_models_before_test(
    ranked: pd.DataFrame, models_dir: Path, reports_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    overall = ranked.iloc[0].to_dict()
    overall["selection_scope"] = "best_overall"
    selected.append(overall)
    for (_, _), group in ranked.groupby(["target", "feature_set"], sort=False):
        row = group.sort_values("Selection Rank").iloc[0].to_dict()
        row["selection_scope"] = "best_per_feature_set"
        selected.append(row)
    selected_frame = pd.DataFrame(selected).drop_duplicates(subset=["file", "selection_scope"])
    selected_frame.to_csv(reports_dir / "selected_models_before_test.csv", index=False, encoding="utf-8-sig")
    source = models_dir / str(overall["file"])
    if source.exists():
        shutil.copy2(source, models_dir / "best_classifier.pkl")
    return selected_frame.to_dict("records"), overall


def _evaluate_selected_models(
    selected: list[dict[str, Any]], paths: OutputPaths
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    prediction_dir = paths.reports_dir / "test_predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for item in selected:
        target, fs = str(item["target"]), str(item["feature_set"])
        x_path = paths.data_dir / f"X_test_{fs}.joblib"
        y_path = paths.data_dir / f"y_test_{target}.joblib"
        model_path = paths.models_dir / str(item["file"])
        if not (x_path.exists() and y_path.exists() and model_path.exists()):
            print(f"[CLASS][TEST][WARN] Missing test/model files for {item['file']}")
            continue
        package = joblib.load(model_path)
        model = package["model"] if isinstance(package, dict) else package
        X = _ensure_2d(joblib.load(x_path))
        y_raw = _ensure_1d(joblib.load(y_path))
        if isinstance(package, dict):
            boundary_info = package.get("class_boundary_values") or {}
            package_names = package.get("class_names") or {}
        else:
            boundary_info = {}
            package_names = {}

        if isinstance(boundary_info, dict) and boundary_info.get("mode") == "categorical":
            class_values = list(boundary_info.get("class_values", []))
            globals()["CLASS_LABELS"] = list(range(len(class_values)))
            globals()["CLASS_NAMES"] = {i: str(v) for i, v in enumerate(class_values)}
            class_names = _normalize_class_names(package_names, target)
            mask = pd.Series(y_raw).notna().to_numpy() & np.all(np.isfinite(X), axis=1)
            X, y = X[mask], _encode_target_values(y_raw[mask], boundary_info)
        else:
            globals()["CLASS_LABELS"] = [0, 1, 2]
            if not (isinstance(boundary_info, dict) and "lower" in boundary_info and "upper" in boundary_info):
                boundary_info = {"lower": 0.75, "upper": 1.25, "mode": "threshold"}
            class_names = _normalize_class_names(package_names, target)
            y_numeric = pd.to_numeric(pd.Series(y_raw), errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(y_numeric) & np.all(np.isfinite(X), axis=1)
            X, y = X[mask], _continuous_to_class(y_numeric[mask], boundary_info)
        metrics = _evaluate(model, X, y, class_names)
        rows.append({
            "selection_scope": item["selection_scope"], "target": target,
            "model": item["model"], "feature_set": fs, "file": item["file"],
            "n_test": len(y), "macro_f1_test": metrics["macro_f1"],
            "balanced_accuracy_test": metrics["balanced_accuracy"],
            "accuracy_test": metrics["accuracy"],
            "weighted_f1_test": metrics["weighted_f1"],
            "cv_macro_f1": item.get("cv_macro_f1", np.nan),
            "cv_test_gap": item.get("cv_macro_f1", np.nan) - metrics["macro_f1"],
            "test_used_for_selection": False,
        })
        predictions = model.predict(X)
        pd.DataFrame({
            "actual_class": y, "predicted_class": predictions,
            "actual_label": [class_names[int(v)] for v in y],
            "predicted_label": [class_names[int(v)] for v in predictions],
        }).to_csv(prediction_dir / f"{Path(str(item['file'])).stem}_test_predictions.csv", index=False)
        with (prediction_dir / f"{Path(str(item['file'])).stem}_test_report.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, default=_json_default)
    frame = pd.DataFrame(rows)
    frame.to_csv(paths.reports_dir / "selected_models_final_test_evaluation.csv", index=False, encoding="utf-8-sig")
    return frame


def train_classification(
    data_dir: str | Path,
    models_dir: str | Path,
    reports_dir: str | Path,
    params_path: str | Path = "params.yaml",
) -> list[dict[str, Any]]:
    """Run modular classifier training with CV-only selection and final test isolation."""
    paths = OutputPaths(
        Path(data_dir), _class_output_dir(models_dir), _class_output_dir(reports_dir)
    )
    paths.models_dir.mkdir(parents=True, exist_ok=True)
    paths.reports_dir.mkdir(parents=True, exist_ok=True)

    params = _load_params(params_path)
    class_config = params.get("classification", {}) or {}
    if not bool(class_config.get("enabled", True)):
        print("[CLASS] Classification is disabled in params.yaml")
        return []
    feature_sets = params.get("TARGETS", {}) or {}
    preprocessing_config = params.get("preprocessing", {}) or {}
    evaluation_config = params.get("evaluation", {}) or {}
    generate_permutation = bool(evaluation_config.get("generate_permutation_sensitivity", True))
    permutation_repeats = int(evaluation_config.get("permutation_sensitivity_repeats", 30))
    permutation_n_jobs = int(evaluation_config.get("permutation_sensitivity_n_jobs", -1))
    random_state = int(class_config.get("random_state", params.get("random_state", 42)))
    # params.yaml is the single source of truth for active classifiers.
    # Preferred schema:
    # classification:
    #   models: [LOGREG, SVC, KNN, DT, RF, ET, GBC, XGB, ANN]
    wanted_models_raw = class_config.get(
        "models",
        params.get("classification_models", available_classifiers()),
    )
    wanted_models = []
    seen_models = set()
    for raw_name in wanted_models_raw:
        name = str(raw_name).strip().upper()
        if name and name not in seen_models:
            wanted_models.append(name)
            seen_models.add(name)
    validate_requested_classifiers(wanted_models)
    classification_targets = class_config.get("targets")
    if classification_targets in (None, [], "all", "ALL", "*"):
        classification_targets = list(feature_sets.keys())
    elif isinstance(classification_targets, str):
        classification_targets = [classification_targets]

    print("\n" + "=" * 72)
    print("CLASSIFICATION: DEVELOPMENT CV -> SELECTION -> INDEPENDENT TEST")
    print(f"Models output:  {paths.models_dir}")
    print(f"Reports output: {paths.reports_dir}")
    print("=" * 72)

    all_results: list[dict[str, Any]] = []
    for target, fs_definitions in feature_sets.items():
        if classification_targets and target not in classification_targets:
            continue
        y_train_path = paths.data_dir / f"y_train_{target}.joblib"
        if not y_train_path.exists():
            print(f"[CLASS][WARN] Missing {y_train_path.name}")
            continue
        y_raw = _ensure_1d(joblib.load(y_train_path))
        target_mode = _classification_target_mode(params, target, y_raw)

        if target_mode == "categorical":
            y_mask = pd.Series(y_raw).notna().to_numpy()
            y_train, class_names, class_values = _encode_existing_classes(y_raw[y_mask])
            globals()["CLASS_LABELS"] = list(range(len(class_values)))
            globals()["CLASS_NAMES"] = dict(class_names)
            boundary_info = {
                "mode": "categorical",
                "source": "existing_labels",
                "method": "direct_label_encoding",
                "class_values": class_values,
                "class_names": class_names,
                "training_class_counts": np.bincount(y_train, minlength=len(class_values)).tolist(),
            }
            print(
                f"[CLASS] Existing classes for {target}: "
                f"{class_values}; counts={boundary_info['training_class_counts']}"
            )
        else:
            y_numeric = pd.to_numeric(pd.Series(y_raw), errors="coerce").to_numpy(dtype=float)
            y_mask = np.isfinite(y_numeric)
            y_raw = y_numeric
            globals()["CLASS_LABELS"] = [0, 1, 2]
            globals()["CLASS_NAMES"] = {0: "Inactive", 1: "Normal", 2: "Active"}
            try:
                boundary_info = _resolve_class_boundaries(params, target, y_raw[y_mask])
            except ClassificationBoundaryCancelled as error:
                print(f"[CLASS][INFO] {error}")
                print(
                    "[CLASS][INFO] Classification training stopped by user. "
                    "No automatic class limits were applied."
                )
                return []
            class_names = _normalize_class_names(boundary_info.get("class_names"), target)
            boundary_info["mode"] = "threshold"
            boundary_info["class_names"] = class_names
            y_train = _continuous_to_class(y_raw[y_mask], boundary_info)
            print(
                f"[CLASS] Boundaries for {target}: "
                f"lower={boundary_info['lower']:.6g}, upper={boundary_info['upper']:.6g} "
                f"(source={boundary_info.get('source')}, method={boundary_info.get('method')})"
            )

        groups_path = paths.data_dir / f"groups_train_{target}.joblib"
        if not groups_path.exists():
            groups_path = paths.data_dir / "groups_train.joblib"
        groups = (
            np.asarray(joblib.load(groups_path)).ravel()[y_mask]
            if groups_path.exists()
            else None
        )
        metadata = _load_aligned_train_metadata(
            paths.data_dir, target, y_mask
        )

        if groups is not None:
            unique_groups = np.unique(groups)
            print(
                f"[CLASS][GROUP CV] {target}: "
                f"{len(unique_groups)} development groups across "
                f"{len(y_train)} samples."
            )
            for label in CLASS_LABELS:
                label_groups = len(
                    np.unique(groups[y_train == label])
                )
                print(
                    f"[CLASS][GROUP CV]   class {label} "
                    f"({class_names[label]}): "
                    f"{int(np.sum(y_train == label))} samples in "
                    f"{label_groups} groups."
                )
        else:
            print(
                f"[CLASS][GROUP CV][WARN] {target}: "
                "no compatible groups_train.joblib found; "
                "row-level stratified CV will be used."
            )

        try:
            cv_plan = _build_cv_plan(y_train, groups, params, random_state)
        except ValueError as error:
            counts = np.bincount(
                y_train.astype(int), minlength=len(CLASS_LABELS)
            ).tolist()
            _show_classification_cv_error(target, error, counts)
            print(
                "[CLASS][INFO] Classification training terminated cleanly "
                "because the class distribution is not usable for CV."
            )
            return []
        print(f"\n[CLASS] Target={target}; classes={np.bincount(y_train, minlength=len(CLASS_LABELS)).tolist()}; CV={cv_plan.mode}")

        for fs, feature_names in fs_definitions.items():
            x_train_path = paths.data_dir / f"X_train_{fs}.joblib"
            if not x_train_path.exists():
                print(f"[CLASS][WARN] Missing {x_train_path.name}")
                continue
            X_train = _ensure_2d(joblib.load(x_train_path))[y_mask]
            finite = np.all(np.isfinite(X_train), axis=1)
            X_train_local, y_train_local = X_train[finite], y_train[finite]
            groups_local = (
                cv_plan.groups[finite]
                if cv_plan.groups is not None else None
            )
            metadata_local = (
                metadata.loc[finite].reset_index(drop=True)
                if metadata is not None else None
            )

            try:
                local_plan = _build_cv_plan(
                    y_train_local,
                    groups_local,
                    params,
                    random_state,
                )
            except ValueError as error:
                counts = np.bincount(
                    y_train_local.astype(int), minlength=len(CLASS_LABELS)
                ).tolist()
                _show_classification_cv_error(
                    f"{target} / {fs}", error, counts
                )
                print(
                    f"[CLASS][INFO] Classification training terminated cleanly "
                    f"at {target}/{fs} because the class distribution is not "
                    "usable for CV."
                )
                return []

            for model_name in wanted_models:
                try:
                    print(f"[CLASS] Training {model_name} | {target} | {fs}")
                    nested = _nested_cv_classification(
                        model_name=model_name,
                        params=params,
                        preprocessing_config=preprocessing_config,
                        X=X_train_local,
                        y=y_train_local,
                        groups=groups_local,
                        random_state=random_state,
                        metadata=metadata_local,
                        class_names=class_names,
                    )

                    # Final inner search on all development data. Its estimator
                    # is saved, but model ranking uses only outer-CV metrics.
                    search, explanation = _make_search(
                        model_name, params, preprocessing_config,
                        local_plan, random_state,
                    )
                    search.fit(
                        X_train_local,
                        y_train_local,
                        **({"groups": groups_local} if groups_local is not None else {}),
                    )
                    fitted = search.best_estimator_
                    train_macro = nested["train_macro_mean"]
                    cv_macro = nested["outer_macro_mean"]
                    cv_sd = nested["outer_macro_std"]
                    gap = train_macro - cv_macro

                    nested_dir = (
                        paths.reports_dir / "nested_cv" / target / fs
                    )
                    nested_dir.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame(nested["fold_rows"]).to_csv(
                        nested_dir / f"{model_name}_nested_cv_folds.csv",
                        index=False,
                        encoding="utf-8-sig",
                    )
                    group_diagnostics = pd.DataFrame(
                        nested.get("group_diagnostic_rows", [])
                    )
                    if not group_diagnostics.empty:
                        group_diagnostics.to_csv(
                            nested_dir
                            / f"{model_name}_nested_cv_group_diagnostics.csv",
                            index=False,
                            encoding="utf-8-sig",
                        )
                    diagnostics = _classification_diagnostics(
                        train_macro, cv_macro, cv_sd, np.nan
                    )
                    hp = _generate_hyperparameter_analysis(search, paths.reports_dir, target, fs, model_name)
                    curve = {}
                    if bool(evaluation_config.get("generate_learning_curves", True)):
                        curve = _generate_learning_curve_report(
                            fitted, X_train_local, y_train_local, local_plan.stability_cv,
                            paths.reports_dir, target, fs, model_name,
                            scoring=str(evaluation_config.get("learning_curve_scoring_classification", "f1_macro")),
                            train_sizes=evaluation_config.get("learning_curve_train_sizes", [0.2, 0.4, 0.6, 0.8, 1.0]),
                            groups=groups_local,
                        )

                    permutation_sensitivity = {"performed": False}
                    if generate_permutation:
                        try:
                            permutation_sensitivity = _generate_permutation_sensitivity_classification(
                                fitted, X_train_local, y_train_local, list(feature_names),
                                paths.reports_dir, target, fs, model_name,
                                n_repeats=permutation_repeats,
                                random_state=random_state,
                                n_jobs=permutation_n_jobs,
                            )
                        except Exception as error:
                            permutation_sensitivity = {
                                "performed": False, "error": str(error)
                            }
                            print(f"[CLASS][WARN] Permutation sensitivity failed for {model_name}/{fs}: {error}")

                    validation = {}
                    xv, yv = paths.data_dir / f"X_val_{fs}.joblib", paths.data_dir / f"y_val_{target}.joblib"
                    if xv.exists() and yv.exists():
                        X_val, y_val_raw = _ensure_2d(joblib.load(xv)), _ensure_1d(joblib.load(yv))
                        val_mask = pd.Series(y_val_raw).notna().to_numpy() & np.all(np.isfinite(X_val), axis=1)
                        validation = _evaluate(
                            fitted,
                            X_val[val_mask],
                            _encode_target_values(y_val_raw[val_mask], boundary_info),
                            class_names,
                        )

                    model_path = paths.models_dir / f"{model_name}_{fs}_{target}_class.pkl"
                    package = {
                        "model": fitted, "model_type": model_name, "model_explanation": explanation,
                        "feature_set": fs, "feature_names": list(feature_names), "target": target,
                        "class_names": class_names,
                        "classification_mode": boundary_info.get("mode", "threshold"),
                        "class_boundary_values": (
                            {
                                "lower": float(boundary_info["lower"]),
                                "upper": float(boundary_info["upper"]),
                                "source": boundary_info.get("source"),
                                "method": boundary_info.get("method"),
                                "quantiles": boundary_info.get("quantiles"),
                                "mode": "threshold",
                            }
                            if boundary_info.get("mode") != "categorical"
                            else {
                                "mode": "categorical",
                                "source": "existing_labels",
                                "method": "direct_label_encoding",
                                "class_values": list(boundary_info.get("class_values", [])),
                            }
                        ),
                        "class_boundaries": (
                            _boundary_description(target, boundary_info)
                            if boundary_info.get("mode") != "categorical"
                            else {class_names[i]: f"{target} = {class_names[i]}" for i in CLASS_LABELS}
                        ),
                        "best_params": search.best_params_,
                        "cv_mode": "nested_" + nested["outer_mode"],
                        "group_aware_cv": groups_local is not None,
                        "n_development_groups": (
                            int(len(np.unique(groups_local)))
                            if groups_local is not None else None
                        ),
                        "cv_folds": nested["outer_folds"],
                        "cv_repeats": nested["outer_repeats"],
                        "inner_cv_folds": local_plan.folds,
                        "nested_cv": True,
                        "selection_policy": "Cross-validation only", "test_policy": "Final holdout only",
                        "test_used_for_selection": False, "preprocessing_config": preprocessing_config,
                    }
                    joblib.dump(package, model_path)
                    result = {
                        "model": model_name, "feature_set": fs, "fs": fs, "target": target,
                        "file": model_path.name, "model_path": str(model_path),
                        "best_params": search.best_params_, "best_tuning_cv_macro_f1": float(search.best_score_),
                        "hyperparameter_selection_method": type(search).__name__,
                        "cv_train_macro_f1": train_macro,
                        "cv_train_macro_f1_std": nested["train_macro_std"],
                        "cv_macro_f1": cv_macro,
                        "cv_macro_f1_std": cv_sd,
                        "cv_macro_f1_min": float(np.min(nested["outer_macro_scores"])),
                        "cv_macro_f1_max": float(np.max(nested["outer_macro_scores"])),
                        "cv_balanced_accuracy": nested["outer_balanced_mean"],
                        "cv_balanced_accuracy_std": nested["outer_balanced_std"],
                        "cv_accuracy": nested["outer_accuracy_mean"],
                        "cv_accuracy_std": nested["outer_accuracy_std"],
                        "cv_generalization_gap": gap, "train_cv_gap": gap,
                        "abs_train_cv_gap": abs(gap), "generalization_diagnosis": diagnostics.get("generalization_diagnosis", ""),
                        "possible_overfitting": diagnostics.get("possible_overfitting", False),
                        "possible_underfitting": diagnostics.get("possible_underfitting", False),
                        "large_generalization_gap": diagnostics.get("large_generalization_gap", False),
                        "high_cv_instability": diagnostics.get("high_cv_instability", False),
                        "high_between_group_variability": diagnostics.get("high_between_group_variability", False),
                        "hyperparameter_analysis": hp, "learning_curve": curve,
                        "permutation_sensitivity": permutation_sensitivity,
                        "cv_mode": "nested_" + nested["outer_mode"],
                        "group_aware_cv": groups_local is not None,
                        "n_development_groups": (
                            int(len(np.unique(groups_local)))
                            if groups_local is not None else None
                        ),
                        "cv_folds": nested["outer_folds"],
                        "cv_repeats": nested["outer_repeats"],
                        "inner_cv_folds": local_plan.folds,
                        "nested_cv": True,
                        "nested_best_params_per_fold": nested["best_params_per_fold"],
                        "n_train": len(y_train_local), "n_features": X_train_local.shape[1],
                        "classification_mode": boundary_info.get("mode", "threshold"),
                        "class_boundary_lower": (float(boundary_info["lower"]) if boundary_info.get("mode") != "categorical" else np.nan),
                        "class_boundary_upper": (float(boundary_info["upper"]) if boundary_info.get("mode") != "categorical" else np.nan),
                        "class_boundary_source": boundary_info.get("source"),
                        "class_boundary_method": boundary_info.get("method"),
                        "validation": validation, "test_used_for_selection": False,
                    }
                    all_results.append(result)
                    print(
                        f"  Outer-train Macro-F1={train_macro:.4f} | "
                        f"Nested outer-CV Macro-F1={cv_macro:.4f} ± {cv_sd:.4f} | "
                        f"Balanced Acc={result['cv_balanced_accuracy']:.4f} | "
                        f"{diagnostics['generalization_diagnosis']}"
                    )
                except Exception as error:
                    print(f"[CLASS][ERROR] {model_name}/{fs} failed: {error}")

    if not all_results:
        print(
            "[CLASS][WARN] No classification model trained successfully. "
            "Classification training will terminate cleanly."
        )
        return []

    if bool(evaluation_config.get("generate_learning_curves", True)):
        _summarize_classification_learning_curves(paths.reports_dir)

    with (paths.reports_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(all_results, handle, indent=2, default=_json_default)
    with (paths.reports_dir / "classification_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(all_results, handle, indent=2, default=_json_default)

    summary = _flatten_results(all_results)
    # Ensure CV-only fields survive the legacy flattener.
    extra = pd.DataFrame([{k: v for k, v in item.items() if k in {
        "file", "cv_balanced_accuracy", "cv_balanced_accuracy_std", "cv_accuracy",
        "cv_accuracy_std", "cv_macro_f1_min", "cv_macro_f1_max", "abs_train_cv_gap",
        "cv_folds", "cv_repeats", "n_train", "n_features", "test_used_for_selection"
    }} for item in all_results])
    summary = summary.drop(columns=[c for c in extra.columns if c != "file" and c in summary.columns], errors="ignore").merge(extra, on="file", how="left")
    ranked = _cv_only_rank(summary)
    ranked.to_csv(paths.reports_dir / "classification_summary.csv", index=False, encoding="utf-8-sig")
    ranked.to_csv(paths.reports_dir / "classification_ranking.csv", index=False, encoding="utf-8-sig")
    _generate_results_classification_summary(ranked, paths.reports_dir)
    _generate_paper_classification_report(all_results, paths.reports_dir)

    selected, overall = _select_models_before_test(ranked, paths.models_dir, paths.reports_dir)
    final_test = _evaluate_selected_models(selected, paths)
    meta = {
        "model": overall["model"], "feature_set": overall["feature_set"],
        "target": overall["target"], "file": overall["file"],
        "selection_basis": "Cross-validation only",
        "cv_macro_f1": overall["cv_macro_f1"],
        "cv_macro_f1_std": overall["cv_macro_f1_std"],
        "cv_balanced_accuracy": overall.get("cv_balanced_accuracy"),
        "train_cv_gap": overall.get("train_cv_gap"),
        "test_used_for_selection": False,
    }
    with (paths.reports_dir / "best_classifier_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, default=_json_default)

    print("\n" + "=" * 72)
    print("CLASSIFICATION TRAINING COMPLETED")
    print(f"Models trained: {len(all_results)}")
    print(f"Selected overall: {overall['model']} / {overall['feature_set']}")
    print(f"Nested outer-CV Macro-F1: {overall['cv_macro_f1']:.4f} ± {overall['cv_macro_f1_std']:.4f}")
    if not final_test.empty:
        row = final_test[final_test["selection_scope"] == "best_overall"]
        if not row.empty:
            print(f"Final test Macro-F1: {row.iloc[0]['macro_f1_test']:.4f}")
    print("The independent test set was excluded from tuning and selection.")
    print("=" * 72)
    return all_results

def main() -> None:
    parser = argparse.ArgumentParser(description="Train multiclass soil-activity classifiers with CV-only selection")
    parser.add_argument("--data", required=True, help="Processed data directory")
    parser.add_argument("--models_dir", required=True, help="Classification models output directory (used exactly as provided)")
    parser.add_argument("--reports_dir", required=True, help="Classification reports output directory (used exactly as provided)")
    parser.add_argument("--params", default="params.yaml", help="Shared params.yaml")
    args = parser.parse_args()
    train_classification(args.data, args.models_dir, args.reports_dir, args.params)


if __name__ == "__main__":
    main()
