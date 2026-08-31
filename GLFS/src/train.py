from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    KFold,
    RandomizedSearchCV,
    RepeatedKFold,
    cross_val_score,
    learning_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    SplineTransformer,
    StandardScaler,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

# Support both ``python src/train.py`` and ``python -m src.train``.
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
for path in (SRC_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from train_model_code.registry import (
        load_model_spec,
        validate_requested_models,
    )
except ImportError:  # package execution
    from src.train_model_code.registry import (
        load_model_spec,
        validate_requested_models,
    )

try:
    from metrics_extras import (
        anderson_darling_stat,
        ioa_willmott,
        ios_skill,
        p_within_percent,
    )
except ImportError:
    try:
        from src.metrics_extras import (
            anderson_darling_stat,
            ioa_willmott,
            ios_skill,
            p_within_percent,
        )
    except ImportError:
        # Safe fallbacks keep this training file independently runnable.
        def ioa_willmott(y_true, y_pred):
            y_true = np.asarray(y_true, dtype=float)
            y_pred = np.asarray(y_pred, dtype=float)
            denominator = np.sum(
                (np.abs(y_pred - np.mean(y_true)) + np.abs(y_true - np.mean(y_true))) ** 2
            )
            return np.nan if denominator == 0 else 1 - np.sum((y_pred - y_true) ** 2) / denominator

        def ios_skill(y_true, y_pred):
            y_true = np.asarray(y_true, dtype=float)
            y_pred = np.asarray(y_pred, dtype=float)
            denominator = np.sum((y_true - np.mean(y_true)) ** 2)
            return np.nan if denominator == 0 else 1 - np.sum((y_true - y_pred) ** 2) / denominator

        def p_within_percent(y_true, y_pred, pct=20.0):
            y_true = np.asarray(y_true, dtype=float)
            y_pred = np.asarray(y_pred, dtype=float)
            tolerance = np.maximum(np.abs(y_true) * pct / 100.0, np.finfo(float).eps)
            return 100.0 * np.mean(np.abs(y_pred - y_true) <= tolerance)

        def anderson_darling_stat(residuals):
            return np.nan

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ---------------------------------------------------------------------------
# Step 1 — Small data structures make the workflow explicit and testable.
# ---------------------------------------------------------------------------
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


class NumpyJSONEncoder(json.JSONEncoder):
    """Serialize NumPy and estimator values in report files."""

    def default(self, obj: Any):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (tuple, set)):
            return list(obj)
        return str(obj)


# ---------------------------------------------------------------------------
# Step 2 — Configuration loading and validation.
# ---------------------------------------------------------------------------
def load_params(params_path: str | Path) -> dict[str, Any]:
    path = Path(params_path)
    if not path.exists():
        raise FileNotFoundError(f"Parameters file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        params = yaml.safe_load(handle) or {}

    required = ["TARGETS", "models", "param_grids"]
    missing = [key for key in required if key not in params]
    if missing:
        raise KeyError(f"params.yaml is missing required sections: {missing}")
    if not isinstance(params["models"], list) or not params["models"]:
        raise ValueError("params.yaml 'models' must be a non-empty list.")

    # params.yaml is the single source of truth for ACTIVE models.
    # The registry only provides estimator construction recipes.
    requested_models = []
    seen = set()
    for raw_name in params["models"]:
        name = str(raw_name).strip().upper()
        if not name:
            raise ValueError("params.yaml contains an empty model name.")
        if name not in seen:
            requested_models.append(name)
            seen.add(name)

    validate_requested_models(requested_models)
    params["models"] = requested_models
    return params


def load_split_info(data_dir: Path) -> dict[str, Any]:
    path = data_dir / "split_info.yaml"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def ensure_2d(values: Any) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix, received shape {array.shape}.")
    return array


def save_split_csv(
    output_dir: Path,
    split_name: str,
    target: str,
    feature_set: str,
    feature_names: list[str],
    X: np.ndarray,
    y: np.ndarray,
    source_row_indices: np.ndarray,
) -> Path:
    """Save the exact filtered split used by training/evaluation as CSV.

    The exported file contains a ``source_row_index`` column so rows can be
    traced back to the original prepared Joblib split, followed by feature
    columns and the target column. Files are saved once per target/feature set.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    X = ensure_2d(X)
    y = np.asarray(y, dtype=float).ravel()
    source_row_indices = np.asarray(source_row_indices, dtype=int).ravel()

    if len(feature_names) != X.shape[1]:
        feature_names = [f"feature_{index + 1}" for index in range(X.shape[1])]
    if not (len(X) == len(y) == len(source_row_indices)):
        raise ValueError(
            f"Cannot save {split_name} split for {target}/{feature_set}: "
            "X, y, and row-index lengths differ."
        )

    frame = pd.DataFrame(X, columns=list(feature_names))
    frame.insert(0, "source_row_index", source_row_indices)
    frame[target] = y

    output_path = output_dir / f"{split_name}_{target}_{feature_set}.csv"
    frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def export_train_test_csv_splits(
    paths: OutputPaths,
    target: str,
    feature_set: str,
    feature_names: list[str],
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_source_indices: np.ndarray,
) -> tuple[Path, Path | None]:
    """Export train and test data used by the experiment to separate CSVs."""
    # Save exported partitions with the experiment reports, not beside the
    # input Joblib files. This makes the generated CSVs immediately visible
    # under reports_Ac/Regression (or whichever reports directory is passed).
    csv_dir = paths.reports_dir / "data_splits"
    train_csv = save_split_csv(
        csv_dir, "train", target, feature_set, feature_names,
        X_train, y_train, train_source_indices,
    )

    X_test_path = paths.data_dir / f"X_test_{feature_set}.joblib"
    y_test_path = paths.data_dir / f"y_test_{target}.joblib"
    if not X_test_path.exists() or not y_test_path.exists():
        print(f"[SPLIT][WARN] Test files missing for {target}/{feature_set}; train CSV saved only.")
        return train_csv, None

    X_test_all = ensure_2d(joblib.load(X_test_path))
    y_test_all = np.asarray(joblib.load(y_test_path), dtype=float).ravel()
    if len(X_test_all) != len(y_test_all):
        raise ValueError(
            f"Test row mismatch for {target}/{feature_set}: "
            f"X={len(X_test_all)}, y={len(y_test_all)}"
        )
    test_valid = np.isfinite(y_test_all) & np.all(np.isfinite(X_test_all), axis=1)
    test_source_indices = np.flatnonzero(test_valid)
    test_csv = save_split_csv(
        csv_dir, "test", target, feature_set, feature_names,
        X_test_all[test_valid], y_test_all[test_valid], test_source_indices,
    )
    # Keep a small manifest so the exported partitions and their sizes are
    # easy to audit and cite in a reproducible experiment.
    manifest_path = csv_dir / "split_manifest.csv"
    manifest_row = pd.DataFrame([{
        "target": target,
        "feature_set": feature_set,
        "train_csv": train_csv.name,
        "test_csv": test_csv.name,
        "n_train": len(X_train),
        "n_test": len(X_test_all[test_valid]),
        "n_features": len(feature_names),
        "features": " | ".join(map(str, feature_names)),
    }])
    if manifest_path.exists():
        old_manifest = pd.read_csv(manifest_path)
        old_manifest = old_manifest[
            ~((old_manifest["target"] == target) &
              (old_manifest["feature_set"] == feature_set))
        ]
        manifest_row = pd.concat([old_manifest, manifest_row], ignore_index=True)
    manifest_row.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    return train_csv, test_csv


# ---------------------------------------------------------------------------
# Step 3 — Build preprocessing from params.yaml.
# Everything remains inside Pipeline, so it is fitted separately in each fold.
# ---------------------------------------------------------------------------
def make_scaler(name: str):
    choices = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
        "maxabs": MaxAbsScaler(),
        "none": None,
    }
    if name not in choices:
        raise ValueError(f"Unsupported scaling method: {name}")
    return choices[name]


def build_numeric_steps(config: dict[str, Any]) -> list[tuple[str, Any]]:
    steps: list[tuple[str, Any]] = []

    power = str(config.get("power_transform", "none")).lower()
    power_choices = {
        "none": None,
        "yeo-johnson": PowerTransformer(method="yeo-johnson"),
        "box-cox": PowerTransformer(method="box-cox"),
        "quantile_uniform": QuantileTransformer(output_distribution="uniform"),
        "quantile_normal": QuantileTransformer(output_distribution="normal"),
    }
    if power not in power_choices:
        raise ValueError(f"Unsupported power_transform: {power}")
    if power_choices[power] is not None:
        steps.append(("power_transform", power_choices[power]))

    scaler = make_scaler(str(config.get("scaling", "standard")).lower())
    if scaler is not None:
        steps.append(("scaler", scaler))

    normalization = str(config.get("normalization", "none")).lower()
    if normalization != "none":
        if normalization not in {"l1", "l2", "max"}:
            raise ValueError(f"Unsupported normalization: {normalization}")
        steps.append(("normalizer", Normalizer(norm=normalization)))

    if bool(config.get("polynomial_features", False)):
        steps.append((
            "polynomial",
            PolynomialFeatures(
                degree=int(config.get("poly_degree", 2)),
                interaction_only=bool(config.get("poly_interaction_only", False)),
                include_bias=bool(config.get("poly_include_bias", False)),
            ),
        ))

    if bool(config.get("spline_features", False)):
        steps.append((
            "spline",
            SplineTransformer(
                n_knots=int(config.get("spline_n_knots", 5)),
                degree=int(config.get("spline_degree", 3)),
                extrapolation=str(config.get("spline_extrapolation", "constant")),
            ),
        ))
    return steps


def build_preprocessor(config: dict[str, Any]):
    categorical_columns = list(config.get("categorical_columns", []) or [])
    numeric_columns = list(config.get("numeric_columns", []) or [])
    encoding = config.get("categorical_encoding")

    if categorical_columns and encoding:
        if not numeric_columns:
            raise ValueError(
                "numeric_columns must be provided when categorical_columns are configured."
            )
        numeric_pipeline = Pipeline(build_numeric_steps(config))
        if str(encoding).lower() == "onehot":
            encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
                drop="first" if config.get("drop_first", False) else None,
            )
        elif str(encoding).lower() == "ordinal":
            encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        else:
            raise ValueError(f"Unsupported categorical_encoding: {encoding}")

        return ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipeline, numeric_columns),
                ("categorical", encoder, categorical_columns),
            ],
            remainder="drop" if config.get("drop_remainder", True) else "passthrough",
        )

    steps = build_numeric_steps(config)
    return Pipeline(steps) if steps else "passthrough"


# ---------------------------------------------------------------------------
# Step 4 — Read and sanitize each model's grid from params.yaml.
# ---------------------------------------------------------------------------
def convert_scalar(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    lowered = value.strip().lower()
    if lowered in {"none", "null"}:
        return None
    if lowered in {"true", "yes"}:
        return True
    if lowered in {"false", "no"}:
        return False
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def sanitize_grid(model_name: str, raw_grid: dict[str, Any]) -> tuple[dict[str, list[Any]], int, int]:
    grid = dict(raw_grid or {})
    cv = int(grid.pop("_cv_", 5))
    n_jobs = int(grid.pop("_n_jobs_", -1))
    clean: dict[str, list[Any]] = {}

    for parameter, candidates in grid.items():
        candidates = candidates if isinstance(candidates, list) else [candidates]
        if model_name == "ANN" and parameter == "hidden_layer_sizes":
            clean[parameter] = [
                tuple(candidate) if isinstance(candidate, list) else candidate
                for candidate in candidates
            ]
        else:
            clean[parameter] = [convert_scalar(candidate) for candidate in candidates]
    return clean, cv, n_jobs


def prefix_model_parameters(grid: dict[str, list[Any]]) -> dict[str, list[Any]]:
    return {f"model__{name}": values for name, values in grid.items()}


def unprefix_parameters(params: dict[str, Any]) -> dict[str, Any]:
    return {name.removeprefix("model__"): value for name, value in params.items()}


def count_grid_combinations(grid: dict[str, Iterable[Any]]) -> int:
    total = 1
    for values in grid.values():
        total *= max(1, len(list(values)))
    return total


# ---------------------------------------------------------------------------
# Step 5 — Cross-validation planning.
# ---------------------------------------------------------------------------
def make_cv_plan(
    n_samples: int,
    requested_folds: int,
    repeats: int,
    random_state: int,
    groups: np.ndarray | None,
    model_cv_override: int | None = None,
) -> CVPlan:
    folds = max(2, min(model_cv_override or requested_folds, n_samples))

    if groups is not None and len(np.unique(groups)) >= folds:
        group_cv = GroupKFold(n_splits=folds)
        return CVPlan(group_cv, group_cv, groups, "group_kfold", folds, 1)

    tuning = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    stability = RepeatedKFold(
        n_splits=folds,
        n_repeats=max(1, repeats),
        random_state=random_state,
    )
    return CVPlan(tuning, stability, None, "repeated_kfold", folds, max(1, repeats))


def load_groups(data_dir: Path, target: str, valid_mask: np.ndarray) -> np.ndarray | None:
    for candidate in (
        data_dir / f"groups_train_{target}.joblib",
        data_dir / "groups_train.joblib",
    ):
        if candidate.exists():
            groups = np.asarray(joblib.load(candidate)).ravel()
            if len(groups) == len(valid_mask):
                return groups[valid_mask]
            print(f"[WARN] Ignoring {candidate.name}: incompatible number of rows.")
    return None


def load_aligned_train_metadata(
    data_dir: Path,
    target: str,
    valid_mask: np.ndarray,
) -> pd.DataFrame | None:
    """Load optional row-aligned metadata for fold diagnostics.

    The function intentionally accepts metadata only when its row count exactly
    matches the unfiltered training arrays.  This prevents accidental joining
    of Project/Location labels to the wrong observations.  Supported files are
    CSV or Joblib DataFrames/dictionaries with conventional names.
    """
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
            print(f"[WARN] Could not read optional metadata {candidate.name}: {error}")
            continue

        if len(frame) != len(valid_mask):
            print(
                f"[WARN] Ignoring {candidate.name}: metadata rows={len(frame)} "
                f"but training rows={len(valid_mask)}."
            )
            continue
        frame = frame.loc[np.asarray(valid_mask, dtype=bool)].reset_index(drop=True)
        print(f"[GROUP DIAGNOSTICS] Using aligned metadata: {candidate.name}")
        return frame
    return None


def _first_existing_column(frame: pd.DataFrame | None, names: list[str]) -> str | None:
    if frame is None:
        return None
    lower = {str(column).lower(): str(column) for column in frame.columns}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def build_outer_group_diagnostics(
    *,
    fold_index: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_values: np.ndarray | None,
    metadata: pd.DataFrame | None,
    validation_idx: np.ndarray,
) -> list[dict[str, Any]]:
    """Return one diagnostic row per held-out group in an outer fold."""
    if group_values is None:
        return []

    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    validation_groups = np.asarray(group_values)[validation_idx]
    validation_meta = metadata.iloc[validation_idx].reset_index(drop=True) if metadata is not None else None
    project_col = _first_existing_column(validation_meta, ["Project", "project", "Project_Name", "project_name"])
    location_col = _first_existing_column(validation_meta, ["Location_No", "location_no", "Location", "location"])

    rows: list[dict[str, Any]] = []
    for group in np.unique(validation_groups):
        mask = validation_groups == group
        yt = y_true[mask]
        yp = y_pred[mask]
        mse = mean_squared_error(yt, yp)
        row: dict[str, Any] = {
            "outer_fold": fold_index,
            "group": str(group),
            "n_samples": int(mask.sum()),
            "target_mean": float(np.mean(yt)),
            "target_std": float(np.std(yt, ddof=1)) if len(yt) > 1 else 0.0,
            "target_min": float(np.min(yt)),
            "target_max": float(np.max(yt)),
            "prediction_mean": float(np.mean(yp)),
            "mae": float(mean_absolute_error(yt, yp)),
            "rmse": float(np.sqrt(mse)),
            "bias": float(np.mean(yp - yt)),
            "r2": float(r2_score(yt, yp)) if len(yt) >= 2 else np.nan,
        }
        if validation_meta is not None:
            meta_group = validation_meta.loc[mask]
            if project_col is not None:
                projects = sorted({str(v) for v in meta_group[project_col].dropna().unique()})
                row["project"] = " | ".join(projects)
            if location_col is not None:
                locations = sorted({str(v) for v in meta_group[location_col].dropna().unique()})
                row["location_no"] = " | ".join(locations)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Step 6 — Metrics and transparent generalization diagnosis.
# ---------------------------------------------------------------------------
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mse_value = mean_squared_error(y_true, y_pred)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse_value),
        "rmse": float(np.sqrt(mse_value)),
        "ioa": float(ioa_willmott(y_true, y_pred)),
        "ios": float(ios_skill(y_true, y_pred)),
        "p20": float(p_within_percent(y_true, y_pred, pct=20.0)),
        "bias": float(np.mean(y_pred - y_true)),
        "ad_stat": float(anderson_darling_stat(y_true - y_pred)),
    }


def diagnose_generalization(train_r2: float, cv_mean: float, cv_std: float) -> dict[str, Any]:
    """Describe generalization without equating a group-CV gap with overfitting.

    In spatial/grouped validation, a large train--CV gap can reflect ordinary
    model overfitting, genuine between-location heterogeneity, or both.  The
    report therefore uses neutral, evidence-based wording and keeps the raw
    gap/variability values available for interpretation.
    """
    gap = train_r2 - cv_mean
    absolute_gap = abs(gap)
    underfit = train_r2 < 0.40 and cv_mean < 0.40
    large_gap = gap >= 0.20
    unstable = cv_std >= 0.20
    stable = absolute_gap <= 0.10 and cv_mean >= 0.40 and not unstable

    if underfit:
        diagnosis = "Low train and grouped-CV performance"
    elif large_gap and unstable:
        diagnosis = "Large train-CV generalization gap with high between-group variability"
    elif large_gap:
        diagnosis = "Large train-CV generalization gap"
    elif unstable:
        diagnosis = "High between-group CV variability"
    elif stable:
        diagnosis = "Stable grouped generalization"
    else:
        diagnosis = "Moderate grouped generalization"

    return {
        "train_cv_gap": gap,
        "abs_train_cv_gap": absolute_gap,
        # Backward-compatible columns retained for downstream scripts.
        # These are flags only; grouped CV does not prove causal overfitting.
        "possible_overfitting": large_gap,
        "possible_underfitting": underfit,
        "large_generalization_gap": large_gap,
        "high_cv_instability": unstable,
        "high_between_group_variability": unstable,
        "stable_generalization": stable,
        "generalization_diagnosis": diagnosis,
    }


# ---------------------------------------------------------------------------
# Step 7 — Construct and tune one model.
# ---------------------------------------------------------------------------
def create_search(
    model_name: str,
    params: dict[str, Any],
    preprocessor: Any,
    cv_plan: CVPlan,
    random_state: int,
):
    spec = load_model_spec(model_name)
    estimator = spec.build_estimator(random_state)
    if estimator is None:
        dependency = f" Install '{spec.optional_dependency}'." if spec.optional_dependency else ""
        raise ImportError(f"{model_name} is unavailable.{dependency}")

    pipeline = Pipeline([
        ("preprocessing", clone(preprocessor) if preprocessor != "passthrough" else "passthrough"),
        ("model", estimator),
    ])

    raw_grid = (params.get("param_grids", {}) or {}).get(model_name, {})
    grid, _, configured_n_jobs = sanitize_grid(model_name, raw_grid)
    prefixed = prefix_model_parameters(grid)
    if not prefixed:
        return pipeline, False, spec.explanation

    strategy = params.get("training_strategy", {}) or {}
    quick_mode = bool(strategy.get("quick_mode", False))
    combinations = count_grid_combinations(prefixed)
    max_combinations = 20 if quick_mode else 50

    # Large grids use a reproducible random subset; small grids are exhaustive.
    if combinations > max_combinations:
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=prefixed,
            n_iter=min(max_combinations, combinations),
            scoring="r2",
            cv=cv_plan.tuning_cv,
            n_jobs=configured_n_jobs,
            refit=True,
            random_state=random_state,
            return_train_score=True,
            error_score=np.nan,
        )
    else:
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=prefixed,
            scoring="r2",
            cv=cv_plan.tuning_cv,
            n_jobs=configured_n_jobs,
            refit=True,
            return_train_score=True,
            error_score=np.nan,
        )
    return search, True, spec.explanation



def nested_cv_regression(
    model_name: str,
    params: dict[str, Any],
    preprocessor: Any,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None,
    requested_folds: int,
    repeats: int,
    random_state: int,
    model_cv_override: int | None,
    metadata: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Estimate generalization with outer CV and tune only inside each outer fold.

    The outer test fold is never used by GridSearchCV/RandomizedSearchCV.
    A separate final search is later fitted on the complete development set.
    """
    outer_plan = make_cv_plan(
        n_samples=len(y),
        requested_folds=requested_folds,
        repeats=repeats,
        random_state=random_state,
        groups=groups,
        model_cv_override=model_cv_override,
    )
    split_args = (X, y, groups) if groups is not None else (X, y)
    outer_splits = list(outer_plan.stability_cv.split(*split_args))

    outer_scores: list[float] = []
    outer_train_scores: list[float] = []
    fold_rows: list[dict[str, Any]] = []
    group_diagnostic_rows: list[dict[str, Any]] = []
    best_params_per_fold: list[dict[str, Any]] = []

    for fold_index, (train_idx, test_idx) in enumerate(outer_splits, start=1):
        X_inner, y_inner = X[train_idx], y[train_idx]
        X_outer, y_outer = X[test_idx], y[test_idx]
        groups_inner = groups[train_idx] if groups is not None else None

        inner_plan = make_cv_plan(
            n_samples=len(y_inner),
            requested_folds=requested_folds,
            repeats=1,
            random_state=random_state + fold_index,
            groups=groups_inner,
            model_cv_override=model_cv_override,
        )
        search, used_grid, _ = create_search(
            model_name, params, preprocessor, inner_plan,
            random_state + fold_index,
        )
        fit_kwargs = (
            {"groups": groups_inner}
            if groups_inner is not None and used_grid
            else {}
        )
        search.fit(X_inner, y_inner, **fit_kwargs)
        estimator = search.best_estimator_ if used_grid else search

        train_predictions = np.asarray(estimator.predict(X_inner), dtype=float).ravel()
        outer_predictions = np.asarray(estimator.predict(X_outer), dtype=float).ravel()
        train_r2 = float(r2_score(y_inner, train_predictions))
        outer_r2 = float(r2_score(y_outer, outer_predictions))
        outer_mae = float(mean_absolute_error(y_outer, outer_predictions))
        outer_rmse = float(np.sqrt(mean_squared_error(y_outer, outer_predictions)))
        outer_bias = float(np.mean(outer_predictions - y_outer))
        outer_train_scores.append(train_r2)
        outer_scores.append(outer_r2)
        fold_params = (
            unprefix_parameters(search.best_params_) if used_grid else {}
        )
        best_params_per_fold.append(fold_params)
        fold_row = {
            "outer_fold": fold_index,
            "n_inner_train": len(train_idx),
            "n_outer_validation": len(test_idx),
            "train_r2": train_r2,
            "outer_r2": outer_r2,
            "outer_mae": outer_mae,
            "outer_rmse": outer_rmse,
            "outer_bias": outer_bias,
            "outer_target_mean": float(np.mean(y_outer)),
            "outer_target_std": float(np.std(y_outer, ddof=1)) if len(y_outer) > 1 else 0.0,
            "outer_target_min": float(np.min(y_outer)),
            "outer_target_max": float(np.max(y_outer)),
            "inner_target_mean": float(np.mean(y_inner)),
            "inner_target_std": float(np.std(y_inner, ddof=1)) if len(y_inner) > 1 else 0.0,
            "train_outer_gap": train_r2 - outer_r2,
            "best_params": json.dumps(fold_params, cls=NumpyJSONEncoder),
        }

        if groups is not None:
            train_group_values = set(np.asarray(groups)[train_idx].tolist())
            validation_group_values = set(
                np.asarray(groups)[test_idx].tolist()
            )
            overlap = train_group_values.intersection(
                validation_group_values
            )
            if overlap:
                raise RuntimeError(
                    f"Group leakage detected in outer fold {fold_index}: "
                    f"{sorted(overlap)}"
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
            build_outer_group_diagnostics(
                fold_index=fold_index,
                y_true=y_outer,
                y_pred=outer_predictions,
                group_values=groups,
                metadata=metadata,
                validation_idx=test_idx,
            )
        )
        fold_rows.append(fold_row)

    outer_array = np.asarray(outer_scores, dtype=float)
    train_array = np.asarray(outer_train_scores, dtype=float)
    return {
        "outer_scores": outer_array,
        "outer_train_scores": train_array,
        "outer_r2_mean": float(np.nanmean(outer_array)),
        "outer_r2_std": (
            float(np.nanstd(outer_array, ddof=1))
            if len(outer_array) > 1 else 0.0
        ),
        "outer_train_r2_mean": float(np.nanmean(train_array)),
        "outer_train_r2_std": (
            float(np.nanstd(train_array, ddof=1))
            if len(train_array) > 1 else 0.0
        ),
        "fold_rows": fold_rows,
        "group_diagnostic_rows": group_diagnostic_rows,
        "best_params_per_fold": best_params_per_fold,
        "outer_mode": outer_plan.mode,
        "outer_folds": outer_plan.folds,
        "outer_repeats": outer_plan.repeats,
    }


def save_permutation_sensitivity_regression(
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
    """Compute raw-feature permutation sensitivity using R² score decrease.

    Permutation is applied to the original input columns while the fitted
    pipeline performs all preprocessing internally. This keeps reported
    sensitivity aligned with the named feature set and avoids leakage.
    """
    output_dir = reports_dir / "permutation_sensitivity" / target / feature_set
    output_dir.mkdir(parents=True, exist_ok=True)
    result = permutation_importance(
        estimator, X, y, scoring="r2", n_repeats=max(1, int(n_repeats)),
        random_state=random_state, n_jobs=n_jobs,
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
        "scoring": "r2",
        "n_repeats": int(n_repeats),
    }).sort_values("absolute_importance", ascending=False).reset_index(drop=True)
    path = output_dir / f"{model_name}_permutation_sensitivity.csv"
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return {
        "performed": True,
        "scoring": "r2",
        "n_repeats": int(n_repeats),
        "file": str(path),
        "features": frame.to_dict(orient="records"),
    }


def _grouped_learning_curve_scores(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv_plan: CVPlan,
    train_sizes: list[float],
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute a leakage-safe learning curve by sampling whole groups.

    ``sklearn.model_selection.learning_curve`` interprets fractional
    ``train_sizes`` as numbers of *rows* available in the smallest CV
    training fold. With grouped data this can truncate a fold in the middle
    of a group and can also produce confusing effective final sample sizes.

    This implementation keeps groups intact. For every outer GroupKFold
    split and every requested fraction, it selects a reproducible subset of
    complete training groups, fits a clone of the supplied estimator, and
    evaluates on the untouched validation groups.

    Returns
    -------
    effective_sizes_mean : np.ndarray
        Mean number of training rows actually used across folds.
    effective_sizes_min : np.ndarray
        Minimum number of training rows used across folds.
    train_scores : np.ndarray
        Shape (n_sizes, n_folds).
    cv_scores : np.ndarray
        Shape (n_sizes, n_folds).
    """
    if cv_plan.groups is None:
        raise ValueError("Grouped learning curve requires cv_plan.groups.")

    X = ensure_2d(X)
    y = np.asarray(y, dtype=float).ravel()
    groups = np.asarray(cv_plan.groups).ravel()
    requested = np.asarray(train_sizes, dtype=float)

    if len(X) != len(y) or len(groups) != len(y):
        raise ValueError(
            "Grouped learning curve requires X, y, and groups to have "
            "identical row counts."
        )
    if np.any(requested <= 0) or np.any(requested > 1):
        raise ValueError(
            "Grouped learning-curve train_sizes must be fractions in (0, 1]."
        )

    splits = list(cv_plan.tuning_cv.split(X, y, groups))
    n_sizes = len(requested)
    n_folds = len(splits)

    train_scores = np.full((n_sizes, n_folds), np.nan, dtype=float)
    cv_scores = np.full((n_sizes, n_folds), np.nan, dtype=float)
    effective_sizes = np.zeros((n_sizes, n_folds), dtype=int)

    for fold_index, (train_idx, validation_idx) in enumerate(splits):
        fold_groups = np.asarray(groups[train_idx])
        unique_groups = np.unique(fold_groups)

        # Deterministic but fold-specific ordering prevents always selecting
        # the same alphabetical/numeric groups at small learning-curve sizes.
        rng = np.random.default_rng(random_state + fold_index)
        shuffled_groups = unique_groups.copy()
        rng.shuffle(shuffled_groups)

        for size_index, fraction in enumerate(requested):
            n_groups = max(
                1,
                min(
                    len(shuffled_groups),
                    int(np.ceil(float(fraction) * len(shuffled_groups))),
                ),
            )
            selected_groups = set(shuffled_groups[:n_groups].tolist())
            selected_mask = np.array(
                [group in selected_groups for group in fold_groups],
                dtype=bool,
            )
            subset_idx = train_idx[selected_mask]

            # R² requires at least two observations. Extremely small grouped
            # subsets are marked NaN rather than producing misleading scores.
            if len(subset_idx) < 2 or len(validation_idx) < 2:
                continue

            fitted = clone(estimator)
            fitted.fit(X[subset_idx], y[subset_idx])

            train_pred = fitted.predict(X[subset_idx])
            validation_pred = fitted.predict(X[validation_idx])

            effective_sizes[size_index, fold_index] = len(subset_idx)
            train_scores[size_index, fold_index] = r2_score(
                y[subset_idx], train_pred
            )
            cv_scores[size_index, fold_index] = r2_score(
                y[validation_idx], validation_pred
            )

    effective_sizes_mean = np.rint(
        np.nanmean(
            np.where(effective_sizes > 0, effective_sizes, np.nan),
            axis=1,
        )
    ).astype(int)
    effective_sizes_min = np.nanmin(
        np.where(effective_sizes > 0, effective_sizes, np.nan),
        axis=1,
    ).astype(int)

    return (
        effective_sizes_mean,
        effective_sizes_min,
        train_scores,
        cv_scores,
    )


def save_learning_curve(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    cv_plan: CVPlan,
    reports_dir: Path,
    target: str,
    feature_set: str,
    model_name: str,
    train_sizes: list[float],
) -> None:
    """Save leakage-safe learning-curve diagnostics.

    Grouped experiments use complete sampling groups at every training size.
    Non-grouped experiments retain scikit-learn's standard learning_curve.
    """
    folder = reports_dir / "learning_curves" / target / feature_set
    folder.mkdir(parents=True, exist_ok=True)

    requested_sizes = np.asarray(train_sizes, dtype=float)

    if cv_plan.groups is not None:
        (
            sizes_mean,
            sizes_min,
            train_scores,
            cv_scores,
        ) = _grouped_learning_curve_scores(
            estimator=estimator,
            X=X,
            y=y,
            cv_plan=cv_plan,
            train_sizes=list(requested_sizes),
            random_state=42,
        )
        size_fraction = requested_sizes
        learning_curve_mode = "whole_group_subsampling"
    else:
        sizes, train_scores, cv_scores = learning_curve(
            estimator,
            X,
            y,
            groups=None,
            cv=cv_plan.tuning_cv,
            scoring="r2",
            train_sizes=requested_sizes,
            n_jobs=-1,
            shuffle=True,
            random_state=42,
            error_score=np.nan,
        )
        sizes_mean = np.asarray(sizes, dtype=int)
        sizes_min = np.asarray(sizes, dtype=int)
        size_fraction = requested_sizes
        learning_curve_mode = "row_subsampling"

    frame = pd.DataFrame({
        "train_fraction": size_fraction,
        "train_size": sizes_mean,
        "train_size_min_across_folds": sizes_min,
        "train_r2_mean": np.nanmean(train_scores, axis=1),
        "train_r2_sd": np.nanstd(train_scores, axis=1, ddof=1),
        "cv_r2_mean": np.nanmean(cv_scores, axis=1),
        "cv_r2_sd": np.nanstd(cv_scores, axis=1, ddof=1),
        "learning_curve_mode": learning_curve_mode,
        "group_aware": cv_plan.groups is not None,
    })
    frame.to_csv(
        folder / f"{model_name}_learning_curve.csv",
        index=False,
        encoding="utf-8-sig",
    )


def summarize_learning_curves(reports_dir: Path) -> pd.DataFrame:
    """Aggregate model learning-curve CSVs by target and feature set.

    For every ``learning_curves/<target>/<feature_set>`` directory, this writes:
    ``learning_curve_model_summary.csv`` and
    ``learning_curve_interpretation.txt``. A combined table and overview are
    also written directly under ``learning_curves``.
    """
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
                    print(f"[WARN] Could not read learning curve {csv_path}: {error}")
                    continue
                required = {"train_size", "train_r2_mean", "train_r2_sd", "cv_r2_mean", "cv_r2_sd"}
                if frame.empty or not required.issubset(frame.columns):
                    print(f"[WARN] Learning-curve columns missing in {csv_path}")
                    continue
                first, last = frame.iloc[0], frame.iloc[-1]
                gaps = frame["train_r2_mean"] - frame["cv_r2_mean"]
                cv_gain = float(last["cv_r2_mean"] - first["cv_r2_mean"])
                final_gap = float(last["train_r2_mean"] - last["cv_r2_mean"])

                # Evidence that more data may help should depend primarily on
                # whether validation performance is still improving as data
                # increase. A large train-CV gap is an overfitting diagnostic,
                # but it does not logically imply that additional observations
                # cannot help.
                if len(frame) >= 2:
                    recent_cv_gain = float(
                        frame.iloc[-1]["cv_r2_mean"]
                        - frame.iloc[-2]["cv_r2_mean"]
                    )
                else:
                    recent_cv_gain = 0.0

                more_data_signal = bool(
                    cv_gain >= 0.05
                    and recent_cv_gain >= -0.01
                )
                if abs(final_gap) <= 0.05:
                    gap_label = "small"
                elif abs(final_gap) <= 0.10:
                    gap_label = "moderate"
                else:
                    gap_label = "large"
                if cv_gain >= 0.02:
                    trend = "improving"
                elif cv_gain <= -0.02:
                    trend = "declining"
                else:
                    trend = "near plateau"
                model = csv_path.name.removesuffix("_learning_curve.csv")
                item = {
                    "target": target_dir.name,
                    "feature_set": feature_dir.name,
                    "model": model,
                    "initial_train_size": int(first["train_size"]),
                    "final_train_size": int(last["train_size"]),
                    "initial_train_r2": float(first["train_r2_mean"]),
                    "final_train_r2": float(last["train_r2_mean"]),
                    "initial_cv_r2": float(first["cv_r2_mean"]),
                    "final_cv_r2": float(last["cv_r2_mean"]),
                    "cv_r2_gain": cv_gain,
                    "recent_cv_r2_gain": recent_cv_gain,
                    "final_train_cv_gap": final_gap,
                    "max_abs_train_cv_gap": float(np.nanmax(np.abs(gaps))),
                    "initial_train_sd": float(first["train_r2_sd"]),
                    "final_train_sd": float(last["train_r2_sd"]),
                    "train_sd_change": float(last["train_r2_sd"] - first["train_r2_sd"]),
                    "initial_cv_sd": float(first["cv_r2_sd"]),
                    "final_cv_sd": float(last["cv_r2_sd"]),
                    "cv_sd_change": float(last["cv_r2_sd"] - first["cv_r2_sd"]),
                    "validation_trend": trend,
                    "final_gap_level": gap_label,
                    "more_data_likely_helpful": more_data_signal,
                }
                feature_rows.append(item)
                rows.append(item)

            if not feature_rows:
                continue
            summary = pd.DataFrame(feature_rows).sort_values(
                ["final_cv_r2", "final_cv_sd", "max_abs_train_cv_gap"],
                ascending=[False, True, True],
            )
            summary.to_csv(feature_dir / "learning_curve_model_summary.csv", index=False, encoding="utf-8-sig")

            best = summary.iloc[0]
            improving = int((summary["validation_trend"] == "improving").sum())
            small_gap = int((summary["final_gap_level"] == "small").sum())
            lower_cv_sd = int((summary["cv_sd_change"] < 0).sum())
            mean_gain = float(summary["cv_r2_gain"].mean())
            median_gap = float(summary["final_train_cv_gap"].abs().median())
            likely = int(summary["more_data_likely_helpful"].sum())
            lines = [
                f"LEARNING-CURVE INTERPRETATION: {target_dir.name} / {feature_dir.name}",
                "=" * 72,
                f"Models summarized: {len(summary)}",
                f"Best final CV R²: {best['model']} ({best['final_cv_r2']:.3f}; final gap={best['final_train_cv_gap']:.3f}).",
                f"Mean CV R² change from the smallest to largest training size: {mean_gain:+.3f}.",
                f"Models with improving CV performance (gain >= 0.02): {improving}/{len(summary)}.",
                f"Models with a small final train-CV gap (|gap| <= 0.05): {small_gap}/{len(summary)}.",
                f"Models with reduced CV variability at the largest size: {lower_cv_sd}/{len(summary)}.",
                f"Median absolute final train-CV gap: {median_gap:.3f}.",
                f"Models showing a continued more-data signal: {likely}/{len(summary)}.",
                "",
            ]
            if improving >= max(1, len(summary) // 2):
                lines.append("Overall interpretation: validation performance generally improved as the number of training observations/groups increased. This is descriptive evidence that additional independent sampling groups may improve generalization, although the remaining train-CV gap should be evaluated separately.")
            elif median_gap > 0.10:
                lines.append("Overall interpretation: validation gains were limited while the final generalization gap remained comparatively large; additional data and/or stronger regularization should be considered.")
            else:
                lines.append("Overall interpretation: learning curves were broadly stable and close to a plateau, with no strong evidence of severe overfitting at the largest training size.")
            lines.extend([
                "",
                "These are descriptive learning-curve diagnostics, not formal hypothesis tests.",
            ])
            (feature_dir / "learning_curve_interpretation.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    combined = pd.DataFrame(rows)
    if combined.empty:
        return combined
    combined.to_csv(root / "learning_curve_all_models_summary.csv", index=False, encoding="utf-8-sig")
    fs = (
        combined.groupby(["target", "feature_set"], as_index=False)
        .agg(
            models=("model", "count"),
            mean_cv_r2_gain=("cv_r2_gain", "mean"),
            median_final_cv_r2=("final_cv_r2", "median"),
            median_abs_final_gap=("final_train_cv_gap", lambda s: float(np.median(np.abs(s)))),
            mean_final_cv_sd=("final_cv_sd", "mean"),
            models_likely_helped_by_more_data=("more_data_likely_helpful", "sum"),
        )
        .sort_values(["target", "median_final_cv_r2"], ascending=[True, False])
    )
    fs.to_csv(root / "learning_curve_feature_set_summary.csv", index=False, encoding="utf-8-sig")
    overview = ["LEARNING-CURVE OVERVIEW BY FEATURE SET", "=" * 72, ""]
    for row in fs.itertuples(index=False):
        overview.append(
            f"{row.target}/{row.feature_set}: median final CV R²={row.median_final_cv_r2:.3f}, "
            f"mean gain={row.mean_cv_r2_gain:+.3f}, median |gap|={row.median_abs_final_gap:.3f}, "
            f"more-data signal={int(row.models_likely_helped_by_more_data)}/{int(row.models)} models."
        )
    overview.extend(["", "Interpretations are descriptive and should not be reported as p-values or formal significance tests."])
    (root / "learning_curve_feature_set_interpretation.txt").write_text("\n".join(overview) + "\n", encoding="utf-8")
    print(f"[LEARNING CURVES] Summary tables and interpretations saved under: {root}")
    return combined


# ---------------------------------------------------------------------------
# Step 8 — Reporting and CV-only selection.
# ---------------------------------------------------------------------------
def write_development_reports(results: list[dict[str, Any]], reports_dir: Path) -> pd.DataFrame:
    frame = pd.DataFrame(results)
    frame["rank_cv_r2"] = frame["cv_r2_mean"].rank(ascending=False, method="average")
    frame["rank_cv_stability"] = frame["cv_r2_std"].rank(ascending=True, method="average")
    frame["rank_train_cv_gap"] = frame["abs_train_cv_gap"].rank(ascending=True, method="average")
    frame["selection_rank_score"] = (
        0.60 * frame["rank_cv_r2"]
        + 0.20 * frame["rank_cv_stability"]
        + 0.20 * frame["rank_train_cv_gap"]
    )
    frame["Selection Rank"] = frame["selection_rank_score"].rank(
        ascending=True, method="min"
    ).astype("Int64")
    frame = frame.sort_values(
        ["Selection Rank", "cv_r2_mean", "cv_r2_std", "abs_train_cv_gap"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)

    # Existing downstream filenames are retained.
    frame.to_csv(reports_dir / "regression_summary.csv", index=False)
    frame.to_csv(reports_dir / "Regrassion_summary.csv", index=False)
    frame.to_csv(reports_dir / "regression_ranking.csv", index=False)
    frame.to_json(
        reports_dir / "metrics.json",
        orient="records",
        indent=2,
        default_handler=str,
    )
    paper_columns = [
        "Selection Rank", "target", "model", "feature_set", "r2_train",
        "cv_r2_mean", "cv_r2_std", "cv_r2_min", "cv_r2_max",
        "train_cv_gap", "abs_train_cv_gap", "generalization_diagnosis",
        "n_train", "n_features", "file",
    ]
    frame[[column for column in paper_columns if column in frame.columns]].to_csv(
        reports_dir / "Results_regression_cv_selection.csv", index=False, encoding="utf-8-sig"
    )
    return frame


def selection_key(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(row["cv_r2_mean"]),
        -float(row["cv_r2_std"]),
        -float(row["abs_train_cv_gap"]),
    )


def select_models(results: list[dict[str, Any]], models_dir: Path, reports_dir: Path):
    frame = pd.DataFrame(results)
    selected: list[dict[str, Any]] = []

    for feature_set, group in frame.groupby("feature_set"):
        row = group.sort_values(
            ["cv_r2_mean", "cv_r2_std", "abs_train_cv_gap"],
            ascending=[False, True, True],
        ).iloc[0].to_dict()
        row["selection_scope"] = f"best_for_{feature_set}"
        selected.append(row)
        shutil.copy2(row["model_path"], models_dir / f"best_{feature_set}.pkl")

    overall = max(results, key=selection_key).copy()
    overall["selection_scope"] = "best_overall"
    selected.append(overall)
    shutil.copy2(overall["model_path"], models_dir / "best_overall.pkl")
    shutil.copy2(overall["model_path"], models_dir / "best.pkl")

    pd.DataFrame(selected).to_csv(
        reports_dir / "selected_models_before_test.csv", index=False, encoding="utf-8-sig"
    )
    return selected, overall


# ---------------------------------------------------------------------------
# Step 9 — Final independent test evaluation of selected models only.
# ---------------------------------------------------------------------------
def evaluate_selected_models(
    selected: list[dict[str, Any]], paths: OutputPaths
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    prediction_dir = paths.reports_dir / "test_predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    for result in selected:
        package = joblib.load(result["model_path"])
        estimator = package["model"] if isinstance(package, dict) else package
        feature_set = result["feature_set"]
        target = result["target"]
        X_path = paths.data_dir / f"X_test_{feature_set}.joblib"
        y_path = paths.data_dir / f"y_test_{target}.joblib"
        if not X_path.exists() or not y_path.exists():
            print(f"[TEST][WARN] Missing test files for {result['file']}")
            continue

        X_test = ensure_2d(joblib.load(X_path))
        y_test = np.asarray(joblib.load(y_path), dtype=float).ravel()
        mask = np.isfinite(y_test) & np.all(np.isfinite(X_test), axis=1)
        X_test, y_test = X_test[mask], y_test[mask]
        predictions = np.asarray(estimator.predict(X_test), dtype=float).ravel()
        metrics = regression_metrics(y_test, predictions)

        row = {
            "selection_scope": result["selection_scope"],
            "target": target,
            "model": result["model"],
            "feature_set": feature_set,
            "file": result["file"],
            "n_test": len(y_test),
            **{f"{name}_test": value for name, value in metrics.items()},
            "cv_r2_mean": result["cv_r2_mean"],
            "cv_r2_std": result["cv_r2_std"],
            "cv_test_gap": result["cv_r2_mean"] - metrics["r2"],
            "test_used_for_selection": False,
        }
        rows.append(row)
        pd.DataFrame({
            "actual": y_test,
            "predicted": predictions,
            "residual": y_test - predictions,
        }).to_csv(
            prediction_dir / f"{Path(result['file']).stem}_test_predictions.csv",
            index=False,
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(
        paths.reports_dir / "selected_models_final_test_evaluation.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return frame


def _target_is_categorical_for_regression(params: dict[str, Any], target: str, raw_values: Any) -> bool:
    """Return True when a target is configured/detected as existing class labels.

    Regression on arbitrary class codes is skipped by default because it is
    generally not a meaningful continuous-target problem. Set
    ``regression.force_categorical_targets: true`` to override intentionally.
    """
    reg_cfg = params.get("regression", {}) or {}
    if bool(reg_cfg.get("force_categorical_targets", False)):
        return False
    class_cfg = params.get("classification", {}) or {}
    modes = class_cfg.get("target_modes", {}) or {}
    raw_mode = modes.get(target, modes.get(str(target), "auto")) if isinstance(modes, dict) else "auto"
    mode = str(raw_mode or "auto").strip().lower().replace("-", "_")
    if mode in {"categorical", "category", "labels", "label", "direct", "existing_classes", "existing"}:
        return True
    if mode in {"threshold", "thresholds", "continuous", "continuous_threshold", "three_class"}:
        return False
    boundaries = class_cfg.get("class_boundaries", {}) or {}
    candidate = boundaries.get(target, {}) if isinstance(boundaries, dict) else {}
    if isinstance(candidate, dict) and candidate.get("lower") is not None and candidate.get("upper") is not None:
        return False
    series = pd.Series(np.asarray(raw_values).ravel()).dropna()
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        return True
    nunique, n = int(numeric.nunique()), int(len(numeric))
    return nunique <= max(20, int(np.ceil(np.sqrt(max(n, 1))))) and nunique / max(n, 1) <= 0.20


# ---------------------------------------------------------------------------
# Main orchestration — intentionally readable from top to bottom.
# ---------------------------------------------------------------------------
def run(data_dir: str, models_dir: str, reports_dir: str, params_path: str = "params.yaml"):
    paths = OutputPaths(Path(data_dir), Path(models_dir), Path(reports_dir))
    paths.models_dir.mkdir(parents=True, exist_ok=True)
    paths.reports_dir.mkdir(parents=True, exist_ok=True)

    # Step A: load the experiment definition.
    params = load_params(params_path)
    preprocessing_config = params.get("preprocessing", {}) or {}
    preprocessor = build_preprocessor(preprocessing_config)
    evaluation = params.get("evaluation", {}) or {}
    random_state = int(evaluation.get("random_state", 42))
    default_folds = int(evaluation.get("cv_splits", params.get("cv_splits", 5)))
    repeats = int(evaluation.get("repeated_cv_repeats", 10))
    nested_outer_repeats = int(
        evaluation.get("nested_cv_outer_repeats", 1)
    )
    generate_curves = bool(evaluation.get("generate_learning_curves", False))
    generate_permutation = bool(evaluation.get("generate_permutation_sensitivity", True))
    permutation_repeats = int(evaluation.get("permutation_sensitivity_repeats", 30))
    permutation_n_jobs = int(evaluation.get("permutation_sensitivity_n_jobs", -1))
    curve_sizes = evaluation.get("learning_curve_train_sizes", [0.2, 0.4, 0.6, 0.8, 1.0])

    split_info = load_split_info(paths.data_dir)
    if split_info.get("validation_split_used") not in (None, False):
        print("[WARN] split_info.yaml does not confirm the intended train/test-only workflow.")

    all_results: list[dict[str, Any]] = []

    

# Step B: iterate through targets and their feature sets from params.yaml.
    for target, feature_sets in params["TARGETS"].items():
        y_path = paths.data_dir / f"y_train_{target}.joblib"
        if not y_path.exists():
            print(f"[WARN] Missing target file: {y_path}")
            continue
        y_loaded = joblib.load(y_path)
        if _target_is_categorical_for_regression(params, target, y_loaded):
            print(
                f"[REGRESSION][SKIP] Target '{target}' is an existing categorical/class target. "
                "Regression is skipped for this target; classification can use the labels directly. "
                "Set regression.force_categorical_targets: true only if regression on class codes is intentional."
            )
            continue
        y_all = np.asarray(y_loaded, dtype=float).ravel()

        for feature_set, feature_names in feature_sets.items():
            X_path = paths.data_dir / f"X_train_{feature_set}.joblib"
            if not X_path.exists():
                print(f"[WARN] Missing feature file: {X_path}")
                continue

            X_all = ensure_2d(joblib.load(X_path))
            if len(X_all) != len(y_all):
                raise ValueError(
                    f"Row mismatch for {target}/{feature_set}: X={len(X_all)}, y={len(y_all)}"
                )
            valid = np.isfinite(y_all) & np.all(np.isfinite(X_all), axis=1)
            train_source_indices = np.flatnonzero(valid)
            X_train, y_train = X_all[valid], y_all[valid]
            groups = load_groups(paths.data_dir, target, valid)
            metadata = load_aligned_train_metadata(paths.data_dir, target, valid)
            n_samples, n_features = X_train.shape

            if groups is not None:
                unique_groups = np.unique(groups)
                print(
                    f"[GROUP CV] {target}/{feature_set}: "
                    f"{len(unique_groups)} development groups across "
                    f"{n_samples} samples."
                )
            else:
                print(
                    f"[GROUP CV][WARN] {target}/{feature_set}: "
                    "no compatible groups_train.joblib was found; "
                    "row-level CV will be used."
                )

            # Save the exact train/test rows used by this feature set as CSV.
            # Output location: <reports_dir>/data_splits/.
            train_csv, test_csv = export_train_test_csv_splits(
                paths=paths,
                target=target,
                feature_set=feature_set,
                feature_names=list(feature_names),
                X_train=X_train,
                y_train=y_train,
                train_source_indices=train_source_indices,
            )
            print(f"[SPLIT] Train CSV: {train_csv}")
            if test_csv is not None:
                print(f"[SPLIT] Test CSV:  {test_csv}")

            # Step C: train only models explicitly listed in params.yaml.
            for model_name in params["models"]:
                raw_grid = (params.get("param_grids", {}) or {}).get(model_name, {})
                _, model_cv, _ = sanitize_grid(model_name, raw_grid)
                cv_plan = make_cv_plan(
                    n_samples=n_samples,
                    requested_folds=default_folds,
                    repeats=repeats,
                    random_state=random_state,
                    groups=groups,
                    model_cv_override=model_cv,
                )

                print(f"\n[TRAIN] {target}/{feature_set}/{model_name}")
                try:
                    nested = nested_cv_regression(
                        model_name=model_name,
                        params=params,
                        preprocessor=preprocessor,
                        X=X_train,
                        y=y_train,
                        groups=groups,
                        requested_folds=default_folds,
                        repeats=nested_outer_repeats,
                        random_state=random_state,
                        model_cv_override=model_cv,
                        metadata=metadata,
                    )

                    # Refit once on all development data after unbiased outer-CV
                    # estimation. This fitted package is used only after ranking.
                    search, used_grid, explanation = create_search(
                        model_name, params, preprocessor, cv_plan, random_state
                    )
                    fit_kwargs = {
                        "groups": cv_plan.groups
                    } if cv_plan.groups is not None and used_grid else {}
                    search.fit(X_train, y_train, **fit_kwargs)
                except Exception as error:
                    print(f"[TRAIN][WARN] {model_name} failed: {error}")
                    continue

                fitted = search.best_estimator_ if used_grid else search
                best_params = unprefix_parameters(search.best_params_) if used_grid else {}
                train_predictions = fitted.predict(X_train)
                train_metrics = regression_metrics(y_train, train_predictions)

                cv_scores = nested["outer_scores"]
                cv_mean = nested["outer_r2_mean"]
                cv_std = nested["outer_r2_std"]
                nested_train_mean = nested["outer_train_r2_mean"]
                diagnostics = diagnose_generalization(
                    nested_train_mean, cv_mean, cv_std
                )

                nested_dir = (
                    paths.reports_dir / "nested_cv" / target / feature_set
                )
                nested_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(nested["fold_rows"]).to_csv(
                    nested_dir / f"{model_name}_nested_cv_folds.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
                group_diagnostics = pd.DataFrame(nested["group_diagnostic_rows"])
                if not group_diagnostics.empty:
                    group_diagnostics.to_csv(
                        nested_dir / f"{model_name}_nested_cv_group_diagnostics.csv",
                        index=False,
                        encoding="utf-8-sig",
                    )

                # Step D: save a self-describing package, not only a bare estimator.
                model_path = paths.models_dir / f"{model_name}_{feature_set}_{target}.pkl"
                package = {
                    "model": fitted,
                    "feature_names": list(feature_names),
                    "target": target,
                    "feature_set": feature_set,
                    "model_type": model_name,
                    "model_explanation": explanation,
                    "pipeline_contains_preprocessing": True,
                    "preprocessing_config": preprocessing_config,
                    "best_params": best_params,
                    "cv_mode": "nested_" + nested["outer_mode"],
                    "nested_cv": True,
                    "inner_cv_folds": cv_plan.folds,
                    "outer_cv_folds": nested["outer_folds"],
                    "outer_cv_repeats": nested["outer_repeats"],
                    "validation_split_used": False,
                    "selection_policy": "Cross-validation only",
                    "test_policy": "Final holdout only",
                }
                joblib.dump(package, model_path)

                permutation_sensitivity = {"performed": False}
                if generate_permutation:
                    try:
                        permutation_sensitivity = save_permutation_sensitivity_regression(
                            fitted, X_train, y_train, list(feature_names),
                            paths.reports_dir, target, feature_set, model_name,
                            n_repeats=permutation_repeats,
                            random_state=random_state,
                            n_jobs=permutation_n_jobs,
                        )
                    except Exception as error:
                        permutation_sensitivity = {
                            "performed": False, "error": str(error)
                        }
                        print(f"[WARN] Permutation sensitivity failed for {model_name}: {error}")

                result = {
                    "model": model_name,
                    "feature_set": feature_set,
                    "fs": feature_set,
                    "target": target,
                    "file": model_path.name,
                    "model_path": str(model_path),
                    "r2_train": nested_train_mean,
                    "r2_refit_full_train": train_metrics["r2"],
                    "mae_train": train_metrics["mae"],
                    "rmse_train": train_metrics["rmse"],
                    "cv_r2_mean": cv_mean,
                    "cv_r2_std": cv_std,
                    "cv_r2_min": float(np.min(cv_scores)) if cv_scores.size else np.nan,
                    "cv_r2_max": float(np.max(cv_scores)) if cv_scores.size else np.nan,
                    "cv_r2_scores": json.dumps(cv_scores.tolist()),
                    "cv_mode": "nested_" + nested["outer_mode"],
                    "cv_folds": nested["outer_folds"],
                    "cv_repeats": nested["outer_repeats"],
                    "inner_cv_folds": cv_plan.folds,
                    "nested_cv": True,
                    "nested_best_params_per_fold": json.dumps(
                        nested["best_params_per_fold"],
                        cls=NumpyJSONEncoder,
                    ),
                    **diagnostics,
                    "n_train": n_samples,
                    "n_features": n_features,
                    "used_grid": used_grid,
                    "best_params": json.dumps(best_params, cls=NumpyJSONEncoder),
                    "preprocessing": json.dumps(preprocessing_config, cls=NumpyJSONEncoder),
                    "validation_split_used": False,
                    "test_used_for_selection": False,
                    "permutation_sensitivity": permutation_sensitivity,
                }
                all_results.append(result)

                if generate_curves:
                    try:
                        save_learning_curve(
                            fitted, X_train, y_train, cv_plan, paths.reports_dir,
                            target, feature_set, model_name, curve_sizes,
                        )
                    except Exception as error:
                        print(f"[WARN] Learning curve failed for {model_name}: {error}")

                print(
                    f"  Outer-train R²={nested_train_mean:.4f} | "
                    f"Nested outer-CV R²={cv_mean:.4f} ± {cv_std:.4f} | "
                    f"{diagnostics['generalization_diagnosis']}"
                )

    if not all_results:
        raise RuntimeError("No regression model trained successfully.")

    if generate_curves:
        summarize_learning_curves(paths.reports_dir)

    # Step E: rank with development CV only, then select.
    ranking = write_development_reports(all_results, paths.reports_dir)
    rank_lookup = dict(zip(ranking["file"], ranking["Selection Rank"]))
    for result in all_results:
        result["Selection Rank"] = int(rank_lookup[result["file"]])

    selected, overall = select_models(all_results, paths.models_dir, paths.reports_dir)

    # Step F: only now touch the independent test set.
    final_test = evaluate_selected_models(selected, paths)
    best_meta = {
        "model": overall["model"],
        "feature_set": overall["feature_set"],
        "target": overall["target"],
        "file": overall["file"],
        "selection_basis": "Cross-validation only",
        "cv_r2_mean": overall["cv_r2_mean"],
        "cv_r2_std": overall["cv_r2_std"],
        "train_cv_gap": overall["train_cv_gap"],
        "test_used_for_selection": False,
    }
    with (paths.reports_dir / "best_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(best_meta, handle, indent=2, cls=NumpyJSONEncoder)

    print("\n" + "=" * 72)
    print("TRAINING COMPLETED")
    print("=" * 72)
    print(f"Models trained: {len(all_results)}")
    print(f"Best overall: {overall['model']} / {overall['feature_set']}")
    print(f"Nested outer-CV R²: {overall['cv_r2_mean']:.4f} ± {overall['cv_r2_std']:.4f}")
    if not final_test.empty:
        row = final_test[final_test["selection_scope"] == "best_overall"]
        if not row.empty:
            print(f"Final test R²: {row.iloc[0]['r2_test']:.4f}")
    print("The independent test set was excluded from tuning and ranking.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train configured regression models with CV-only model selection."
    )
    parser.add_argument("--data", required=True, help="Prepared data directory")
    parser.add_argument("--models_dir", required=True, help="Output model directory")
    parser.add_argument("--reports_dir", required=True, help="Output report directory")
    parser.add_argument("--params", default="params.yaml", help="Experiment YAML file")
    parser.add_argument(
        "--skip_classification",
        action="store_true",
        help="Compatibility flag; classification is not started by this script.",
    )
    arguments = parser.parse_args()
    run(arguments.data, arguments.models_dir, arguments.reports_dir, arguments.params)
