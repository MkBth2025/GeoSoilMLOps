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
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RepeatedKFold, GroupKFold, cross_val_score

try:
    from metrics_extras import ioa_willmott, ios_skill, p_within_percent, anderson_darling_stat
except ImportError:
    from src.metrics_extras import ioa_willmott, ios_skill, p_within_percent, anderson_darling_stat


def _load_params(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _ensure_2d(X: Any) -> np.ndarray:
    X = np.asarray(X)
    return X.reshape(-1, 1) if X.ndim == 1 else X


def _ensure_1d(y: Any) -> np.ndarray:
    return np.asarray(y, dtype=float).ravel()


def safe_metrics(y_true: Any, y_pred: Any) -> dict:
    y_true, y_pred = _ensure_1d(y_true), _ensure_1d(y_pred)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) < 2:
        return {k: np.nan for k in ["r2", "mae", "mse", "rmse", "ioa", "ios", "p20", "ad_stat", "bias"]}
    mse_value = float(mean_squared_error(y_true, y_pred))
    residual = y_true - y_pred
    result = {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": mse_value,
        "rmse": float(np.sqrt(mse_value)),
        "bias": float(np.mean(y_pred - y_true)),
    }
    for key, fn in [
        ("ioa", lambda: ioa_willmott(y_true, y_pred)),
        ("ios", lambda: ios_skill(y_true, y_pred)),
        ("p20", lambda: p_within_percent(y_true, y_pred, 20.0)),
        ("ad_stat", lambda: anderson_darling_stat(residual)),
    ]:
        try:
            result[key] = float(fn())
        except Exception:
            result[key] = np.nan
    return result


def parse_model_filename(filename: str):
    stem = Path(filename).stem
    if stem.startswith("best"):
        return "best", None, None
    parts = stem.split("_")
    return (parts[0], parts[1], "_".join(parts[2:])) if len(parts) >= 3 else (None, None, None)


def load_model_package(path: Path) -> dict:
    loaded = joblib.load(path)
    if isinstance(loaded, dict):
        model = loaded.get("model")
        if model is None:
            raise ValueError(f"No model in {path.name}")
        return {
            "model": model,
            "model_type": loaded.get("model_type"),
            "feature_set": loaded.get("feature_set"),
            "target": loaded.get("target"),
            "feature_names": list(loaded.get("feature_names") or []),
            "pipeline_contains_preprocessing": bool(loaded.get("pipeline_contains_preprocessing", False)),
            "best_params": loaded.get("best_params", {}),
        }
    return {
        "model": loaded,
        "model_type": type(loaded).__name__,
        "feature_set": None,
        "target": None,
        "feature_names": [],
        "pipeline_contains_preprocessing": hasattr(loaded, "steps"),
        "best_params": {},
    }


def training_cv_scores(model, X_train, y_train, data_dir: Path, target: str, params: dict):
    cfg = params.get("evaluation", {}) or {}
    random_state = int(cfg.get("random_state", 42))
    folds = max(2, min(int(cfg.get("cv_splits", params.get("cv_splits", 5))), len(y_train)))
    repeats = max(1, int(cfg.get("repeated_cv_repeats", 10)))

    groups = None
    for path in (data_dir / f"groups_train_{target}.joblib", data_dir / "groups_train.joblib"):
        if path.exists():
            groups = np.asarray(joblib.load(path)).ravel()
            break

    if groups is not None and len(groups) == len(y_train) and len(np.unique(groups)) >= folds:
        cv, mode, repeats_used = GroupKFold(n_splits=folds), "group", 1
    else:
        groups = None
        cv = RepeatedKFold(n_splits=folds, n_repeats=repeats, random_state=random_state)
        mode, repeats_used = "repeated_kfold", repeats

    scores = cross_val_score(
        clone(model), X_train, y_train, groups=groups, cv=cv,
        scoring="r2", n_jobs=-1, error_score=np.nan
    )
    scores = np.asarray(scores, dtype=float)
    return scores[np.isfinite(scores)], mode, folds, repeats_used


def compute_diagnostics(row: dict) -> dict:
    train = float(row.get("r2_train", np.nan))
    cv = float(row.get("cv_r2_mean", np.nan))
    sd = float(row.get("cv_r2_std", np.nan))
    test = float(row.get("r2_test", np.nan))
    train_cv_gap = train - cv
    cv_test_gap = cv - test
    underfit = np.isfinite(train) and np.isfinite(cv) and train < 0.40 and cv < 0.40
    overfit = np.isfinite(train_cv_gap) and train_cv_gap >= 0.20
    unstable = np.isfinite(sd) and sd >= 0.20
    test_shift = np.isfinite(cv_test_gap) and abs(cv_test_gap) >= 0.20
    stable = abs(train_cv_gap) <= 0.10 and cv >= 0.40 and not unstable
    if underfit:
        diagnosis = "Possible underfitting"
    elif overfit and unstable:
        diagnosis = "Overfitting with high CV instability"
    elif overfit:
        diagnosis = "Possible overfitting"
    elif unstable:
        diagnosis = "High CV instability"
    elif test_shift:
        diagnosis = "Holdout-sensitive"
    elif stable:
        diagnosis = "Stable generalization"
    else:
        diagnosis = "Moderate generalization"
    return {
        "train_cv_gap": train_cv_gap,
        "cv_test_gap": cv_test_gap,
        "abs_train_cv_gap": abs(train_cv_gap),
        "abs_cv_test_gap": abs(cv_test_gap),
        "possible_overfitting": bool(overfit),
        "possible_underfitting": bool(underfit),
        "high_cv_instability": bool(unstable),
        "holdout_sensitive": bool(test_shift),
        "stable_generalization": bool(stable),
        "generalization_diagnosis": diagnosis,
    }


def evaluate_model_file(model_path: Path, data_dir: Path, params: dict):
    package = load_model_package(model_path)
    parsed_model, parsed_fs, parsed_target = parse_model_filename(model_path.name)
    model_name = package["model_type"] or parsed_model or type(package["model"]).__name__
    fs = package["feature_set"] or parsed_fs
    target = package["target"] or parsed_target
    if not fs or not target:
        raise ValueError(f"Cannot determine feature set/target for {model_path.name}")

    paths = {
        "X_train": data_dir / f"X_train_{fs}.joblib",
        "X_test": data_dir / f"X_test_{fs}.joblib",
        "y_train": data_dir / f"y_train_{target}.joblib",
        "y_test": data_dir / f"y_test_{target}.joblib",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing train/test files: " + ", ".join(missing))

    X_train, X_test = _ensure_2d(joblib.load(paths["X_train"])), _ensure_2d(joblib.load(paths["X_test"]))
    y_train, y_test = _ensure_1d(joblib.load(paths["y_train"])), _ensure_1d(joblib.load(paths["y_test"]))
    model = package["model"]

    expected = len(package["feature_names"]) or getattr(model, "n_features_in_", X_train.shape[1])
    if X_train.shape[1] != int(expected) or X_test.shape[1] != int(expected):
        raise ValueError(f"Feature mismatch: model expects {expected}")

    row = {
        "model": model_name, "feature_set": fs, "target": target,
        "file": model_path.name,
        "pipeline_contains_preprocessing": package["pipeline_contains_preprocessing"],
        "n_features": X_train.shape[1], "n_train": len(y_train), "n_test": len(y_test),
        "best_params": json.dumps(package["best_params"], ensure_ascii=False, default=str),
        "validation_split_used": False,
    }
    predictions = []
    for split, X, y in [("train", X_train, y_train), ("test", X_test, y_test)]:
        pred = _ensure_1d(model.predict(X))
        for key, value in safe_metrics(y, pred).items():
            row[f"{key}_{split}"] = value
        for i, (actual, predicted) in enumerate(zip(y, pred)):
            predictions.append({
                "index": i, "y_true": float(actual), "y_pred": float(predicted),
                "residual": float(actual - predicted), "model": model_name,
                "fs": fs, "target": target, "file": model_path.name, "split": split,
            })

    scores, mode, folds, repeats = training_cv_scores(model, X_train, y_train, data_dir, target, params)
    row.update({
        "cv_r2_mean": float(np.mean(scores)) if scores.size else np.nan,
        "cv_r2_std": float(np.std(scores, ddof=1)) if scores.size > 1 else 0.0,
        "cv_r2_min": float(np.min(scores)) if scores.size else np.nan,
        "cv_r2_max": float(np.max(scores)) if scores.size else np.nan,
        "cv_mode": mode, "cv_folds": folds, "cv_repeats": repeats,
    })
    row.update(compute_diagnostics(row))
    return row, predictions


def add_ranks(df: pd.DataFrame) -> pd.DataFrame:
    ranked = df.copy()
    ranked["rank_cv_r2"] = ranked["cv_r2_mean"].rank(ascending=False, method="min")
    ranked["rank_cv_stability"] = ranked["cv_r2_std"].rank(ascending=True, method="min")
    ranked["rank_train_cv_gap"] = ranked["abs_train_cv_gap"].rank(ascending=True, method="min")
    ranked["selection_rank_score"] = (
        0.60 * ranked["rank_cv_r2"]
        + 0.20 * ranked["rank_cv_stability"]
        + 0.20 * ranked["rank_train_cv_gap"]
    )
    ranked["Selection Rank"] = ranked["selection_rank_score"].rank(
        ascending=True, method="min"
    ).astype("Int64")

    ranked["rank_test_r2"] = ranked["r2_test"].rank(ascending=False, method="min")
    ranked["rank_test_rmse"] = ranked["rmse_test"].rank(ascending=True, method="min")
    ranked["rank_test_mae"] = ranked["mae_test"].rank(ascending=True, method="min")
    ranked["test_rank_score"] = (
        0.50 * ranked["rank_test_r2"]
        + 0.30 * ranked["rank_test_rmse"]
        + 0.20 * ranked["rank_test_mae"]
    )
    ranked["Test Performance Rank"] = ranked["test_rank_score"].rank(
        ascending=True, method="min"
    ).astype("Int64")
    return ranked.sort_values(
        ["Selection Rank", "cv_r2_mean", "cv_r2_std"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def auto_select_model_file(models_dir: Path, reports_dir: Path) -> str:
    for name in ("best_overall.pkl", "best.pkl"):
        if (models_dir / name).exists():
            return name
    for report in ("full_model_selection_ranking.csv", "regression_evaluation_ranking.csv", "regression_ranking.csv"):
        path = reports_dir / report
        if path.exists():
            df = pd.read_csv(path)
            if "Selection Rank" in df.columns:
                df = df.sort_values("Selection Rank")
            for value in df.get("file", pd.Series(dtype=str)).dropna().astype(str):
                filename = Path(value).name
                if not filename.lower().startswith("best") and (models_dir / filename).exists():
                    return filename
    ordinary = [p.name for p in models_dir.glob("*.pkl") if not p.name.startswith("best")]
    if len(ordinary) == 1:
        return ordinary[0]
    raise FileNotFoundError("Could not automatically determine the selected model.")


def run(data_dir: str, models_dir: str, reports_dir: str, model_file: str | None = None, params_path: str = "params.yaml") -> None:
    data_dir, models_dir, reports_dir = Path(data_dir), Path(models_dir), Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    params = _load_params(params_path)
    model_file = model_file or auto_select_model_file(models_dir, reports_dir)
    model_path = models_dir / model_file
    row, predictions = evaluate_model_file(model_path, data_dir, params)
    df = add_ranks(pd.DataFrame([row]))
    stem = model_path.stem
    df.to_csv(reports_dir / f"eval_{stem}.csv", index=False)
    df.to_json(reports_dir / f"eval_{stem}.json", orient="records", indent=2)
    pd.DataFrame(predictions).to_csv(reports_dir / f"predictions_{stem}.csv", index=False)
    print(f"[EVAL] Completed CV-only selection / final-test reporting for {model_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a selected regression model using repeated CV and an independent test set.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--models_dir", required=True)
    parser.add_argument("--reports_dir", required=True)
    parser.add_argument("--model_file", default=None)
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()
    run(args.data, args.models_dir, args.reports_dir, args.model_file, args.params)
