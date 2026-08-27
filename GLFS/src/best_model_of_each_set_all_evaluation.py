from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _resolve_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "feature_set" not in df.columns and "fs" in df.columns:
        df["feature_set"] = df["fs"]
    if "fs" not in df.columns and "feature_set" in df.columns:
        df["fs"] = df["feature_set"]
    for column in [
        "r2_train", "cv_r2_mean", "cv_r2_std", "r2_test",
        "mae_test", "rmse_test",
    ]:
        df[column] = _numeric(df, column)
    if "model" not in df.columns or "feature_set" not in df.columns:
        raise ValueError("Input must contain model and feature_set/fs columns.")
    if "target" not in df.columns:
        df["target"] = "unspecified_target"
    return df


def _rank(series: pd.Series, ascending: bool) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").rank(
        ascending=ascending, method="average", na_option="bottom"
    )


def add_scientific_ranks(df: pd.DataFrame) -> pd.DataFrame:
    df = _resolve_schema(df)
    df["train_cv_gap"] = df["r2_train"] - df["cv_r2_mean"]
    df["cv_test_gap"] = df["cv_r2_mean"] - df["r2_test"]
    df["abs_train_cv_gap"] = df["train_cv_gap"].abs()
    df["abs_cv_test_gap"] = df["cv_test_gap"].abs()

    df["possible_underfitting"] = (df["r2_train"] < 0.40) & (df["cv_r2_mean"] < 0.40)
    df["possible_overfitting"] = df["train_cv_gap"] >= 0.20
    df["high_cv_instability"] = df["cv_r2_std"] >= 0.20
    df["holdout_sensitive"] = df["abs_cv_test_gap"] >= 0.20
    df["stable_generalization"] = (
        (df["abs_train_cv_gap"] <= 0.10)
        & (df["cv_r2_mean"] >= 0.40)
        & (~df["high_cv_instability"])
    )

    diagnosis = []
    for _, row in df.iterrows():
        if row["possible_underfitting"]:
            value = "Possible underfitting"
        elif row["possible_overfitting"] and row["high_cv_instability"]:
            value = "Overfitting with high CV instability"
        elif row["possible_overfitting"]:
            value = "Possible overfitting"
        elif row["high_cv_instability"]:
            value = "High CV instability"
        elif row["holdout_sensitive"]:
            value = "Holdout-sensitive"
        elif row["stable_generalization"]:
            value = "Stable generalization"
        else:
            value = "Moderate generalization"
        diagnosis.append(value)
    df["generalization_diagnosis"] = diagnosis

    # Model selection: CV only. No validation or test metrics.
    df["rank_cv_r2"] = _rank(df["cv_r2_mean"], ascending=False)
    df["rank_cv_stability"] = _rank(df["cv_r2_std"], ascending=True)
    df["rank_train_cv_gap"] = _rank(df["abs_train_cv_gap"], ascending=True)
    df["selection_rank_score"] = (
        0.60 * df["rank_cv_r2"]
        + 0.20 * df["rank_cv_stability"]
        + 0.20 * df["rank_train_cv_gap"]
    )
    df["Selection Rank"] = df["selection_rank_score"].rank(
        ascending=True, method="min"
    ).astype("Int64")

    # Test rank is descriptive only.
    df["rank_test_r2"] = _rank(df["r2_test"], ascending=False)
    df["rank_test_rmse"] = _rank(df["rmse_test"], ascending=True)
    df["rank_test_mae"] = _rank(df["mae_test"], ascending=True)
    df["test_rank_score"] = (
        0.50 * df["rank_test_r2"]
        + 0.30 * df["rank_test_rmse"]
        + 0.20 * df["rank_test_mae"]
    )
    df["Test Performance Rank"] = df["test_rank_score"].rank(
        ascending=True, method="min"
    ).astype("Int64")

    df["rank_generalization_gap"] = _rank(df["abs_train_cv_gap"], ascending=True)
    df["rank_generalization_cv_sd"] = _rank(df["cv_r2_std"], ascending=True)
    df["Generalization Rank"] = (
        0.60 * df["rank_generalization_gap"]
        + 0.40 * df["rank_generalization_cv_sd"]
    ).rank(ascending=True, method="min").astype("Int64")
    return df


def _filter_model_rows(df: pd.DataFrame) -> pd.DataFrame:
    mask = ~df["model"].astype(str).str.contains("best", case=False, na=False)
    mask &= ~df["feature_set"].astype(str).str.contains("best", case=False, na=False)
    if "file" in df.columns:
        mask &= ~df["file"].astype(str).str.startswith("best", na=False)
    return df.loc[mask].copy()


def analyze_best_models(csv_path: str, output_path: str | None = None):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {csv_path}")
    output_dir = Path(output_path) if output_path else csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _filter_model_rows(add_scientific_ranks(pd.read_csv(csv_path)))
    full = df.sort_values(
        ["Selection Rank", "cv_r2_mean", "cv_r2_std"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    full.to_csv(output_dir / "full_model_scientific_ranking.csv", index=False)
    full.to_csv(output_dir / "full_model_selection_ranking.csv", index=False)
    full.sort_values("Test Performance Rank").to_csv(
        output_dir / "full_model_test_performance_ranking.csv", index=False
    )
    full.sort_values("Generalization Rank").to_csv(
        output_dir / "full_model_generalization_ranking.csv", index=False
    )

    best_rows = []
    for fs, group in full.groupby("feature_set", sort=True):
        selected = group.sort_values(
            ["Selection Rank", "cv_r2_mean", "cv_r2_std"],
            ascending=[True, False, True],
        ).iloc[0]
        best_rows.append(selected)
        group.sort_values("Selection Rank").to_csv(
            output_dir / f"ranking_{fs}_scientific.csv", index=False
        )

    best = pd.DataFrame(best_rows).sort_values(["Selection Rank", "feature_set"])
    columns = [
        "Selection Rank", "Test Performance Rank", "Generalization Rank",
        "target", "model", "feature_set", "r2_train", "cv_r2_mean",
        "cv_r2_std", "r2_test", "train_cv_gap", "cv_test_gap",
        "rmse_test", "generalization_diagnosis", "file",
    ]
    best[[c for c in columns if c in best.columns]].to_csv(
        output_dir / "best_model_per_featureset_selection.csv",
        index=False, encoding="utf-8-sig"
    )

    (output_dir / "ranking_methodology.txt").write_text(
        "CV-ONLY REGRESSION MODEL SELECTION\n"
        "==================================\n\n"
        "Selection Rank excludes all test metrics:\n"
        "- 60% mean repeated/group cross-validation R2\n"
        "- 20% cross-validation stability\n"
        "- 20% absolute train-CV R2 gap\n\n"
        "The independent test set is evaluated only after selection and its rank is descriptive.\n",
        encoding="utf-8",
    )
    return best, full


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank regression models using CV only.")
    parser.add_argument("--input", "-i", default="all_evaluations.csv")
    parser.add_argument("--output", "-o")
    args = parser.parse_args()
    try:
        analyze_best_models(args.input, args.output)
    except Exception as error:
        print(f"Error: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
