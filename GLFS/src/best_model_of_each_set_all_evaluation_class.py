from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# GAP-AWARE CLASSIFICATION MODEL RANKING
# ============================================================

def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _resolve_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize trainclass/evaluator naming variants."""
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
        "train_balanced_accuracy": [
            "train_balanced_accuracy",
            "balanced_accuracy_train",
        ],
        "val_balanced_accuracy": [
            "val_balanced_accuracy",
            "balanced_accuracy_val",
        ],
        "test_balanced_accuracy": [
            "test_balanced_accuracy",
            "balanced_accuracy_test",
        ],
        "train_accuracy": ["train_accuracy", "accuracy_train"],
        "val_accuracy": ["val_accuracy", "accuracy_val"],
        "test_accuracy": ["test_accuracy", "accuracy_test"],
        "train_mcc": ["train_mcc", "mcc_train"],
        "val_mcc": ["val_mcc", "mcc_val"],
        "test_mcc": ["test_mcc", "mcc_test"],
        "val_log_loss": ["val_log_loss", "log_loss_val"],
        "test_log_loss": ["test_log_loss", "log_loss_test"],
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

    df["val_balanced_error"] = 1.0 - df["val_balanced_accuracy"]
    df["test_balanced_error"] = 1.0 - df["test_balanced_accuracy"]

    if "model" not in df.columns:
        raise ValueError("Input CSV does not contain a model column.")
    if "feature_set" not in df.columns:
        raise ValueError("Input CSV does not contain fs/feature_set.")
    if "target" not in df.columns:
        df["target"] = "unspecified_target"

    return df


def add_generalization_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate classification generalization gaps and diagnoses."""
    df = _resolve_schema(df)

    df["train_cv_gap"] = df["train_macro_f1"] - df["cv_macro_f1"]
    df["train_val_gap"] = df["train_macro_f1"] - df["val_macro_f1"]
    df["cv_val_gap"] = df["cv_macro_f1"] - df["val_macro_f1"]
    df["val_test_gap"] = df["val_macro_f1"] - df["test_macro_f1"]

    df["abs_train_cv_gap"] = df["train_cv_gap"].abs()
    df["abs_cv_val_gap"] = df["cv_val_gap"].abs()
    df["abs_val_test_gap"] = df["val_test_gap"].abs()

    df["possible_underfitting"] = (
        (df["train_macro_f1"] < 0.55)
        & (df["cv_macro_f1"] < 0.50)
        & (df["val_macro_f1"] < 0.50)
    )
    df["possible_overfitting"] = df["train_cv_gap"] >= 0.15
    df["high_cv_instability"] = df["cv_macro_f1_std"] >= 0.15
    df["split_sensitive"] = df["abs_cv_val_gap"] >= 0.15
    df["stable_generalization"] = (
        (df["abs_train_cv_gap"] <= 0.08)
        & (df["cv_macro_f1"] >= 0.50)
        & (~df["high_cv_instability"])
    )

    diagnoses: list[str] = []
    for _, row in df.iterrows():
        if bool(row["possible_underfitting"]):
            diagnosis = "Possible underfitting"
        elif bool(row["possible_overfitting"]) and bool(row["high_cv_instability"]):
            diagnosis = "Overfitting with high CV instability"
        elif bool(row["possible_overfitting"]):
            diagnosis = "Possible overfitting"
        elif bool(row["high_cv_instability"]):
            diagnosis = "High CV instability"
        elif bool(row["split_sensitive"]):
            diagnosis = "Split-sensitive"
        elif bool(row["stable_generalization"]):
            diagnosis = "Stable generalization"
        else:
            diagnosis = "Moderate generalization"
        diagnoses.append(diagnosis)

    df["generalization_diagnosis"] = diagnoses
    return df


def _rank(series: pd.Series, ascending: bool) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").rank(
        ascending=ascending,
        method="average",
        na_option="bottom",
    )


def add_scientific_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """Create separate selection, test, and generalization rankings.

    Selection Rank:
        Uses only training-CV and validation information.
        Test metrics are excluded.

    Test Performance Rank:
        Descriptive final holdout comparison only.

    Generalization Rank:
        Uses train-CV gap, CV variability, and CV-validation gap.
    """
    df = add_generalization_diagnostics(df)

    # Primary scientific classifier-selection criteria.
    df["rank_cv_macro_f1"] = _rank(df["cv_macro_f1"], ascending=False)
    df["rank_val_macro_f1"] = _rank(df["val_macro_f1"], ascending=False)
    df["rank_cv_stability"] = _rank(df["cv_macro_f1_std"], ascending=True)
    df["rank_train_cv_gap"] = _rank(df["abs_train_cv_gap"], ascending=True)
    df["rank_val_balanced_error"] = _rank(
        df["val_balanced_error"],
        ascending=True,
    )

    df["selection_rank_score"] = (
        0.40 * df["rank_cv_macro_f1"]
        + 0.25 * df["rank_val_macro_f1"]
        + 0.15 * df["rank_cv_stability"]
        + 0.15 * df["rank_train_cv_gap"]
        + 0.05 * df["rank_val_balanced_error"]
    )
    df["Selection Rank"] = df["selection_rank_score"].rank(
        ascending=True,
        method="min",
    ).astype("Int64")

    # Descriptive test performance rank.
    if df["test_macro_f1"].notna().any():
        df["rank_test_macro_f1"] = _rank(
            df["test_macro_f1"],
            ascending=False,
        )
        df["rank_test_balanced_accuracy"] = _rank(
            df["test_balanced_accuracy"],
            ascending=False,
        )
        df["rank_test_mcc"] = _rank(df["test_mcc"], ascending=False)

        df["test_rank_score"] = (
            0.50 * df["rank_test_macro_f1"]
            + 0.30 * df["rank_test_balanced_accuracy"]
            + 0.20 * df["rank_test_mcc"]
        )
        df["Test Performance Rank"] = df["test_rank_score"].rank(
            ascending=True,
            method="min",
        ).astype("Int64")
    else:
        df["Test Performance Rank"] = pd.Series(
            pd.NA,
            index=df.index,
            dtype="Int64",
        )

    # Generalization/stability rank.
    df["rank_generalization_gap"] = _rank(
        df["abs_train_cv_gap"],
        ascending=True,
    )
    df["rank_generalization_cv_sd"] = _rank(
        df["cv_macro_f1_std"],
        ascending=True,
    )
    df["rank_generalization_split_gap"] = _rank(
        df["abs_cv_val_gap"],
        ascending=True,
    )
    df["generalization_rank_score"] = (
        0.45 * df["rank_generalization_gap"]
        + 0.35 * df["rank_generalization_cv_sd"]
        + 0.20 * df["rank_generalization_split_gap"]
    )
    df["Generalization Rank"] = df["generalization_rank_score"].rank(
        ascending=True,
        method="min",
    ).astype("Int64")

    # Agreement of selection criteria.
    criterion_ranks = df[
        [
            "rank_cv_macro_f1",
            "rank_val_macro_f1",
            "rank_cv_stability",
            "rank_train_cv_gap",
            "rank_val_balanced_error",
        ]
    ].to_numpy(dtype=float)

    stability = []
    for row in criterion_ranks:
        mean_rank = np.nanmean(row)
        if not np.isfinite(mean_rank) or mean_rank == 0:
            stability.append(np.nan)
        else:
            stability.append(1.0 / (1.0 + np.nanstd(row) / mean_rank))
    df["selection_rank_stability"] = stability

    return df


def _filter_model_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove copied best*.pkl aliases and summary rows."""
    mask = ~df["model"].astype(str).str.contains("best", case=False, na=False)
    mask &= ~df["feature_set"].astype(str).str.contains(
        "best",
        case=False,
        na=False,
    )
    if "file" in df.columns:
        mask &= ~df["file"].astype(str).str.startswith("best", na=False)
    return df.loc[mask].copy()


def _paper_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
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
        "cv_val_gap",
        "val_test_gap",
        "abs_train_cv_gap",
        "abs_cv_val_gap",
        "generalization_diagnosis",
        "selection_rank_stability",
        "file",
    ]
    return [column for column in preferred if column in df.columns]


def analyze_best_models(
    csv_path: str,
    output_path: str | None = None,
) -> pd.DataFrame:
    """Rank classifiers and identify the best classifier per feature set."""
    csv_path = Path(csv_path)
    output_dir = Path(output_path) if output_path else csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df = _filter_model_rows(_resolve_schema(df))

    if df.empty:
        print(
            "[CLASS_RANK][WARN] No ordinary classifier rows remain after filtering. "
            "Ranking will terminate cleanly."
        )
        return pd.DataFrame()

    ranked = add_scientific_ranks(df).sort_values(
        ["Selection Rank", "cv_macro_f1", "val_macro_f1"],
        ascending=[True, False, False],
        na_position="last",
    ).reset_index(drop=True)

    # Main scientific ranking outputs.
    ranked.to_csv(
        output_dir / "full_classification_scientific_ranking.csv",
        index=False,
        float_format="%.6f",
        encoding="utf-8-sig",
    )
    ranked[_paper_columns(ranked)].to_csv(
        output_dir / "Results_classification_scientific_ranking.csv",
        index=False,
        float_format="%.6f",
        encoding="utf-8-sig",
    )

    selection_columns = [
        "Selection Rank",
        "target",
        "model",
        "feature_set",
        "selection_rank_score",
        "selection_rank_stability",
        "train_macro_f1",
        "cv_macro_f1",
        "cv_macro_f1_std",
        "val_macro_f1",
        "val_balanced_accuracy",
        "train_cv_gap",
        "abs_train_cv_gap",
        "generalization_diagnosis",
        "file",
    ]
    ranked[
        [column for column in selection_columns if column in ranked.columns]
    ].to_csv(
        output_dir / "full_classification_selection_ranking.csv",
        index=False,
        float_format="%.6f",
        encoding="utf-8-sig",
    )

    test_columns = [
        "Test Performance Rank",
        "target",
        "model",
        "feature_set",
        "test_rank_score",
        "test_macro_f1",
        "test_balanced_accuracy",
        "test_accuracy",
        "test_mcc",
        "test_log_loss",
        "file",
    ]
    ranked[
        [column for column in test_columns if column in ranked.columns]
    ].sort_values(
        ["Test Performance Rank", "test_macro_f1"],
        ascending=[True, False],
        na_position="last",
    ).to_csv(
        output_dir / "full_classification_test_performance_ranking.csv",
        index=False,
        float_format="%.6f",
        encoding="utf-8-sig",
    )

    generalization_columns = [
        "Generalization Rank",
        "target",
        "model",
        "feature_set",
        "generalization_rank_score",
        "train_cv_gap",
        "cv_val_gap",
        "val_test_gap",
        "abs_train_cv_gap",
        "abs_cv_val_gap",
        "cv_macro_f1_std",
        "possible_overfitting",
        "possible_underfitting",
        "high_cv_instability",
        "split_sensitive",
        "stable_generalization",
        "generalization_diagnosis",
        "file",
    ]
    ranked[
        [column for column in generalization_columns if column in ranked.columns]
    ].sort_values(
        ["Generalization Rank", "abs_train_cv_gap"],
        ascending=[True, True],
        na_position="last",
    ).to_csv(
        output_dir / "full_classification_generalization_ranking.csv",
        index=False,
        float_format="%.6f",
        encoding="utf-8-sig",
    )

    # Per-feature-set outputs and best classifier.
    best_rows = []
    group_columns = ["target", "feature_set"]
    for group_key, group in ranked.groupby(group_columns, dropna=False):
        target, feature_set = group_key
        ordered = group.sort_values(
            ["Selection Rank", "cv_macro_f1", "val_macro_f1"],
            ascending=[True, False, False],
            na_position="last",
        ).reset_index(drop=True)

        safe_target = str(target).replace("/", "_").replace(" ", "_")
        safe_fs = str(feature_set).replace("/", "_").replace(" ", "_")
        ordered.to_csv(
            output_dir / f"ranking_class_{safe_target}_{safe_fs}.csv",
            index=False,
            float_format="%.6f",
            encoding="utf-8-sig",
        )

        best = ordered.iloc[0].copy()
        best_rows.append(best)

    best_per_fs = pd.DataFrame(best_rows).sort_values(
        ["target", "feature_set", "Selection Rank"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    best_per_fs.to_csv(
        output_dir / "best_classifier_per_featureset_selection.csv",
        index=False,
        float_format="%.6f",
        encoding="utf-8-sig",
    )
    best_per_fs[_paper_columns(best_per_fs)].to_csv(
        output_dir / "best_classifier_comparison_gap_aware.csv",
        index=False,
        float_format="%.6f",
        encoding="utf-8-sig",
    )

    methodology = {
        "primary_ranking": "Selection Rank",
        "test_metrics_used_for_selection": False,
        "selection_weights": {
            "cv_macro_f1": 0.40,
            "validation_macro_f1": 0.25,
            "cv_macro_f1_std": 0.15,
            "absolute_train_cv_gap": 0.15,
            "validation_balanced_error": 0.05,
        },
        "test_performance_weights": {
            "test_macro_f1": 0.50,
            "test_balanced_accuracy": 0.30,
            "test_mcc": 0.20,
        },
        "generalization_weights": {
            "absolute_train_cv_gap": 0.45,
            "cv_macro_f1_std": 0.35,
            "absolute_cv_validation_gap": 0.20,
        },
        "class_definitions": sorted(
            {
                (str(row.get("target", "")), str(row.get("classes", "")), str(row.get("class_boundaries", "")))
                for _, row in ranked.iterrows()
            }
        ),
        "input_file": str(csv_path),
        "number_of_models": int(len(ranked)),
        "number_of_feature_sets": int(
            ranked[["target", "feature_set"]].drop_duplicates().shape[0]
        ),
    }

    with (output_dir / "classification_ranking_methodology.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(methodology, handle, indent=2, ensure_ascii=False)

    methodology_text = [
        "CLASSIFICATION MODEL-RANKING METHODOLOGY",
        "=" * 72,
        "",
        "Primary Selection Rank:",
        "- 40% rank of mean training-CV Macro-F1",
        "- 25% rank of validation Macro-F1",
        "- 15% rank of CV Macro-F1 standard deviation",
        "- 15% rank of absolute train-CV Macro-F1 gap",
        "- 5% rank of validation balanced error",
        "",
        "Test metrics are excluded from classifier selection.",
        "",
        "Descriptive Test Performance Rank:",
        "- 50% rank of test Macro-F1",
        "- 30% rank of test balanced accuracy",
        "- 20% rank of test MCC",
        "",
        "Generalization Rank:",
        "- 45% rank of absolute train-CV gap",
        "- 35% rank of CV standard deviation",
        "- 20% rank of absolute CV-validation gap",
        "",
        "Class definitions are read from the evaluation CSV/model metadata.",
    ]
    (output_dir / "classification_ranking_methodology.txt").write_text(
        "\n".join(methodology_text) + "\n",
        encoding="utf-8",
    )

    print("Gap-aware classification ranking completed.")
    print(f"Input: {csv_path}")
    print(f"Output directory: {output_dir}")
    print(
        "Best-per-feature-set file: "
        f"{output_dir / 'best_classifier_per_featureset_selection.csv'}"
    )
    return best_per_fs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rank all classifiers scientifically and identify the "
            "best classifier for each feature set. Test metrics are excluded "
            "from model selection."
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        default="classification_evaluation_ranking.csv",
        help=(
            "All-classifier evaluation CSV, normally generated by "
            "evaluate_all_class2.py."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output directory; defaults to the input CSV directory.",
    )
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(
            f"[CLASS_RANK][WARN] Input file not found: {args.input}. "
            "Ranking will terminate cleanly."
        )
        return

    try:
        analyze_best_models(args.input, args.output)
    except pd.errors.EmptyDataError:
        print(
            "[CLASS_RANK][WARN] Input CSV is empty. "
            "Ranking will terminate cleanly."
        )
    except (ValueError, KeyError) as error:
        print(
            f"[CLASS_RANK][WARN] Classification ranking cannot continue: {error}. "
            "Ranking will terminate cleanly."
        )


if __name__ == "__main__":
    main()
