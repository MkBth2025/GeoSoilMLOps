#python src/extract_permutation_sensitivity.py  --regression-root reports_Ac/permutation_sensitivity/AC  --classification-root reports_Ac_class/permutation_sensitivity/AC  --output reports_Acpermutation_reports/permutation_all.csv  --summary-output reports_Ac/permutation_reports/permutation_summary.csv
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


MODEL_FILE_RE = re.compile(
    r"^(?P<model>.+?)_permutation_sensitivity\.csv$",
    flags=re.IGNORECASE,
)


def natural_key(value: str) -> tuple:
    """Sort strings such as fs1, fs2, ..., fs10 naturally."""
    parts = re.split(r"(\d+)", str(value))
    return tuple(int(p) if p.isdigit() else p.lower() for p in parts)


def clean_column_name(name: object) -> str:
    """Convert a CSV column name to a consistent snake_case form."""
    text = str(name).strip()
    text = re.sub(r"[%(){}\[\]/\\\-]+", "_", text)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_").lower()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize common permutation-importance column names."""
    df = df.copy()
    df.columns = [clean_column_name(c) for c in df.columns]

    aliases = {
        # Feature name
        "features": "feature",
        "feature_name": "feature",
        "variable": "feature",
        "predictor": "feature",
        "input_feature": "feature",

        # Mean importance
        "importance": "mean_importance",
        "importance_mean": "mean_importance",
        "mean_permutation_importance": "mean_importance",
        "permutation_importance_mean": "mean_importance",
        "mean_importance_score": "mean_importance",

        # Standard deviation
        "std": "importance_std",
        "sd": "importance_std",
        "importance_sd": "importance_std",
        "importance_stddev": "importance_std",
        "std_importance": "importance_std",
        "permutation_importance_std": "importance_std",
        "standard_deviation": "importance_std",

        # Relative contribution
        "relative_importance": "relative_contribution_pct",
        "relative_contribution": "relative_contribution_pct",
        "relative_contribution_percent": "relative_contribution_pct",
        "relative_contribution_percentage": "relative_contribution_pct",
        "contribution_pct": "relative_contribution_pct",
        "contribution_percent": "relative_contribution_pct",
    }

    rename_map = {
        col: aliases[col]
        for col in df.columns
        if col in aliases
    }
    return df.rename(columns=rename_map)


def detect_numeric_importance_column(df: pd.DataFrame) -> str | None:
    """
    Detect the best available importance column if it was not recognized
    through the standard aliases.
    """
    preferred = [
        "mean_importance",
        "importance_mean",
        "importance",
        "score",
        "mean_score",
    ]
    for col in preferred:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            return col

    numeric_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and c not in {
            "importance_std",
            "relative_contribution_pct",
            "rank",
        }
    ]

    importance_like = [
        c for c in numeric_cols
        if "importance" in c or "score" in c
    ]
    if importance_like:
        return importance_like[0]

    return numeric_cols[0] if numeric_cols else None


def make_relative_contribution(
    df: pd.DataFrame,
    importance_col: str,
) -> pd.Series:
    """
    Calculate non-negative percentage contribution within one source file.

    Negative permutation importance can occur. For contribution percentages,
    only positive importance is used. If all values are non-positive, the
    absolute importance values are used so that a descriptive percentage can
    still be reported.
    """
    values = pd.to_numeric(df[importance_col], errors="coerce")

    positive = values.clip(lower=0)
    denominator = positive.sum(skipna=True)

    if denominator > 0:
        return positive / denominator * 100.0

    absolute = values.abs()
    denominator = absolute.sum(skipna=True)

    if denominator > 0:
        return absolute / denominator * 100.0

    return pd.Series(np.nan, index=df.index, dtype=float)


def infer_feature_set(path: Path) -> str:
    """Find the nearest parent folder named fs<number>."""
    for parent in [path.parent, *path.parents]:
        if re.fullmatch(r"fs\d+", parent.name, flags=re.IGNORECASE):
            return parent.name.upper()
    return "UNKNOWN"


def infer_ac_group(path: Path) -> str:
    """Capture the parent group immediately above fsX, usually AC."""
    parents = list(path.parents)
    for i, parent in enumerate(parents):
        if re.fullmatch(r"fs\d+", parent.name, flags=re.IGNORECASE):
            if i + 1 < len(parents):
                return parents[i + 1].name
    return "UNKNOWN"


def iter_permutation_files(root: Path) -> Iterable[Path]:
    """Yield permutation sensitivity CSV files recursively."""
    if not root.exists():
        return []

    files = [
        p for p in root.rglob("*.csv")
        if MODEL_FILE_RE.match(p.name)
    ]
    return sorted(files, key=lambda p: natural_key(str(p)))


def read_one_file(
    csv_path: Path,
    task: str,
    experiment_family: str,
) -> pd.DataFrame:
    """Read one model CSV and attach source metadata."""
    match = MODEL_FILE_RE.match(csv_path.name)
    if match is None:
        raise ValueError(f"Unexpected filename: {csv_path.name}")

    model = match.group("model").strip().upper()
    feature_set = infer_feature_set(csv_path)
    ac_group = infer_ac_group(csv_path)

    df = pd.read_csv(csv_path)
    if df.empty:
        return pd.DataFrame()

    df = normalize_columns(df)

    # Ensure a feature-name column exists.
    if "feature" not in df.columns:
        object_cols = [
            c for c in df.columns
            if not pd.api.types.is_numeric_dtype(df[c])
        ]
        if object_cols:
            df = df.rename(columns={object_cols[0]: "feature"})
        else:
            df.insert(0, "feature", [f"row_{i + 1}" for i in range(len(df))])

    # Ensure a canonical mean importance column exists.
    importance_col = detect_numeric_importance_column(df)
    if importance_col is not None and importance_col != "mean_importance":
        df = df.rename(columns={importance_col: "mean_importance"})

    if "mean_importance" in df.columns:
        df["mean_importance"] = pd.to_numeric(
            df["mean_importance"], errors="coerce"
        )

    if "importance_std" in df.columns:
        df["importance_std"] = pd.to_numeric(
            df["importance_std"], errors="coerce"
        )

    # Generate relative contribution if absent.
    if (
        "relative_contribution_pct" not in df.columns
        and "mean_importance" in df.columns
    ):
        df["relative_contribution_pct"] = make_relative_contribution(
            df, "mean_importance"
        )
    elif "relative_contribution_pct" in df.columns:
        df["relative_contribution_pct"] = pd.to_numeric(
            df["relative_contribution_pct"], errors="coerce"
        )

        # Convert fractions such as 0.35 to percentages if appropriate.
        finite = df["relative_contribution_pct"].dropna()
        if not finite.empty and finite.abs().max() <= 1.0:
            df["relative_contribution_pct"] *= 100.0

    # Add a within-file importance rank.
    if "mean_importance" in df.columns:
        df["importance_rank_within_model"] = (
            df["mean_importance"]
            .rank(method="dense", ascending=False)
            .astype("Int64")
        )

    metadata = {
        "task": task,
        "experiment_family": experiment_family,
        "ac_group": ac_group,
        "feature_set": feature_set,
        "model": model,
        "source_file": csv_path.name,
        "source_path": str(csv_path),
    }

    # Insert metadata at the beginning.
    for key, value in reversed(list(metadata.items())):
        df.insert(0, key, value)

    return df


def collect_folder(
    root: Path,
    task: str,
    experiment_family: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Collect all valid model files below one root directory."""
    frames: list[pd.DataFrame] = []
    warnings: list[str] = []

    files = list(iter_permutation_files(root))
    if not files:
        warnings.append(f"No permutation CSV files found under: {root}")
        return pd.DataFrame(), warnings

    for csv_path in files:
        try:
            frame = read_one_file(
                csv_path=csv_path,
                task=task,
                experiment_family=experiment_family,
            )
            if frame.empty:
                warnings.append(f"Skipped empty file: {csv_path}")
            else:
                frames.append(frame)
        except Exception as exc:
            warnings.append(f"Failed to read {csv_path}: {exc}")

    if not frames:
        return pd.DataFrame(), warnings

    combined = pd.concat(frames, ignore_index=True, sort=False)
    return combined, warnings


def build_summary(combined: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate permutation importance across models for each task/family/
    feature set/feature.
    """
    required = {
        "task",
        "experiment_family",
        "feature_set",
        "feature",
        "mean_importance",
    }
    if not required.issubset(combined.columns):
        return pd.DataFrame()

    data = combined.copy()
    data["mean_importance"] = pd.to_numeric(
        data["mean_importance"], errors="coerce"
    )

    group_cols = [
        "task",
        "experiment_family",
        "ac_group",
        "feature_set",
        "feature",
    ]

    summary = (
        data.groupby(group_cols, dropna=False)
        .agg(
            number_of_models=("model", "nunique"),
            number_of_records=("mean_importance", "count"),
            mean_importance=("mean_importance", "mean"),
            median_importance=("mean_importance", "median"),
            importance_std_across_models=("mean_importance", "std"),
            minimum_importance=("mean_importance", "min"),
            maximum_importance=("mean_importance", "max"),
        )
        .reset_index()
    )

    # Relative contribution based on aggregated positive mean importance
    # inside each task/family/feature-set group.
    summary["relative_contribution_pct"] = (
        summary.groupby(
            ["task", "experiment_family", "ac_group", "feature_set"],
            group_keys=False,
        )["mean_importance"]
        .apply(
            lambda s: (
                s.clip(lower=0) / s.clip(lower=0).sum() * 100.0
                if s.clip(lower=0).sum() > 0
                else s.abs() / s.abs().sum() * 100.0
                if s.abs().sum() > 0
                else pd.Series(np.nan, index=s.index)
            )
        )
    )

    summary["importance_rank"] = (
        summary.groupby(
            ["task", "experiment_family", "ac_group", "feature_set"]
        )["mean_importance"]
        .rank(method="dense", ascending=False)
        .astype("Int64")
    )

    return summary.sort_values(
        by=[
            "task",
            "experiment_family",
            "feature_set",
            "importance_rank",
            "feature",
        ],
        key=lambda s: s.map(natural_key) if s.name == "feature_set" else s,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Combine regression and classification permutation-sensitivity "
            "CSV files into comprehensive CSV outputs."
        )
    )

    parser.add_argument(
        "--regression-root",
        type=Path,
        default=Path(
            "reports_Ac_/permutation_sensitivity/AC"
        ),
        help=(
            "Root containing regression fs1, fs2, fs3, and fs4 folders. "
            "Default: reports_Ac_/permutation_sensitivity/AC"
        ),
    )
    parser.add_argument(
        "--classification-root",
        type=Path,
        default=Path(
            "reports_Ac_class/permutation_sensitivity/AC"
        ),
        help=(
            "Root containing classification fs folders. "
            "Default: reports_Ac_class/permutation_sensitivity/AC"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "reports_permutation/permutation_sensitivity_all_models_comprehensive.csv"
        ),
        help="Output path for the row-level combined CSV.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path(
            "reports_permutation/permutation_sensitivity_feature_summary.csv"
        ),
        help="Output path for the feature-level aggregated summary.",
    )
    args = parser.parse_args()

    regression, reg_warnings = collect_folder(
        root=args.regression_root,
        task="regression",
        experiment_family="GLFS",
    )

    classification, cls_warnings = collect_folder(
        root=args.classification_root,
        task="classification",
        experiment_family="GLFS",
    )

    warnings = reg_warnings + cls_warnings

    available = [
        frame for frame in [regression, classification]
        if not frame.empty
    ]

    if not available:
        print("\n".join(warnings))
        raise SystemExit(
            "No valid permutation-sensitivity data were found."
        )

    combined = pd.concat(available, ignore_index=True, sort=False)

    sort_cols = [
        c for c in [
            "task",
            "experiment_family",
            "feature_set",
            "model",
            "importance_rank_within_model",
            "feature",
        ]
        if c in combined.columns
    ]
    combined = combined.sort_values(sort_cols, kind="stable")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False, encoding="utf-8-sig")

    summary = build_summary(combined)
    if not summary.empty:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(
            args.summary_output,
            index=False,
            encoding="utf-8-sig",
        )

    print("=" * 72)
    print("PERMUTATION-SENSITIVITY EXTRACTION COMPLETED")
    print("=" * 72)
    print(f"Combined rows: {len(combined):,}")
    print(f"Tasks: {sorted(combined['task'].dropna().unique())}")
    print(
        "Feature sets: "
        f"{sorted(combined['feature_set'].dropna().unique(), key=natural_key)}"
    )
    print(f"Models: {sorted(combined['model'].dropna().unique())}")
    print(f"Combined CSV: {args.output.resolve()}")

    if not summary.empty:
        print(f"Summary rows: {len(summary):,}")
        print(f"Summary CSV: {args.summary_output.resolve()}")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")


if __name__ == "__main__":
    main()
