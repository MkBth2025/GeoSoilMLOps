from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Headless/container-safe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

try:
    import yaml
except ImportError as exc:
    raise ImportError(
        "PyYAML is required. Install it with: pip install pyyaml"
    ) from exc


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _log(message: str) -> None:
    print(message, flush=True)


def _safe_filename(value: str) -> str:
    """Convert a column name into a filesystem-safe filename."""
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return name or "column"


def _safe_to_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    converted = df.copy()
    for col in cols:
        if col in converted.columns and not pd.api.types.is_numeric_dtype(converted[col]):
            converted[col] = pd.to_numeric(converted[col], errors="coerce")
    return converted


def _numeric_conversion_ratio(series: pd.Series) -> float:
    """Return the fraction of non-missing values that can be converted to numeric."""
    non_missing = series.dropna()
    if non_missing.empty:
        return 0.0
    converted = pd.to_numeric(non_missing, errors="coerce")
    return float(converted.notna().mean())



# -----------------------------------------------------------------------------
# Parameter-file column discovery
# -----------------------------------------------------------------------------
def _unique_preserve_order(values: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values:
        value = str(value).strip()
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _find_case_insensitive(value: str, candidates: List[str]) -> Optional[str]:
    wanted = str(value).strip().casefold()
    for candidate in candidates:
        if str(candidate).strip().casefold() == wanted:
            return candidate
    return None


def load_columns_from_params(
    params_path: Path,
    requested_target: Optional[str],
    csv_columns: List[str],
) -> tuple[List[str], Optional[str], Dict[str, Any]]:
    """
    Read the target and predictor columns automatically from feature sets in
    ``params.yaml``.

    Expected structure::

        TARGETS:
          AC:
            fs1: ["Ucs_class", "PI/FF"]
            fs2: ["Ucs_class", "PI", "FF"]
            fs3: ["Ucs_class", "LL", "PL", "FF"]
            fs4: ["Ucs_class", "PI"]

    The returned analysis-column list is the union of the selected target and
    all features appearing in its feature sets, with duplicates removed while
    preserving their YAML order.

    ``--target`` is optional when only one target exists in TARGETS.  If more
    than one target exists, ``--target`` selects which target/feature-set group
    is analyzed.
    """
    with params_path.open("r", encoding="utf-8") as stream:
        params = yaml.safe_load(stream) or {}

    if not isinstance(params, dict):
        raise ValueError("The YAML root must be a mapping.")

    # Accept both TARGETS and targets for convenience.
    targets_cfg = params.get("TARGETS", params.get("targets"))
    if not isinstance(targets_cfg, dict) or not targets_cfg:
        raise ValueError(
            "No non-empty TARGETS mapping was found in the parameter file."
        )

    yaml_targets = [str(name) for name in targets_cfg.keys()]

    # Resolve the target automatically whenever possible.
    if requested_target:
        target_key = _find_case_insensitive(requested_target, yaml_targets)
        if target_key is None:
            raise ValueError(
                f"Target '{requested_target}' is not defined under TARGETS. "
                f"Available targets: {yaml_targets}"
            )
    elif len(yaml_targets) == 1:
        target_key = yaml_targets[0]
        _log(f"[INFO] Target automatically selected from params: {target_key}")
    else:
        raise ValueError(
            "Multiple targets are defined under TARGETS. Use --target to select "
            f"one of: {yaml_targets}"
        )

    target_cfg = targets_cfg[target_key]

    # Extract feature-set definitions.  A target may be represented directly
    # as a list, although the normal/expected form is a mapping of fs1, fs2, ...
    feature_sets: Dict[str, List[str]] = {}
    if isinstance(target_cfg, dict):
        for fs_name, values in target_cfg.items():
            if isinstance(values, (list, tuple)):
                feature_sets[str(fs_name)] = [str(item).strip() for item in values]
    elif isinstance(target_cfg, (list, tuple)):
        feature_sets["features"] = [str(item).strip() for item in target_cfg]
    else:
        raise ValueError(
            f"TARGETS.{target_key} must contain feature-set lists (fs1, fs2, ...)."
        )

    if not feature_sets:
        raise ValueError(
            f"No feature sets were found under TARGETS.{target_key}."
        )

    # Union of all feature-set variables, preserving first appearance order.
    feature_columns = _unique_preserve_order(
        [feature for values in feature_sets.values() for feature in values]
    )
    configured = _unique_preserve_order([target_key] + feature_columns)

    # Resolve YAML names against actual CSV column spelling/capitalization.
    resolved: List[str] = []
    missing: List[str] = []
    yaml_to_csv: Dict[str, Optional[str]] = {}
    for name in configured:
        match = _find_case_insensitive(name, csv_columns)
        yaml_to_csv[name] = match
        if match is None:
            missing.append(name)
        else:
            resolved.append(match)

    resolved = _unique_preserve_order(resolved)
    target_csv = _find_case_insensitive(target_key, csv_columns)

    # Also retain each feature set after resolving its names against the CSV.
    resolved_feature_sets: Dict[str, List[str]] = {}
    missing_by_feature_set: Dict[str, List[str]] = {}
    for fs_name, values in feature_sets.items():
        resolved_values: List[str] = []
        missing_values: List[str] = []
        for value in values:
            match = _find_case_insensitive(value, csv_columns)
            if match is None:
                missing_values.append(value)
            else:
                resolved_values.append(match)
        resolved_feature_sets[fs_name] = _unique_preserve_order(resolved_values)
        if missing_values:
            missing_by_feature_set[fs_name] = missing_values

    classification_cfg = params.get("classification", {}) or {}
    target_modes_cfg = classification_cfg.get("target_modes", {}) or {}
    configured_target_mode = (
        str(target_modes_cfg.get(target_key, "auto"))
        if isinstance(target_modes_cfg, dict) else "auto"
    )

    metadata = {
        "source": "params_feature_sets",
        "params_file": str(params_path),
        "available_targets": yaml_targets,
        "configured_target": target_key,
        "resolved_target": target_csv,
        "feature_sets": feature_sets,
        "resolved_feature_sets": resolved_feature_sets,
        "feature_columns_union": feature_columns,
        "configured_columns": configured,
        "resolved_columns": resolved,
        "missing_configured_columns": missing,
        "missing_by_feature_set": missing_by_feature_set,
        "yaml_to_csv": yaml_to_csv,
        "classification_target_mode": configured_target_mode,
    }
    return resolved, target_csv, metadata

def resolve_analysis_configuration(
    df: pd.DataFrame,
    params_argument: Optional[str],
    requested_cols: Optional[List[str]],
    requested_target: Optional[str],
    min_numeric_ratio: float,
) -> tuple[List[str], List[str], Optional[str], Dict[str, Any]]:
    """
    Use params.yaml when it exists; otherwise detect column titles from the CSV.
    """
    csv_columns = [str(col) for col in df.columns]
    detected_numeric = _autodetect_cols(df, min_numeric_ratio=min_numeric_ratio)

    metadata: Dict[str, Any] = {
        "source": "csv",
        "params_file": params_argument,
        "configured_columns": [],
        "resolved_columns": [],
        "missing_configured_columns": [],
    }
    configured: List[str] = []
    resolved_target = (
        _find_case_insensitive(requested_target, csv_columns)
        if requested_target else None
    )

    if params_argument:
        params_path = Path(params_argument)
        if params_path.is_file():
            try:
                configured, params_target, metadata = load_columns_from_params(
                    params_path, requested_target, csv_columns
                )
                if params_target is not None:
                    resolved_target = params_target
                if metadata["missing_configured_columns"]:
                    _log(
                        "[WARN] Columns in params but absent from CSV: "
                        f"{metadata['missing_configured_columns']}"
                    )
                _log(f"[INFO] Column titles loaded from: {params_path}")
                if metadata.get("feature_sets"):
                    _log(
                        "[INFO] Feature sets discovered: "
                        + ", ".join(
                            f"{name}={values}"
                            for name, values in metadata["feature_sets"].items()
                        )
                    )
                    _log(
                        "[INFO] Union of target + feature-set columns: "
                        f"{metadata.get('configured_columns', [])}"
                    )
            except Exception as exc:
                _log(
                    f"[WARN] Could not use '{params_path}': {exc}. "
                    "Using numeric CSV column titles."
                )
                metadata["params_error"] = str(exc)
        else:
            _log(
                f"[WARN] Parameter file not found: {params_path}. "
                "Using numeric CSV column titles."
            )
            metadata["params_error"] = "File not found"

    if requested_cols:
        analysis_cols = []
        missing_requested = []
        for name in requested_cols:
            match = _find_case_insensitive(name, csv_columns)
            if match is None:
                missing_requested.append(name)
            else:
                analysis_cols.append(match)
        analysis_cols = _unique_preserve_order(analysis_cols)
        if missing_requested:
            _log(f"[WARN] --cols absent from CSV: {missing_requested}")
    else:
        analysis_cols = configured or detected_numeric

    normality_candidates = configured or detected_numeric
    normality_cols = [
        col for col in normality_candidates
        if col in df.columns and (
            pd.api.types.is_numeric_dtype(df[col])
            or _numeric_conversion_ratio(df[col]) >= min_numeric_ratio
        )
    ]
    normality_cols = _unique_preserve_order(normality_cols) or detected_numeric
    analysis_cols = analysis_cols or detected_numeric

    if resolved_target:
        mode = str(metadata.get("classification_target_mode", "auto") or "auto").lower()
        is_categorical_target = mode in {
            "categorical", "category", "labels", "label", "direct",
            "existing_classes", "existing"
        }
        if mode == "auto":
            series = df[resolved_target].dropna()
            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.isna().any():
                is_categorical_target = True
            elif len(numeric):
                nunique = int(numeric.nunique())
                n = int(len(numeric))
                is_categorical_target = (
                    nunique <= max(20, int(np.ceil(np.sqrt(max(n, 1)))))
                    and nunique / max(n, 1) <= 0.20
                )
        if is_categorical_target:
            normality_cols = [col for col in normality_cols if col != resolved_target]
            metadata["resolved_target_mode"] = "categorical"
            _log(
                f"[INFO] Categorical target '{resolved_target}' excluded from normality testing; "
                "normality is not defined for class labels."
            )

    if requested_target and resolved_target is None:
        _log(
            f"[WARN] Target '{requested_target}' was not found in the CSV; "
            "target-specific outputs will be skipped."
        )

    return analysis_cols, normality_cols, resolved_target, metadata


# -----------------------------------------------------------------------------
# Data-quality structures
# -----------------------------------------------------------------------------
@dataclass
class ConstraintRule:
    name: str
    condition: str
    passed: bool
    n_violations: int
    examples: List[int]


@dataclass
class QualitySummary:
    total_rows: int
    columns_checked: List[str]
    missing_rate: Dict[str, float]
    duplicate_rows: int
    constraint_rules: List[ConstraintRule]
    risk_level: str


# -----------------------------------------------------------------------------
# Numeric-column detection
# -----------------------------------------------------------------------------
def _autodetect_cols(
    df: pd.DataFrame,
    min_numeric_ratio: float = 0.90,
) -> List[str]:
    """
    Detect genuinely numeric columns and numeric-like text columns.

    A non-numeric column is included when at least ``min_numeric_ratio`` of its
    non-missing values can be converted to numbers.
    """
    detected: List[str] = []

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            detected.append(col)
            continue

        ratio = _numeric_conversion_ratio(df[col])
        if ratio >= min_numeric_ratio:
            detected.append(col)

    return detected


# -----------------------------------------------------------------------------
# Quality checks
# -----------------------------------------------------------------------------
def run_quality_checks(
    df: pd.DataFrame,
    cols: List[str],
    fail_missing: float = 0.25,
    warn_missing: float = 0.05,
) -> QualitySummary:
    present = [col for col in cols if col in df.columns]
    missing = [col for col in cols if col not in df.columns]

    if missing:
        _log(f"[WARN] Missing columns; skipped in checks: {missing}")

    d = _safe_to_numeric(df[present], present)

    missing_rate = d.isnull().mean().to_dict()
    duplicate_rows = int(d.duplicated().sum())
    rules: List[ConstraintRule] = []

    def add_rule(name: str, condition: str, violation_mask: pd.Series) -> None:
        indexes = d.index[violation_mask.fillna(False)].tolist()
        rules.append(
            ConstraintRule(
                name=name,
                condition=condition,
                passed=len(indexes) == 0,
                n_violations=len(indexes),
                examples=indexes[:5],
            )
        )

    # Generic non-negative check retained from the original script.
    for col in present:
        add_rule(
            name=f"{col}_nonnegative",
            condition=f"{col} must be >= 0",
            violation_mask=d[col] < 0,
        )

    percentage_like_cols = [
        col
        for col in present
        if any(
            keyword in col.lower()
            for keyword in ["rate", "ratio", "percent", "percentage", "pct", "%"]
        )
    ]

    for col in percentage_like_cols:
        add_rule(
            name=f"{col}_in_0_100",
            condition=f"{col} must be within [0, 100]",
            violation_mask=(d[col] < 0) | (d[col] > 100),
        )

    status = "PASS"
    if any(float(value or 0.0) >= fail_missing for value in missing_rate.values()):
        status = "FAIL"
    elif (
        any(float(value or 0.0) >= warn_missing for value in missing_rate.values())
        or any(rule.n_violations > 0 for rule in rules)
    ):
        status = "WARN"

    return QualitySummary(
        total_rows=int(len(d)),
        columns_checked=present,
        missing_rate={key: float(value) for key, value in missing_rate.items()},
        duplicate_rows=duplicate_rows,
        constraint_rules=rules,
        risk_level=status,
    )


# -----------------------------------------------------------------------------
# Descriptive statistics and correlations
# -----------------------------------------------------------------------------
def make_table2_stats(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    present = [col for col in cols if col in df.columns]
    if not present:
        raise ValueError("None of the requested columns exist in the data.")

    numeric = _safe_to_numeric(df[present], present)
    summary = numeric.describe().T
    return summary.reindex(
        columns=["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    )


def save_heatmap(
    corr: pd.DataFrame,
    cols: List[str],
    target: Optional[str],
    outpath: Path,
    title: str,
) -> None:
    size = max(8.0, min(18.0, 0.55 * len(cols) + 4.0))
    fig, ax = plt.subplots(figsize=(size, size * 0.80))
    image = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)

    annotation_size = 8 if len(cols) <= 12 else 6
    for i in range(len(cols)):
        for j in range(len(cols)):
            value = corr.values[i, j]
            text = "" if np.isnan(value) else f"{value:.2f}"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="black",
                fontsize=annotation_size,
            )

    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticklabels(cols)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)

    if target in cols:
        index = cols.index(target)
        ax.add_patch(
            plt.Rectangle(
                (index - 0.5, -0.5),
                1,
                len(cols),
                fill=False,
                linewidth=2,
                edgecolor="black",
            )
        )
        ax.add_patch(
            plt.Rectangle(
                (-0.5, index - 0.5),
                len(cols),
                1,
                fill=False,
                linewidth=2,
                edgecolor="black",
            )
        )

    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_correlation_and_heatmaps(
    df: pd.DataFrame,
    cols: List[str],
    target: Optional[str],
    outdir: Path,
) -> Dict[str, Any]:
    present = [col for col in cols if col in df.columns]
    numeric = _safe_to_numeric(df[present], present)

    # Remove columns that became entirely missing after conversion.
    usable = [col for col in present if numeric[col].notna().any()]
    if len(usable) < 2:
        raise ValueError("At least two usable numeric columns are required for correlation.")

    numeric = numeric[usable]
    pearson = numeric.corr(method="pearson")
    spearman = numeric.corr(method="spearman")

    pearson_csv = outdir / "corr_pearson.csv"
    spearman_csv = outdir / "corr_spearman.csv"
    pearson.to_csv(pearson_csv)
    spearman.to_csv(spearman_csv)

    pearson_plot = outdir / "heatmap_corr_pearson.png"
    spearman_plot = outdir / "heatmap_corr_spearman.png"

    save_heatmap(
        pearson,
        usable,
        target,
        pearson_plot,
        "Correlation Heatmap (Pearson)",
    )
    save_heatmap(
        spearman,
        usable,
        target,
        spearman_plot,
        "Correlation Heatmap (Spearman)",
    )

    top_corr = None
    if target in usable:
        target_corr = (
            pearson[target]
            .drop(labels=[target], errors="ignore")
            .dropna()
            .sort_values(key=lambda values: values.abs(), ascending=False)
        )
        top_corr = target_corr.to_frame(name="pearson_with_target")
        top_corr.to_csv(outdir / "top_corr_with_target.csv")

    return {
        "pearson_csv": str(pearson_csv),
        "spearman_csv": str(spearman_csv),
        "heatmap_pearson": str(pearson_plot),
        "heatmap_spearman": str(spearman_plot),
        "top_corr_with_target_csv": (
            None if top_corr is None else str(outdir / "top_corr_with_target.csv")
        ),
        "top_corr_with_target": (
            None if top_corr is None else top_corr["pearson_with_target"].to_dict()
        ),
    }


# -----------------------------------------------------------------------------
# Normality analysis
# -----------------------------------------------------------------------------
def _normality_decision(p_value: Optional[float], alpha: float) -> str:
    if p_value is None or not np.isfinite(p_value):
        return "Not assessed"
    return "Approximately normal" if p_value >= alpha else "Non-normal"


def save_normality_plot(
    df: pd.DataFrame,
    col: str,
    outpath: Path,
    alpha: float = 0.05,
) -> Optional[Dict[str, Any]]:
    """
    Save a histogram with fitted normal density and a normal Q-Q plot.

    Returns a dictionary containing descriptive statistics and formal
    normality-test results.
    """
    if col not in df.columns:
        return None

    values = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
    values = values[np.isfinite(values)]

    if values.size < 3:
        return {
            "column": col,
            "n": int(values.size),
            "plot": None,
            "mean": None,
            "std": None,
            "skewness": None,
            "excess_kurtosis": None,
            "shapiro_w": None,
            "shapiro_p": None,
            "dagostino_k2": None,
            "dagostino_p": None,
            "anderson_statistic": None,
            "anderson_critical_5pct": None,
            "decision_alpha": alpha,
            "normality_decision": "Insufficient data",
            "note": "At least three finite observations are required.",
        }

    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    skewness = float(stats.skew(values, bias=False)) if values.size >= 3 else None
    excess_kurtosis = (
        float(stats.kurtosis(values, fisher=True, bias=False)) if values.size >= 4 else None
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Histogram and fitted normal density.
    axes[0].hist(values, bins="auto", density=True, alpha=0.70, edgecolor="black")
    if std > 0 and np.isfinite(std):
        x_grid = np.linspace(float(np.min(values)), float(np.max(values)), 400)
        axes[0].plot(x_grid, stats.norm.pdf(x_grid, loc=mean, scale=std), linewidth=2)

    axes[0].set_title(f"{col}: histogram and fitted normal curve")
    axes[0].set_xlabel(col)
    axes[0].set_ylabel("Density")
    axes[0].grid(alpha=0.25)

    # Q-Q plot.
    stats.probplot(values, dist="norm", plot=axes[1])
    axes[1].set_title(f"{col}: normal Q-Q plot")
    axes[1].grid(alpha=0.25)

    fig.suptitle(
        f"Normality assessment: {col} (n={values.size}, mean={mean:.3f}, SD={std:.3f})",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(outpath, dpi=250, bbox_inches="tight")
    plt.close(fig)

    # Shapiro-Wilk can be used for larger samples in SciPy, but its p-value may
    # become less reliable for N > 5000. We still report W and mark the caveat.
    shapiro_w: Optional[float]
    shapiro_p: Optional[float]
    shapiro_note: Optional[str] = None
    try:
        shapiro_result = stats.shapiro(values)
        shapiro_w = float(shapiro_result.statistic)
        shapiro_p = float(shapiro_result.pvalue)
        if values.size > 5000:
            shapiro_note = "For n > 5000, the Shapiro-Wilk p-value may be less accurate."
    except Exception as exc:
        shapiro_w = None
        shapiro_p = None
        shapiro_note = f"Shapiro-Wilk failed: {exc}"

    # D'Agostino-Pearson requires at least eight observations.
    dagostino_k2: Optional[float] = None
    dagostino_p: Optional[float] = None
    dagostino_note: Optional[str] = None
    if values.size >= 8:
        try:
            dagostino_result = stats.normaltest(values)
            dagostino_k2 = float(dagostino_result.statistic)
            dagostino_p = float(dagostino_result.pvalue)
        except Exception as exc:
            dagostino_note = f"D'Agostino-Pearson test failed: {exc}"
    else:
        dagostino_note = "At least eight observations are required."

    # Anderson-Darling for the normal distribution.
    anderson_statistic: Optional[float] = None
    anderson_critical_5pct: Optional[float] = None
    anderson_decision: Optional[str] = None
    try:
        anderson_result = stats.anderson(values, dist="norm")
        anderson_statistic = float(anderson_result.statistic)

        significance_levels = np.asarray(anderson_result.significance_level, dtype=float)
        critical_values = np.asarray(anderson_result.critical_values, dtype=float)
        closest_index = int(np.argmin(np.abs(significance_levels - 5.0)))
        anderson_critical_5pct = float(critical_values[closest_index])
        anderson_decision = (
            "Approximately normal"
            if anderson_statistic < anderson_critical_5pct
            else "Non-normal"
        )
    except Exception as exc:
        anderson_decision = f"Not assessed: {exc}"

    decision = _normality_decision(shapiro_p, alpha)

    return {
        "column": col,
        "n": int(values.size),
        "plot": str(outpath),
        "mean": mean,
        "std": std,
        "skewness": skewness,
        "excess_kurtosis": excess_kurtosis,
        "shapiro_w": shapiro_w,
        "shapiro_p": shapiro_p,
        "dagostino_k2": dagostino_k2,
        "dagostino_p": dagostino_p,
        "anderson_statistic": anderson_statistic,
        "anderson_critical_5pct": anderson_critical_5pct,
        "anderson_decision_5pct": anderson_decision,
        "decision_alpha": alpha,
        "normality_decision": decision,
        "note": shapiro_note,
        "dagostino_note": dagostino_note,
    }



def save_all_normality_plot(
    df: pd.DataFrame,
    numeric_cols: List[str],
    results: Dict[str, Dict[str, Any]],
    outpath: Path,
    max_columns_per_page: int = 12,
) -> Optional[str]:
    """
    Save a publication-ready combined normality figure.

    Each variable occupies one row: histogram on the left and normal Q--Q
    plot on the right. The first output is saved as ``all normality.png``.
    If more than ``max_columns_per_page`` variables are present, additional
    page images are saved automatically.
    """
    usable_cols = [
        col for col in numeric_cols
        if col in df.columns
        and col in results
        and int(results[col].get("n") or 0) >= 3
    ]
    if not usable_cols:
        return None

    chunks = [
        usable_cols[i:i + max_columns_per_page]
        for i in range(0, len(usable_cols), max_columns_per_page)
    ]

    first_path: Optional[str] = None
    for page_index, page_cols in enumerate(chunks, start=1):
        n_rows = len(page_cols)
        fig, axes = plt.subplots(
            n_rows, 2, figsize=(11.5, max(4.5, 2.45 * n_rows)), squeeze=False
        )

        for row_index, col in enumerate(page_cols):
            values = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            result = results[col]
            mean = float(result["mean"])
            std = float(result["std"])
            shapiro_w = result.get("shapiro_w")
            shapiro_p = result.get("shapiro_p")
            decision = result.get("normality_decision", "Not assessed")

            hist_ax = axes[row_index, 0]
            qq_ax = axes[row_index, 1]

            hist_ax.hist(values, bins="auto", density=True, alpha=0.72,
                         edgecolor="black", linewidth=0.6)
            if std > 0 and np.isfinite(std):
                x_grid = np.linspace(float(np.min(values)), float(np.max(values)), 400)
                hist_ax.plot(x_grid, stats.norm.pdf(x_grid, loc=mean, scale=std),
                             linewidth=1.7)

            if shapiro_p is None or not np.isfinite(shapiro_p):
                p_text = "p = NA"
            elif shapiro_p < 0.001:
                p_text = "p < 0.001"
            else:
                p_text = f"p = {shapiro_p:.3f}"
            w_text = "W = NA" if shapiro_w is None else f"W = {shapiro_w:.3f}"

            hist_ax.set_title(f"{col}: histogram ({w_text}, {p_text}; {decision})",
                              fontsize=9, pad=5)
            hist_ax.set_xlabel(col, fontsize=8)
            hist_ax.set_ylabel("Density", fontsize=8)
            hist_ax.tick_params(axis="both", labelsize=7)
            hist_ax.grid(alpha=0.20, linewidth=0.5)

            stats.probplot(values, dist="norm", plot=qq_ax)
            qq_ax.set_title(f"{col}: normal Q-Q plot", fontsize=9, pad=5)
            qq_ax.set_xlabel("Theoretical quantiles", fontsize=8)
            qq_ax.set_ylabel("Ordered values", fontsize=8)
            qq_ax.tick_params(axis="both", labelsize=7)
            qq_ax.grid(alpha=0.20, linewidth=0.5)
            for line in qq_ax.get_lines():
                if line.get_linestyle() == "None":
                    line.set_markersize(3.0)
                else:
                    line.set_linewidth(1.2)

        title = "Normality assessment of numeric study variables"
        if len(chunks) > 1:
            title += f" (page {page_index} of {len(chunks)})"
        fig.suptitle(title, fontsize=13, y=0.998)
        fig.tight_layout(rect=(0, 0, 1, 0.992), h_pad=1.5, w_pad=1.4)

        page_path = outpath if page_index == 1 else outpath.with_name(
            f"{outpath.stem}_page_{page_index}{outpath.suffix}"
        )
        fig.savefig(page_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        if first_path is None:
            first_path = str(page_path)

    return first_path

def run_normality_analysis(
    df: pd.DataFrame,
    numeric_cols: List[str],
    outdir: Path,
    alpha: float,
) -> tuple[Dict[str, Dict[str, Any]], Path, Optional[Path]]:
    normality_dir = _ensure_dir(outdir / "normality")
    results: Dict[str, Dict[str, Any]] = {}

    for col in numeric_cols:
        filename = f"{_safe_filename(col)}_normality.png"
        result = save_normality_plot(
            df=df,
            col=col,
            outpath=normality_dir / filename,
            alpha=alpha,
        )
        if result is not None:
            results[col] = result

    summary_rows = []
    for col, result in results.items():
        summary_rows.append(
            {
                "column": col,
                "n": result.get("n"),
                "mean": result.get("mean"),
                "std": result.get("std"),
                "skewness": result.get("skewness"),
                "excess_kurtosis": result.get("excess_kurtosis"),
                "shapiro_w": result.get("shapiro_w"),
                "shapiro_p": result.get("shapiro_p"),
                "dagostino_k2": result.get("dagostino_k2"),
                "dagostino_p": result.get("dagostino_p"),
                "anderson_statistic": result.get("anderson_statistic"),
                "anderson_critical_5pct": result.get("anderson_critical_5pct"),
                "anderson_decision_5pct": result.get("anderson_decision_5pct"),
                "alpha": result.get("decision_alpha"),
                "normality_decision_shapiro": result.get("normality_decision"),
                "plot": result.get("plot"),
                "note": result.get("note"),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = outdir / "normality_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    combined_plot = outdir / "all normality.png"
    combined_result = save_all_normality_plot(
        df=df, numeric_cols=numeric_cols, results=results, outpath=combined_plot
    )
    combined_plot_path = Path(combined_result) if combined_result else None

    return results, summary_csv, combined_plot_path


# -----------------------------------------------------------------------------
# Markdown report
# -----------------------------------------------------------------------------
def write_markdown_report(
    outdir: Path,
    quality: Optional[QualitySummary],
    stats_csv: Optional[Path],
    corr_info: Optional[Dict[str, Any]],
    normality_results: Dict[str, Dict[str, Any]],
    normality_summary_csv: Optional[Path],
    all_normality_plot: Optional[Path],
    target: Optional[str],
    errors: List[str],
) -> None:
    lines: List[str] = []
    lines.append("# Data Quality and Statistical Analysis\n")
    lines.append(f"- **Working directory:** `{os.getcwd()}`")
    lines.append(f"- **Output directory:** `{outdir.resolve()}`\n")

    if quality:
        lines.append(f"- **Rows checked:** {quality.total_rows}")
        lines.append(f"- **Columns analyzed:** {', '.join(quality.columns_checked)}")
        lines.append(f"- **Duplicate rows:** {quality.duplicate_rows}")
        lines.append(f"- **Overall status:** **{quality.risk_level}**\n")

        lines.append("## Missingness\n")
        lines.append("| Column | Missing % |")
        lines.append("|---|---:|")
        for col, rate in quality.missing_rate.items():
            lines.append(f"| {col} | {100.0 * float(rate):.2f}% |")

        lines.append("\n## Constraint Checks\n")
        if not quality.constraint_rules:
            lines.append("_No constraints were evaluated._\n")
        else:
            lines.append("| Rule | Condition | Passed | Violations | Example indexes |")
            lines.append("|---|---|:---:|---:|---|")
            for rule in quality.constraint_rules:
                examples = ", ".join(map(str, rule.examples)) if rule.examples else "-"
                passed = "Yes" if rule.passed else "No"
                lines.append(
                    f"| {rule.name} | {rule.condition} | {passed} | "
                    f"{rule.n_violations} | {examples} |"
                )
    else:
        lines.append("_Quality checks were unavailable because of errors._\n")

    lines.append("\n## Descriptive Statistics\n")
    lines.append(f"- CSV: `{stats_csv.name if stats_csv else 'N/A'}`\n")

    lines.append("## Correlations\n")
    if corr_info:
        lines.append(f"- Pearson CSV: `{Path(corr_info['pearson_csv']).name}`")
        lines.append(f"- Spearman CSV: `{Path(corr_info['spearman_csv']).name}`")
        lines.append(
            f"- Heatmaps: `{Path(corr_info['heatmap_pearson']).name}`, "
            f"`{Path(corr_info['heatmap_spearman']).name}`\n"
        )

        if target and corr_info.get("top_corr_with_target"):
            lines.append(f"**Pearson correlations with target `{target}`:**\n")
            lines.append("| Feature | r |")
            lines.append("|---|---:|")
            ordered = sorted(
                corr_info["top_corr_with_target"].items(),
                key=lambda item: -abs(item[1]),
            )
            for feature, value in ordered:
                lines.append(f"| {feature} | {value:.3f} |")
            lines.append("")
    else:
        lines.append("_Correlation artifacts were unavailable._\n")

    lines.append("## Normality Analysis\n")
    if normality_summary_csv:
        lines.append(f"- Summary CSV: `{normality_summary_csv.name}`")
    if all_normality_plot:
        lines.append(f"- Combined publication figure: `{all_normality_plot.name}`")
    lines.append(
        "- Interpretation: for the Shapiro-Wilk test, p >= alpha does not reject "
        "normality; p < alpha indicates evidence against normality."
    )

    if normality_results:
        lines.append("\n| Column | n | Shapiro W | p-value | Decision | Plot |")
        lines.append("|---|---:|---:|---:|---|---|")
        for col, result in normality_results.items():
            w = result.get("shapiro_w")
            p = result.get("shapiro_p")
            w_text = "-" if w is None else f"{w:.5f}"
            p_text = "-" if p is None else f"{p:.6g}"
            plot = result.get("plot")
            plot_text = "-" if not plot else Path(plot).name
            lines.append(
                f"| {col} | {result.get('n', '-')} | {w_text} | {p_text} | "
                f"{result.get('normality_decision', '-')} | `{plot_text}` |"
            )
    else:
        lines.append("\n_No usable numeric columns were available for normality analysis._")

    if errors:
        lines.append("\n## Errors\n")
        for error in errors:
            lines.append(f"- {error}")

    report_path = outdir / "data_quality_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# Command-line interface
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Data quality, descriptive statistics, correlation analysis, and "
            "normality assessment for all numeric columns in a CSV file."
        )
    )
    parser.add_argument(
        "--data",
        "--input",
        dest="data",
        required=True,
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--cols",
        nargs="+",
        help=(
            "Optional columns for quality checks, descriptive statistics, and "
            "correlations. Normality analysis still covers all detected numeric "
            "columns."
        ),
    )
    parser.add_argument(
        "--target",
        default=None,
        help=(
            "Optional target override. Normally unnecessary: when TARGETS in "
            "params.yaml contains one target, it is selected automatically."
        ),
    )
    parser.add_argument(
        "--params",
        default="params.yaml",
        help=(
            "Path to params.yaml. Target and data columns are automatically built "
            "from TARGETS.<target>.fs* feature sets. If the file is unavailable, "
            "numeric CSV columns are detected instead. Default: params.yaml."
        ),
    )
    parser.add_argument(
        "--outdir",
        "--out",
        dest="outdir",
        default="reports/data_quality",
        help="Output directory.",
    )
    parser.add_argument(
        "--fail-missing",
        type=float,
        default=0.25,
        help="Failure threshold for missing rate per column; default: 0.25.",
    )
    parser.add_argument(
        "--warn-missing",
        type=float,
        default=0.05,
        help="Warning threshold for missing rate per column; default: 0.05.",
    )
    parser.add_argument(
        "--normality-alpha",
        type=float,
        default=0.05,
        help="Significance level for normality decisions; default: 0.05.",
    )
    parser.add_argument(
        "--min-numeric-ratio",
        type=float,
        default=0.90,
        help=(
            "Minimum fraction of convertible non-missing values required to treat "
            "a text column as numeric; default: 0.90."
        ),
    )
    args = parser.parse_args()

    if not 0.0 < args.normality_alpha < 1.0:
        parser.error("--normality-alpha must be between 0 and 1.")
    if not 0.0 <= args.min_numeric_ratio <= 1.0:
        parser.error("--min-numeric-ratio must be between 0 and 1.")

    outdir = _ensure_dir(Path(args.outdir))
    errors: List[str] = []

    _log(f"[INFO] Current working directory: {os.getcwd()}")
    _log(f"[INFO] Reading data from: {args.data}")
    _log(f"[INFO] Writing outputs under: {outdir.resolve()}")

    try:
        df = pd.read_csv(args.data)
    except Exception as exc:
        message = f"Could not read CSV: {exc}"
        print(f"[FATAL] {message}", file=sys.stderr)
        (outdir / "data_quality_report.json").write_text(
            json.dumps({"error": message}, indent=2),
            encoding="utf-8",
        )
        return

    (
        analysis_cols,
        normality_cols,
        target_column,
        params_metadata,
    ) = resolve_analysis_configuration(
        df=df,
        params_argument=args.params,
        requested_cols=args.cols,
        requested_target=args.target,
        min_numeric_ratio=args.min_numeric_ratio,
    )

    if not analysis_cols:
        message = "No usable analysis columns were found."
        print(f"[FATAL] {message}", file=sys.stderr)
        return
    if not normality_cols:
        message = "No numeric or numeric-like columns were detected."
        print(f"[FATAL] {message}", file=sys.stderr)
        return

    _log(f"[INFO] Columns used for quality/stats/correlation: {analysis_cols}")
    _log(f"[INFO] Columns used for normality analysis: {normality_cols}")
    _log(f"[INFO] Resolved target column: {target_column}")

    quality: Optional[QualitySummary] = None
    try:
        quality = run_quality_checks(
            df,
            analysis_cols,
            fail_missing=args.fail_missing,
            warn_missing=args.warn_missing,
        )
    except Exception as exc:
        errors.append(f"quality_checks: {exc}")

    stats_csv: Optional[Path] = None
    try:
        stats_table = make_table2_stats(df, analysis_cols)
        stats_csv = outdir / "table2_stats.csv"
        stats_table.to_csv(stats_csv)
        _log(f"[OK] Descriptive statistics saved: {stats_csv}")
    except Exception as exc:
        errors.append(f"descriptive_statistics: {exc}")

    corr_info: Optional[Dict[str, Any]] = None
    try:
        corr_info = save_correlation_and_heatmaps(
            df,
            analysis_cols,
            target_column,
            outdir,
        )
        _log("[OK] Correlation CSV files and heatmaps saved.")
    except Exception as exc:
        errors.append(f"correlations: {exc}")

    normality_results: Dict[str, Dict[str, Any]] = {}
    normality_summary_csv: Optional[Path] = None
    all_normality_plot: Optional[Path] = None
    try:
        (normality_results, normality_summary_csv, all_normality_plot) = run_normality_analysis(
            df=df,
            numeric_cols=normality_cols,
            outdir=outdir,
            alpha=args.normality_alpha,
        )
        _log(f"[OK] Normality plots saved under: {outdir / 'normality'}")
        _log(f"[OK] Normality summary saved: {normality_summary_csv}")
        if all_normality_plot:
            _log(f"[OK] Combined normality figure saved: {all_normality_plot}")
    except Exception as exc:
        errors.append(f"normality_analysis: {exc}")

    try:
        summary_json = {
            "cwd": os.getcwd(),
            "input_csv": str(Path(args.data)),
            "outdir": str(outdir.resolve()),
            "analysis_columns": analysis_cols,
            "normality_columns": normality_cols,
            "column_configuration": params_metadata,
            "quality": (
                None
                if quality is None
                else {
                    "total_rows": quality.total_rows,
                    "columns_checked": quality.columns_checked,
                    "missing_rate": quality.missing_rate,
                    "duplicate_rows": quality.duplicate_rows,
                    "risk_level": quality.risk_level,
                    "constraints": [asdict(rule) for rule in quality.constraint_rules],
                }
            ),
            "artifacts": {
                "stats_csv": None if stats_csv is None else str(stats_csv),
                "correlation": corr_info,
                "normality_summary_csv": (
                    None
                    if normality_summary_csv is None
                    else str(normality_summary_csv)
                ),
                "normality": normality_results,
                "all_normality_plot": (
                    None if all_normality_plot is None else str(all_normality_plot)
                ),
            },
            "target_requested": args.target,
            "target": target_column,
            "normality_alpha": args.normality_alpha,
            "errors": errors,
        }
        json_path = outdir / "data_quality_report.json"
        json_path.write_text(
            json.dumps(summary_json, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        _log(f"[OK] JSON report saved: {json_path}")
    except Exception as exc:
        print(f"[WARN] Could not write JSON report: {exc}", file=sys.stderr)

    try:
        write_markdown_report(
            outdir=outdir,
            quality=quality,
            stats_csv=stats_csv,
            corr_info=corr_info,
            normality_results=normality_results,
            normality_summary_csv=normality_summary_csv,
            all_normality_plot=all_normality_plot,
            target=target_column,
            errors=errors,
        )
        _log(f"[OK] Markdown report saved: {outdir / 'data_quality_report.md'}")
    except Exception as exc:
        print(f"[WARN] Could not write Markdown report: {exc}", file=sys.stderr)

    if errors:
        _log("[WARN] Completed with some errors. Review data_quality_report.json.")
    else:
        _log("[DONE] Data-quality and normality analysis completed successfully.")


if __name__ == "__main__":
    main()
