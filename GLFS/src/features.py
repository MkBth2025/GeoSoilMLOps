from __future__ import annotations

import argparse
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.model_selection import GroupShuffleSplit, train_test_split

# Optional tqdm support
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Simple fallback that doesn't show progress
    class tqdm:
        def __init__(self, iterable=None, desc=None, disable=True, **kwargs):
            self.iterable = iterable
            self.desc = desc
        
        def __iter__(self):
            if self.iterable is not None:
                for item in self.iterable:
                    yield item
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            pass
        
        def update(self, n=1):
            pass

def _load_params(path: str | Path) -> dict[str, Any]:
    """Load parameters from YAML file with validation."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parameter file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        params = yaml.safe_load(handle) or {}
    
    # Validate required sections exist
    if "split" not in params:
        raise ValueError("Missing 'split' section in params.yaml")
    if "TARGETS" not in params:
        raise ValueError("Missing 'TARGETS' section in params.yaml")
    
    return params


def _validate_ratios(train: float, test: float) -> None:
    """Validate train/test ratios."""
    ratios = np.asarray([train, test], dtype=float)
    if np.any(ratios <= 0.0):
        raise ValueError("Train and test ratios must be positive.")
    if not np.isclose(ratios.sum(), 1.0, atol=1e-8):
        raise ValueError(f"Train and test ratios must sum to 1.0; received {ratios.sum():.8f}.")


def _validate_split_config(params: dict[str, Any]) -> None:
    """Validate split configuration parameters."""
    split_cfg = params.get("split", {})
    
    # ``mode`` is optional. The workflow is fixed-split by design, so when
    # the key is absent we use the same default already used by ``run()``.
    # This keeps older params.yaml files compatible.
    mode = str(split_cfg.get("mode", "fixed")).strip().lower()

    # Train/test ratios remain required because silently inventing them could
    # change an existing experiment.
    if "train" not in split_cfg:
        raise ValueError("Missing 'train' ratio in split configuration")
    if "test" not in split_cfg:
        raise ValueError("Missing 'test' ratio in split configuration")

    # Validate mode.
    #
    # Backward compatibility:
    # Older params.yaml files may use ``mode: random`` to mean a single,
    # reproducible random train/test holdout. This is still a FIXED split
    # because the indices are generated once with ``random_state`` and saved.
    # Therefore, accept both values and normalize ``random`` later in ``run()``.
    valid_modes = {"fixed", "random"}
    if mode not in valid_modes:
        raise ValueError(
            "split.mode must be either 'fixed' or the backward-compatible "
            f"'random'; received {mode!r}."
        )

    if mode == "random":
        warnings.warn(
            "split.mode: random is treated as a fixed reproducible holdout. "
            "For clarity in publication workflows, prefer split.mode: fixed "
            "with stratification_method: random or none.",
            UserWarning,
        )
    
    # Validate stratification method
    strat_method = str(split_cfg.get("stratification_method", "ac_quantile")).strip().lower()
    valid_methods = {"ac_quantile", "ac_class", "none", "random"}
    if strat_method not in valid_methods:
        raise ValueError(f"stratification_method must be one of: {valid_methods}")
    
    # Validate quantile bins if using ac_quantile
    if strat_method == "ac_quantile":
        requested_bins = int(split_cfg.get("ac_quantile_bins", 5))
        if requested_bins < 2:
            raise ValueError("ac_quantile_bins must be >= 2")
        if requested_bins > 20:
            warnings.warn(f"High number of quantile bins ({requested_bins}) requested, may result in small bins")


def _validate_no_target_leakage(feature_sets: dict[str, Any]) -> None:
    """Validate that targets are not used as features."""
    for target, definitions in feature_sets.items():
        for feature_set, columns in definitions.items():
            if target in columns:
                raise ValueError(
                    f"Target '{target}' appears in feature set '{feature_set}'. "
                    "This would cause data leakage."
                )


def _validate_numeric_conversion(df: pd.DataFrame, columns: list[str]) -> None:
    """Validate that columns can be converted to numeric."""
    for column in columns:
        if column not in df.columns:
            raise KeyError(f"Column '{column}' not found in dataframe")
        
        converted = pd.to_numeric(df[column], errors="coerce")
        if converted.isna().all():
            raise ValueError(f"Column '{column}' contains no valid numeric values")
        
        # Check for excessive invalid values
        invalid_ratio = converted.isna().sum() / len(converted)
        if invalid_ratio > 0.5:
            warnings.warn(
                f"Column '{column}' has {invalid_ratio:.1%} invalid values after "
                "numeric conversion. Consider checking data quality."
            )


def _distributions_match(train_dist: np.ndarray, test_dist: np.ndarray, threshold: float = 0.1) -> bool:
    """Check if two distributions match using chi-squared test."""
    # Remove zeros for chi-squared
    nonzero_mask = (train_dist > 0) | (test_dist > 0)
    train_clean = train_dist[nonzero_mask]
    test_clean = test_dist[nonzero_mask]
    
    if len(train_clean) < 2 or len(test_clean) < 2:
        return True  # Too few classes to test
    
    # Add small epsilon to avoid division by zero
    train_clean = train_clean + 1e-10
    test_clean = test_clean + 1e-10
    
    # Normalize
    train_norm = train_clean / train_clean.sum()
    test_norm = test_clean / test_clean.sum()
    
    # Compute chi-squared statistic
    chi2, p_value = stats.chisquare(train_norm, test_norm)
    
    # If p-value is low, distributions are significantly different
    return p_value > threshold


def _validate_split_quality(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    df: pd.DataFrame,
    labels: np.ndarray | None = None,
) -> None:
    """Validate the quality of the split for ML workflows."""
    n_train = len(train_idx)
    n_test = len(test_idx)
    total = n_train + n_test
    
    # Check split size
    if n_train < 10 or n_test < 10:
        raise ValueError(f"Split too small: train={n_train}, test={n_test}. Need at least 10 each.")
    
    # Check class distribution
    if labels is not None:
        train_labels = labels[train_idx]
        test_labels = labels[test_idx]
        
        train_dist = np.bincount(train_labels)
        test_dist = np.bincount(test_labels)
        
        # Check for empty classes
        if np.any(train_dist == 0) and np.any(test_dist == 0):
            # Only warn if both splits missing the same class
            missing_train = set(np.where(train_dist == 0)[0])
            missing_test = set(np.where(test_dist == 0)[0])
            common_missing = missing_train & missing_test
            if common_missing:
                warnings.warn(f"Classes {sorted(common_missing)} missing from both splits")
        
        # Check distribution similarity
        if not _distributions_match(train_dist, test_dist):
            warnings.warn(
                "Class distributions differ significantly between train and test splits. "
                "Consider using stratified splitting if not already used."
            )


def _filter_available_feature_sets(
    feature_sets: dict[str, Any],
    df: pd.DataFrame,
) -> dict[str, dict[str, list[str]]]:
    """Ignore missing configured targets/features and print alerts."""
    available: dict[str, dict[str, list[str]]] = {}

    for target, definitions in feature_sets.items():
        if target not in df.columns:
            print(f"[FEATURES][ALERT] Missing target '{target}'; skipping it.")
            continue

        if not isinstance(definitions, dict):
            print(f"[FEATURES][ALERT] Invalid definitions for '{target}'; skipping.")
            continue

        kept_sets: dict[str, list[str]] = {}
        for feature_set, columns in definitions.items():
            if not isinstance(columns, (list, tuple)):
                print(f"[FEATURES][ALERT] {target}/{feature_set} is invalid; skipping.")
                continue

            present = [c for c in columns if c in df.columns]
            missing = [c for c in columns if c not in df.columns]

            if missing:
                print(
                    f"[FEATURES][ALERT] {target}/{feature_set}: incomplete feature set. "
                    f"Missing {missing}; skipping the entire feature set."
                )
                continue

            kept_sets[feature_set] = present

        if kept_sets:
            available[target] = kept_sets

    return available

def _validate_split_coverage(train_idx: np.ndarray, test_idx: np.ndarray, df: pd.DataFrame) -> None:
    """Validate that the split covers all rows without overlap."""
    if set(train_idx).intersection(test_idx):
        raise RuntimeError("Development and test indices overlap.")
    if len(set(train_idx) | set(test_idx)) != len(df):
        raise RuntimeError("Train/test split does not cover every cleaned row.")




def _classification_target_mode(params: dict[str, Any], target: str, values: Any) -> str:
    """Return ``threshold`` or ``categorical`` for a classification target.

    Explicit YAML takes priority::

        classification:
          target_modes:
            AC: threshold
            Ucs_class: categorical

    In ``auto`` mode, valid configured boundaries imply threshold mode. Otherwise
    non-numeric targets and low-cardinality numeric targets are treated as
    already-classified labels. Continuous numeric targets use threshold mode.
    """
    cfg = params.get("classification", {}) or {}
    modes = cfg.get("target_modes", {}) or {}
    raw_mode = modes.get(target, modes.get(str(target), "auto")) if isinstance(modes, dict) else "auto"
    mode = str(raw_mode or "auto").strip().lower().replace("-", "_")
    if mode in {"categorical", "category", "labels", "label", "direct", "existing_classes", "existing"}:
        return "categorical"
    if mode in {"threshold", "thresholds", "continuous", "continuous_threshold", "three_class"}:
        return "threshold"

    boundaries = cfg.get("class_boundaries", {}) or {}
    candidate = boundaries.get(target, {}) if isinstance(boundaries, dict) else {}
    if isinstance(candidate, dict) and candidate.get("lower") is not None and candidate.get("upper") is not None:
        return "threshold"

    series = pd.Series(values).dropna()
    if series.empty:
        return "threshold"
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        return "categorical"
    nunique = int(numeric.nunique())
    n = int(len(numeric))
    # Conservative low-cardinality rule for integer/class-code targets.
    if nunique <= max(20, int(np.ceil(np.sqrt(max(n, 1))))) and nunique / max(n, 1) <= 0.20:
        return "categorical"
    return "threshold"


def _direct_class_labels(values: Any) -> tuple[np.ndarray, dict[int, str]]:
    """Encode existing binary/multiclass labels as contiguous integer classes."""
    series = pd.Series(values)
    if series.isna().any():
        raise ValueError("Direct classification labels contain missing values after cleaning.")
    # Stable lexical/numeric ordering through pandas' deterministic factorization.
    labels_text = series.astype(str)
    codes, uniques = pd.factorize(labels_text, sort=True)
    if len(uniques) < 2:
        raise ValueError("Classification requires at least two distinct target classes.")
    names = {int(i): str(value) for i, value in enumerate(uniques.tolist())}
    return np.asarray(codes, dtype=np.int64), names

def _automatic_class_boundaries(values: Any) -> tuple[float, float]:
    """Infer two ordered three-class boundaries from numeric target values."""
    array = np.asarray(values, dtype=float).ravel()
    array = array[np.isfinite(array)]

    if array.size < 3:
        raise ValueError(
            "Automatic classification boundaries require at least three "
            "finite target values."
        )

    unique = np.unique(array)
    if unique.size < 3:
        raise ValueError(
            "Automatic classification boundaries require at least three "
            "distinct target values."
        )

    lower = float(np.quantile(array, 1.0 / 3.0))
    upper = float(np.quantile(array, 2.0 / 3.0))

    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        # Fallback for heavily tied/discrete targets.
        n = unique.size
        lower_pos = max(0, min(n - 2, n // 3 - 1))
        upper_pos = max(lower_pos + 1, min(n - 2, (2 * n) // 3 - 1))

        lower = float((unique[lower_pos] + unique[lower_pos + 1]) / 2.0)
        upper = float((unique[upper_pos] + unique[upper_pos + 1]) / 2.0)

    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError(
            "Could not infer two valid ordered classification boundaries "
            "from the target distribution."
        )

    return lower, upper


def _resolve_class_boundaries(
    params: dict[str, Any],
    target: str,
    values: Any,
) -> tuple[float, float, str]:
    """Read target boundaries from params.yaml, otherwise infer them.

    Expected YAML:

    classification:
      class_boundaries:
        AC:
          lower: 0.75
          upper: 1.25
    """
    classification_cfg = params.get("classification", {}) or {}
    boundaries_cfg = classification_cfg.get("class_boundaries", {}) or {}

    target_cfg = (
        boundaries_cfg.get(target, {})
        if isinstance(boundaries_cfg, dict)
        else {}
    )

    if isinstance(target_cfg, dict):
        lower_raw = target_cfg.get("lower")
        upper_raw = target_cfg.get("upper")

        if lower_raw is not None and upper_raw is not None:
            try:
                lower = float(lower_raw)
                upper = float(upper_raw)
            except (TypeError, ValueError):
                lower = np.nan
                upper = np.nan

            if np.isfinite(lower) and np.isfinite(upper) and lower < upper:
                return lower, upper, "params.yaml"

            print(
                "[FEATURES][ALERT] Invalid classification boundaries for "
                f"target '{target}' in params.yaml; automatic boundaries "
                "will be used."
            )

    lower, upper = _automatic_class_boundaries(values)
    return lower, upper, "automatic"


def target_to_three_classes(
    values: Any,
    lower: float,
    upper: float,
) -> np.ndarray:
    """Convert a continuous target to classes 0, 1, 2."""
    values = np.asarray(values, dtype=float).ravel()
    labels = np.ones(values.shape, dtype=np.int64)
    labels[values < float(lower)] = 0
    labels[values > float(upper)] = 2
    return labels


def ac_to_activity_class(
    values: Any,
    lower: float = 0.75,
    upper: float = 1.25,
) -> np.ndarray:
    """Backward-compatible AC wrapper using supplied boundaries."""
    return target_to_three_classes(values, lower, upper)

def _make_quantile_labels(
    values: Any,
    requested_bins: int,
    test_ratio: float,
    min_samples_per_bin: int = 4,
) -> tuple[np.ndarray | None, int]:
    """Create AC quantile labels used only for representative partitioning."""
    series = pd.Series(np.asarray(values, dtype=float).ravel())
    
    # Remove NaN values
    series = series.dropna()
    if len(series) == 0:
        return None, 0
    
    max_bins = max(2, min(int(requested_bins), int(series.nunique())))
    
    # Each bin must provide at least min_samples_per_bin observations in each split
    min_count = max(min_samples_per_bin, int(np.ceil(2.0 / min(test_ratio, 1.0 - test_ratio))))
    
    # Try from max_bins down to 2
    for bins in range(max_bins, 1, -1):
        try:
            labels = pd.qcut(series, q=bins, labels=False, duplicates="drop")
            if labels.isna().any():
                continue
            labels = labels.astype(int)
            counts = labels.value_counts()
            if len(counts) >= 2 and int(counts.min()) >= min_count:
                return labels.to_numpy(dtype=np.int64), int(len(counts))
        except (ValueError, TypeError) as e:
            # Continue to try fewer bins
            continue
    
    return None, 0


def _class_counts(
    values: np.ndarray,
    lower: float,
    upper: float,
) -> dict[str, int]:
    """Count three classes using the resolved target boundaries."""
    counts = np.bincount(
        target_to_three_classes(values, lower, upper),
        minlength=3,
    )
    return {
        "class_0": int(counts[0]),
        "class_1": int(counts[1]),
        "class_2": int(counts[2]),
    }

def _holm_adjust(p_values: list[float]) -> list[float]:
    """Holm-adjust a family of p-values without extra dependencies."""
    p = np.asarray(p_values, dtype=float)
    if p.size == 0:
        return []
    order = np.argsort(p)
    adjusted_sorted = np.empty_like(p)
    running_max = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        candidate = min(1.0, (m - rank) * float(p[idx]))
        running_max = max(running_max, candidate)
        adjusted_sorted[rank] = running_max
    adjusted = np.empty_like(p)
    for rank, idx in enumerate(order):
        adjusted[idx] = adjusted_sorted[rank]
    return adjusted.tolist()


def _activity_class_nonparametric_tests(
    df: pd.DataFrame,
    target: str,
    lower: float,
    upper: float,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Kruskal-Wallis omnibus tests plus Holm-corrected pairwise tests.

    The three target-derived activity classes are tested simultaneously first.
    Only when the omnibus Kruskal-Wallis test is significant (alpha=0.05),
    post-hoc two-sided Mann-Whitney U tests are run for all three class pairs
    and corrected within each feature using Holm's method.
    """
    labels = target_to_three_classes(df[target].to_numpy(dtype=float), lower, upper)
    class_names = {0: "class_0", 1: "class_1", 2: "class_2"}
    omnibus_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []

    for feature in feature_columns:
        values = pd.to_numeric(df[feature], errors="coerce").to_numpy(dtype=float)
        groups = [values[labels == k] for k in range(3)]
        groups = [g[np.isfinite(g)] for g in groups]
        counts = [int(len(g)) for g in groups]

        if any(n == 0 for n in counts):
            omnibus_rows.append({
                "feature": feature, "kruskal_H": np.nan, "p_value": np.nan,
                "significant_0_05": False,
                "n_class_0": counts[0], "n_class_1": counts[1], "n_class_2": counts[2],
                "note": "Kruskal-Wallis not run because at least one class is empty.",
            })
            continue

        try:
            h_stat, p_kw = stats.kruskal(*groups, nan_policy="omit")
        except ValueError as exc:
            omnibus_rows.append({
                "feature": feature, "kruskal_H": np.nan, "p_value": np.nan,
                "significant_0_05": False,
                "n_class_0": counts[0], "n_class_1": counts[1], "n_class_2": counts[2],
                "note": f"Kruskal-Wallis could not be computed: {exc}",
            })
            continue

        significant = bool(p_kw < 0.05)
        omnibus_rows.append({
            "feature": feature, "kruskal_H": float(h_stat), "p_value": float(p_kw),
            "significant_0_05": significant,
            "n_class_0": counts[0], "n_class_1": counts[1], "n_class_2": counts[2],
            "note": "Three-class omnibus Kruskal-Wallis test.",
        })

        if significant:
            raw_rows = []
            raw_p = []
            for a, b in ((0, 1), (0, 2), (1, 2)):
                u_stat, p_u = stats.mannwhitneyu(
                    groups[a], groups[b], alternative="two-sided", method="auto"
                )
                raw_rows.append({
                    "feature": feature,
                    "group_1": class_names[a], "group_2": class_names[b],
                    "n_group_1": counts[a], "n_group_2": counts[b],
                    "mann_whitney_U": float(u_stat), "p_raw": float(p_u),
                })
                raw_p.append(float(p_u))

            adjusted = _holm_adjust(raw_p)
            for row, p_adj in zip(raw_rows, adjusted):
                row["p_holm"] = float(p_adj)
                row["significant_holm_0_05"] = bool(p_adj < 0.05)
                row["correction"] = "Holm (3 pairwise comparisons per feature)"
                pairwise_rows.append(row)

    return pd.DataFrame(omnibus_rows), pd.DataFrame(pairwise_rows)


def _numeric_summary(values: np.ndarray) -> dict[str, float]:
    """Compute numeric summary statistics."""
    values = np.asarray(values, dtype=float).ravel()
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "min": float(np.min(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
    }


def _save_split_files(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    target: str,
    feature_set: str,
    columns: list[str],
    out_dir: Path,
) -> None:
    """Save split files for a target and feature set."""
    y = df[target].to_numpy(dtype=float)
    X = df[columns].to_numpy(dtype=float)
    
    joblib.dump(y[train_idx], out_dir / f"y_train_{target}.joblib")
    joblib.dump(y[test_idx], out_dir / f"y_test_{target}.joblib")
    joblib.dump(X[train_idx], out_dir / f"X_train_{feature_set}.joblib")
    joblib.dump(X[test_idx], out_dir / f"X_test_{feature_set}.joblib")
    
    # Save feature names for interpretability
    joblib.dump(columns, out_dir / f"feature_names_{feature_set}.joblib")


def _cleanup_partial_files(out_dir: Path) -> None:
    """Remove only partial files created by this script.

    The output directory itself is preserved to avoid Windows file-lock errors
    and accidental deletion of unrelated artifacts.
    """
    if not out_dir.exists():
        return

    generated_patterns = (
        "X_train_*.joblib",
        "X_test_*.joblib",
        "y_train_*.joblib",
        "y_test_*.joblib",
        "feature_names_*.joblib",
        "groups_train.joblib",
        "groups_test.joblib",
        "split_info.yaml",
        "activity_class_kruskal_wallis.csv",
        "activity_class_pairwise_holm.csv",
    )
    for pattern in generated_patterns:
        for generated_file in out_dir.glob(pattern):
            try:
                generated_file.unlink()
            except (PermissionError, FileNotFoundError):
                pass


def _get_dependencies() -> dict[str, str]:
    """Get versions of key dependencies."""
    import sklearn
    import scipy
    
    return {
        "sklearn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "joblib": joblib.__version__,
    }


def _infer_separator(filepath: Path) -> str:
    """Infer the separator/delimiter from file extension or content."""
    # Check file extension first
    if filepath.suffix == '.csv':
        return ','
    elif filepath.suffix == '.tsv':
        return '\t'
    else:
        # Try to infer from first line
        try:
            with filepath.open('r', encoding='utf-8') as f:
                first_line = f.readline()
                # Count occurrences of common separators
                comma_count = first_line.count(',')
                tab_count = first_line.count('\t')
                semicolon_count = first_line.count(';')
                
                # Choose the most frequent separator
                max_count = max(comma_count, tab_count, semicolon_count)
                if max_count == 0:
                    return ','
                elif max_count == comma_count:
                    return ','
                elif max_count == tab_count:
                    return '\t'
                else:
                    return ';'
        except Exception:
            return ','


def run(
    input_csv: str,
    out_dir: str,
    params_path: str = "params.yaml",
    verbose: bool = True,
) -> None:
    """
    Create a fixed development/test split for repeated-CV evaluation.
    
    Args:
        input_csv: Path to input CSV file
        out_dir: Output directory for split files
        params_path: Path to parameters YAML file
        verbose: Whether to print progress messages
    
    Returns:
        None
    
    Raises:
        Various exceptions for validation failures or processing errors
    """
    input_csv = Path(input_csv)
    out_dir = Path(out_dir)
    
    # Ensure output directory exists without deleting the whole folder.
    # On Windows, deleting an open/locked directory raises WinError 32.
    # The generated files below are overwritten safely by joblib/yaml.
    out_dir.mkdir(parents=True, exist_ok=True)

    # Remove only stale files generated by this script. Keep unrelated files
    # and subdirectories intact. Locked files are reported and left in place.
    generated_patterns = (
        "X_train_*.joblib",
        "X_test_*.joblib",
        "y_train_*.joblib",
        "y_test_*.joblib",
        "feature_names_*.joblib",
        "groups_train.joblib",
        "groups_test.joblib",
        "split_info.yaml",
        "activity_class_kruskal_wallis.csv",
        "activity_class_pairwise_holm.csv",
    )
    for pattern in generated_patterns:
        for generated_file in out_dir.glob(pattern):
            try:
                generated_file.unlink()
            except PermissionError:
                warnings.warn(
                    f"Could not remove locked output file: {generated_file}. "
                    "It will be overwritten when possible."
                )
    
    try:
        # Load and validate parameters
        if verbose:
            print("[FEATURES] Loading parameters...")
        params = _load_params(params_path)
        _validate_split_config(params)
        
        split_cfg = params.get("split", {})
        configured_mode = str(split_cfg.get("mode", "fixed")).strip().lower()
        effective_mode = "fixed"
        train_ratio = float(split_cfg.get("train", 0.80))
        test_ratio = float(split_cfg.get("test", 0.20))
        random_state = int(split_cfg.get("random_state", 42))
        grouping_enabled = bool(split_cfg.get("grouping_enabled", split_cfg.get("group_column") not in (None, "", False)))
        group_column = split_cfg.get("group_column") if grouping_enabled else None
        if group_column is not None:
            group_column = str(group_column).strip() or None
        bh_column = str(split_cfg.get("bh_column", "")).strip()
        location_column = str(split_cfg.get("location_column", "")).strip()
        stratification_method = str(
            split_cfg.get("stratification_method", "ac_quantile")
        ).strip().lower()
        requested_bins = int(split_cfg.get("ac_quantile_bins", 5))
        cv_splits = int(split_cfg.get("cv_splits", 5))
        cv_repeats = int(split_cfg.get("cv_repeats", 3))
        
        _validate_ratios(train_ratio, test_ratio)
        
        # Load data with inferred separator
        if verbose:
            print("[FEATURES] Loading data...")
        
        # Infer the separator
        separator = _infer_separator(input_csv)
        if verbose:
            print(f"[FEATURES] Using separator: '{separator}'")
        
        # Try with inferred separator and default engine
        try:
            df = pd.read_csv(
                input_csv,
                sep=separator,
                engine="c",  # Use C engine for better performance
                low_memory=False,
            )
        except Exception as e:
            if verbose:
                print(f"[FEATURES] C engine failed, trying with python engine...")
            # Fallback to python engine without low_memory
            df = pd.read_csv(
                input_csv,
                sep=separator,
                engine="python",
            )
        
        df.columns = [str(c).strip() for c in df.columns]

        # ------------------------------------------------------------
        # Optional grouping policy
        # ------------------------------------------------------------
        # Group-aware splitting is opt-in. A dataset may contain Location_No,
        # Site_ID, Borehole, Patient_ID, etc. without automatically changing
        # the validation design. To keep groups intact, configure:
        #
        # split:
        #   grouping_enabled: true
        #   group_column: Location_No
        #
        # For ordinary IID/tabular studies use grouping_enabled: false.
        if grouping_enabled and not group_column:
            raise ValueError(
                "split.grouping_enabled is true but split.group_column is empty."
            )

        # Create PI column if needed
        if "PI" not in df.columns and {"LL", "PL"}.issubset(df.columns):
            if verbose:
                print("[FEATURES] Creating PI column from LL - PL")
            df["PI"] = (
                pd.to_numeric(df["LL"], errors="coerce")
                - pd.to_numeric(df["PL"], errors="coerce")
            )
        
        # Validate feature sets
        feature_sets = params.get("TARGETS", {}) or {}
        if not feature_sets:
            raise ValueError("No definitions found under TARGETS in params.yaml.")
        
        _validate_no_target_leakage(feature_sets)

        # Missing columns are allowed: alert, remove them, and continue.
        feature_sets = _filter_available_feature_sets(feature_sets, df)
        if not feature_sets:
            print(
                "[FEATURES][ALERT] No usable target/feature-set combinations "
                "remain. Nothing to process."
            )
            return

        target_names = list(feature_sets)
        required_columns = set(target_names)
        for definitions in feature_sets.values():
            for columns in definitions.values():
                required_columns.update(columns)

        # Select the target used for stratification. An explicit
        # split.stratification_target has priority. Otherwise prefer AC for
        # backward compatibility, then fall back to the first usable target.
        stratification_target = str(
            split_cfg.get("stratification_target", "")
        ).strip()

        if stratification_target not in target_names:
            if stratification_target:
                print(
                    "[FEATURES][ALERT] split.stratification_target "
                    f"'{stratification_target}' is not available in TARGETS; "
                    "falling back automatically."
                )
            stratification_target = (
                "AC" if "AC" in target_names else target_names[0]
            )

        if verbose:
            print(
                f"[FEATURES] Stratification target: {stratification_target}"
            )

        if group_column:
            if group_column in df.columns:
                required_columns.add(group_column)
            else:
                print(
                    f"[FEATURES][ALERT] Missing group column '{group_column}'; "
                    "group-aware splitting is disabled."
                )
                group_column = None

        # Determine which classification targets are already categorical.
        classification_cfg = params.get("classification", {}) or {}
        configured_class_targets = classification_cfg.get("targets")
        if configured_class_targets in (None, [], "all", "ALL", "*"):
            configured_class_targets = list(target_names)
        elif isinstance(configured_class_targets, str):
            configured_class_targets = [configured_class_targets]
        categorical_targets = {
            target for target in target_names
            if target in configured_class_targets
            and _classification_target_mode(params, target, df[target]) == "categorical"
        }
        if verbose and categorical_targets:
            print(f"[FEATURES] Existing categorical target(s): {sorted(categorical_targets)}")

        # Convert numeric predictors/continuous targets to numeric, but preserve
        # direct class labels (which may be numeric codes or strings).
        numeric_base = required_columns.difference({group_column}) if group_column else set(required_columns)
        numeric_required = sorted(numeric_base.difference(categorical_targets))
        if verbose:
            print("[FEATURES] Converting columns to numeric...")
        
        # Use tqdm only if available and verbose
        iterable = numeric_required
        if HAS_TQDM and verbose:
            iterable = tqdm(numeric_required, desc="Converting columns")
        
        for column in iterable:
            _validate_numeric_conversion(df, [column])
            df[column] = pd.to_numeric(df[column], errors="coerce")
        
        # Remove rows with missing values
        before = len(df)
        df = df.dropna(subset=sorted(set(numeric_required) | set(categorical_targets))).reset_index(drop=True)
        if before != len(df):
            print(f"[FEATURES] Removed {before - len(df)} rows with missing required values.")


        # Validate the sampling-location group after model-column cleaning.
        if group_column:
            group_values = df[group_column].astype("string").str.strip()
            invalid_group = (
                group_values.isna()
                | group_values.eq("")
                | group_values.str.lower().isin({"nan", "none", "null"})
            )
            if invalid_group.any():
                bad_rows = df.index[invalid_group].tolist()
                raise ValueError(
                    f"Group column '{group_column}' contains "
                    f"{int(invalid_group.sum())} missing/blank values "
                    f"(cleaned row indices: {bad_rows[:20]}). "
                    "Every sample must have a valid Location_No/group ID."
                )
            df[group_column] = group_values
        
        if len(df) < 20:
            raise ValueError("The cleaned dataset is too small for an 80/20 holdout workflow.")
        
        indices = np.arange(len(df), dtype=np.int64)
        bins_used = 0
        labels = None
        class_lower = None
        class_upper = None
        boundary_source = None
        
        # Perform split
        if verbose:
            print("[FEATURES] Creating split...")
        
        if group_column:
            groups = df[group_column].astype(str).to_numpy()
            splitter = GroupShuffleSplit(
                n_splits=1, test_size=test_ratio, random_state=random_state
            )
            train_pos, test_pos = next(splitter.split(indices, groups=groups))
            train_idx, test_idx = indices[train_pos], indices[test_pos]
            
            if set(groups[train_idx]).intersection(groups[test_idx]):
                raise RuntimeError("Group leakage detected between development and test sets.")
            
            split_method = (
                "fixed_location_group_aware_train_test"
                if str(group_column) == str(location_column)
                else "fixed_group_aware_train_test"
            )
        else:
            # Prepare stratification labels.
            if stratification_method == "ac_quantile":
                if stratification_target in categorical_targets:
                    labels, direct_names = _direct_class_labels(df[stratification_target].to_numpy())
                    bins_used = int(len(direct_names))
                    print(
                        f"[FEATURES] '{stratification_target}' is an existing categorical target; "
                        "using its class labels for stratification instead of quantile bins."
                    )
                else:
                    labels, bins_used = _make_quantile_labels(
                        df[stratification_target].to_numpy(dtype=float),
                        requested_bins,
                        test_ratio,
                        min_samples_per_bin=4
                    )
                if labels is None:
                    raise ValueError(
                        "Could not create valid quantile bins for target "
                        f"'{stratification_target}'. Reduce ac_quantile_bins "
                        "or check the target distribution."
                    )
                split_method = (
                    f"fixed_class_stratified_{stratification_target}_train_test"
                    if stratification_target in categorical_targets
                    else f"fixed_quantile_stratified_{stratification_target}_q{bins_used}_train_test"
                )

            elif stratification_method == "ac_class":
                if stratification_target in categorical_targets:
                    labels, direct_names = _direct_class_labels(df[stratification_target].to_numpy())
                    counts = np.bincount(labels, minlength=len(direct_names))
                    if np.any(counts < 2):
                        raise ValueError(
                            "Each existing classification class needs at least two samples "
                            f"for target '{stratification_target}'; counts={counts.tolist()}."
                        )
                    boundary_source = "existing_labels"
                    print(
                        f"[FEATURES] Existing classes for {stratification_target}: "
                        f"{direct_names}; counts={counts.tolist()}."
                    )
                    split_method = f"fixed_class_stratified_{stratification_target}_train_test"
                else:
                    target_values = df[stratification_target].to_numpy(dtype=float)
                    class_lower, class_upper, boundary_source = _resolve_class_boundaries(
                        params, stratification_target, target_values
                    )
                    print(
                        "[FEATURES] Classification boundaries for "
                        f"{stratification_target}: lower={class_lower:.6g}, "
                        f"upper={class_upper:.6g} ({boundary_source})."
                    )
                    labels = target_to_three_classes(target_values, class_lower, class_upper)
                    counts = np.bincount(labels, minlength=3)
                    if np.any(counts < 5):
                        raise ValueError(
                            "Each threshold-derived classification class needs at least five samples "
                            f"for target '{stratification_target}'; counts={counts.tolist()}, "
                            f"boundaries=({class_lower:.6g}, {class_upper:.6g})."
                        )
                    split_method = f"fixed_three_class_stratified_{stratification_target}_train_test"

            elif stratification_method in {"none", "random"}:
                split_method = "fixed_random_train_test"
            else:
                raise ValueError(
                    "stratification_method must be ac_quantile, ac_class, none, or random. For categorical targets, ac_quantile/ac_class automatically use existing labels."
                )
            
            # Perform stratified split
            train_idx, test_idx = train_test_split(
                indices,
                test_size=test_ratio,
                random_state=random_state,
                shuffle=True,
                stratify=labels,
            )
            train_idx = np.asarray(train_idx, dtype=np.int64)
            test_idx = np.asarray(test_idx, dtype=np.int64)
        
        # Validate split quality
        _validate_split_coverage(train_idx, test_idx, df)
        _validate_split_quality(train_idx, test_idx, df, labels)
        
        # Save split files
        if verbose:
            print("[FEATURES] Saving split files...")
        
        target_items = feature_sets.items()
        if HAS_TQDM and verbose:
            target_items = tqdm(feature_sets.items(), desc="Processing targets")
        
        for target, definitions in target_items:
            for feature_set, columns in definitions.items():
                _save_split_files(
                    df, train_idx, test_idx, target, feature_set, columns, out_dir
                )
        
        # Save group information if present
        if group_column:
            groups = df[group_column].astype(str).to_numpy()
            joblib.dump(groups[train_idx], out_dir / "groups_train.joblib")
            joblib.dump(groups[test_idx], out_dir / "groups_test.joblib")
        
        # Create split info
        if verbose:
            print("[FEATURES] Creating split information...")
        
        split_info: dict[str, Any] = {
            "created_at": datetime.now().isoformat(),
            "code_version": "1.0.1",  # Could be imported from package
            "dependencies": _get_dependencies(),
            "split_method": split_method,
            "split_mode": effective_mode,
            "split_mode_configured": configured_mode,
            "random_state": random_state,
            "train_ratio_requested": train_ratio,
            "test_ratio_requested": test_ratio,
            "train_size": int(len(train_idx)),
            "test_size": int(len(test_idx)),
            "train_ratio_actual": float(len(train_idx) / len(df)),
            "test_ratio_actual": float(len(test_idx) / len(df)),
            "validation_split_used": False,
            "cv_strategy": (
                f"Group-aware CV on development data (n_splits={cv_splits})"
                if group_column
                else f"RepeatedStratifiedKFold(n_splits={cv_splits}, n_repeats={cv_repeats})"
            ),
            "model_selection_method": (
                "Group-aware cross-validation on development data only"
                if group_column
                else "Repeated cross-validation on development data only"
            ),
            "test_policy": "Independent holdout; excluded from tuning and ranking",
            "stratification_method": stratification_method,
            "stratification_target": stratification_target,
            "classification_boundaries": (
                {
                    "target": stratification_target,
                    "lower": float(class_lower),
                    "upper": float(class_upper),
                    "source": boundary_source,
                }
                if class_lower is not None and class_upper is not None
                else None
            ),
            "ac_quantile_bins_requested": requested_bins if stratification_method == "ac_quantile" else None,
            "ac_quantile_bins_used": bins_used if stratification_method == "ac_quantile" else None,
            "group_column": group_column,
            "grouping_policy": (
                "Keep configured groups intact during splitting/CV; "
                "BH and BH_location are metadata only"
                if group_column
                else "Use explicitly configured group column"
            ) if group_column else None,
            "bh_column": bh_column if bh_column in df.columns else None,
            "location_column": location_column if location_column in df.columns else None,
            "n_unique_groups": (
                int(df[group_column].astype(str).nunique())
                if group_column else None
            ),
            "features_are_scaled": False,
            "preprocessing_policy": "Fit preprocessing only inside CV/model pipelines.",
            "feature_scaling_notes": "Features are unscaled. Scale within CV folds to prevent leakage.",
            "train_indices": train_idx.tolist(),
            "test_indices": test_idx.tolist(),
            "dataset_size": len(df),
        }
        
        # Record group separation for reproducibility.
        if group_column:
            groups_all = df[group_column].astype(str).to_numpy()
            train_groups = sorted(set(groups_all[train_idx]))
            test_groups = sorted(set(groups_all[test_idx]))
            overlap_groups = sorted(set(train_groups).intersection(test_groups))

            if overlap_groups:
                raise RuntimeError(
                    f"Group leakage detected after splitting: {overlap_groups}"
                )

            split_info["group_split_summary"] = {
                "group_column": str(group_column),
                "n_groups_total": int(pd.Series(groups_all).nunique()),
                "n_groups_development": int(len(train_groups)),
                "n_groups_test": int(len(test_groups)),
                "overlap_count": 0,
                "development_groups": train_groups,
                "test_groups": test_groups,
            }

            if str(group_column) == str(location_column):
                split_info["location_group_validation"] = {
                    "enabled": True,
                    "location_column": str(location_column),
                    "n_locations_total": int(pd.Series(groups_all).nunique()),
                    "n_locations_development": int(len(train_groups)),
                    "n_locations_test": int(len(test_groups)),
                    "location_overlap_count": 0,
                    "bh_used_for_grouping": False,
                    "bh_location_used_for_grouping": False,
                }

        # Add distribution and class summaries for the actual target.
        if stratification_target in df.columns:
            if stratification_target in categorical_targets:
                direct_codes, direct_names = _direct_class_labels(df[stratification_target].to_numpy())
                split_info["target_class_boundaries"] = None
                split_info["target_class_mode"] = "existing_labels"
                split_info["target_class_names"] = direct_names
                split_info["target_class_counts"] = {
                    "full": {direct_names[i]: int(np.sum(direct_codes == i)) for i in range(len(direct_names))},
                    "development": {direct_names[i]: int(np.sum(direct_codes[train_idx] == i)) for i in range(len(direct_names))},
                    "test": {direct_names[i]: int(np.sum(direct_codes[test_idx] == i)) for i in range(len(direct_names))},
                }
                split_info["target_distribution_summary"] = {
                    "target": stratification_target,
                    "mode": "categorical",
                    "n_classes": int(len(direct_names)),
                }
            else:
                target_values = df[stratification_target].to_numpy(dtype=float)
                report_lower, report_upper, report_source = _resolve_class_boundaries(
                    params, stratification_target, target_values
                )
                split_info["target_class_boundaries"] = {
                    "target": stratification_target, "lower": float(report_lower),
                    "upper": float(report_upper), "source": report_source,
                }
                split_info["target_class_mode"] = "threshold"
                split_info["target_class_counts"] = {
                    "full": _class_counts(target_values, report_lower, report_upper),
                    "development": _class_counts(target_values[train_idx], report_lower, report_upper),
                    "test": _class_counts(target_values[test_idx], report_lower, report_upper),
                }
                split_info["target_distribution_summary"] = {
                    "target": stratification_target,
                    "full": _numeric_summary(target_values),
                    "development": _numeric_summary(target_values[train_idx]),
                    "test": _numeric_summary(target_values[test_idx]),
                }

            # Backward-compatible names when the target is AC.
            if stratification_target == "AC":
                split_info["ac_class_counts"] = dict(
                    split_info["target_class_counts"]
                )
                split_info["ac_distribution_summary"] = {
                    key: value
                    for key, value in split_info[
                        "target_distribution_summary"
                    ].items()
                    if key != "target"
                }

        # ------------------------------------------------------------
        # Three-class non-parametric analysis
        # ------------------------------------------------------------
        # Reviewer-facing policy: do not use a single Mann-Whitney test as if
        # it represented all three activity classes. First test all classes
        # jointly with Kruskal-Wallis; if significant, run the three explicit
        # pairwise Mann-Whitney U comparisons with Holm correction.
        analysis_target = stratification_target
        if analysis_target in df.columns and analysis_target not in categorical_targets:
            analysis_values = df[analysis_target].to_numpy(dtype=float)
            analysis_lower, analysis_upper, analysis_boundary_source = (
                _resolve_class_boundaries(params, analysis_target, analysis_values)
            )
            analysis_features = sorted({
                column
                for definitions in feature_sets.values()
                for columns in definitions.values()
                for column in columns
                if column in df.columns and column != analysis_target
            })
            kw_df, pairwise_df = _activity_class_nonparametric_tests(
                df=df,
                target=analysis_target,
                lower=analysis_lower,
                upper=analysis_upper,
                feature_columns=analysis_features,
            )
            kw_df.to_csv(out_dir / "activity_class_kruskal_wallis.csv", index=False)
            pairwise_df.to_csv(out_dir / "activity_class_pairwise_holm.csv", index=False)
            split_info["activity_class_statistical_analysis"] = {
                "target": analysis_target,
                "class_boundaries": {
                    "lower": float(analysis_lower),
                    "upper": float(analysis_upper),
                    "source": analysis_boundary_source,
                },
                "omnibus_test": "Kruskal-Wallis across class_0, class_1, and class_2",
                "posthoc_test": "Two-sided Mann-Whitney U for class_0-vs-class_1, class_0-vs-class_2, and class_1-vs-class_2",
                "multiple_testing_correction": "Holm correction across the three pairwise comparisons within each feature",
                "posthoc_policy": "Pairwise tests are reported only when the feature-level Kruskal-Wallis p-value is < 0.05",
                "alpha": 0.05,
                "kruskal_wallis_file": "activity_class_kruskal_wallis.csv",
                "pairwise_file": "activity_class_pairwise_holm.csv",
            }
            if verbose:
                print(
                    "[FEATURES] Statistical analysis: Kruskal-Wallis across "
                    "all three activity classes, followed (when significant) "
                    "by Holm-corrected pairwise Mann-Whitney U tests."
                )

        # Save split info
        with (out_dir / "split_info.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(split_info, handle, sort_keys=False)
        
        # Print summary
        print(f"[FEATURES] Successfully created split: {split_method}")
        if configured_mode == "random":
            print(
                "[FEATURES] Note: split.mode='random' was normalized to a fixed "
                "reproducible holdout using random_state."
            )
        print(f"[FEATURES] Development={len(train_idx)}, test={len(test_idx)}")
        if group_column:
            groups_all = df[group_column].astype(str).to_numpy()
            print(
                f"[FEATURES] Group column='{group_column}': "
                f"development groups={len(set(groups_all[train_idx]))}, "
                f"test groups={len(set(groups_all[test_idx]))}, overlap=0"
            )
        print("[FEATURES] No validation files were created.")
        
    except Exception as e:
        # Clean up partial files on error
        _cleanup_partial_files(out_dir)
        raise RuntimeError(f"Split creation failed: {e}") from e


def main() -> None:
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Create a fixed development/test split for repeated-CV evaluation."
    )
    parser.add_argument("--input", required=True, help="Input CSV file path")
    parser.add_argument("--out", required=True, help="Output directory path")
    parser.add_argument("--params", default="params.yaml", help="Parameters YAML file path")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    args = parser.parse_args()
    run(args.input, args.out, args.params, args.verbose)


if __name__ == "__main__":
    main()