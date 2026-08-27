# src/Multicollinearity.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
import argparse
import re
import unicodedata

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml

from scipy.stats import mannwhitneyu
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# -----------------------------------------------------------------------------
# Configuration helpers
# -----------------------------------------------------------------------------
def _load_params(path: str) -> Dict[str, Any]:
    """Load params.yaml and require a mapping at the YAML root."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Parameter file not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        params = yaml.safe_load(f) or {}

    if not isinstance(params, dict):
        raise ValueError("The YAML root must be a mapping.")

    return params


def _normalize_column_name(value: str) -> str:
    """Normalize a column label for robust YAML-to-CSV matching."""
    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("\ufeff", "").strip().casefold()
    # Treat common separators and punctuation as insignificant for matching.
    return re.sub(r"[^a-z0-9]+", "", text)


def _find_case_insensitive(name: str, columns: List[str]) -> str | None:
    """Return the CSV column matching ``name`` using tolerant normalization."""
    wanted_exact = unicodedata.normalize("NFKC", str(name)).replace("\ufeff", "").strip().casefold()
    wanted_norm = _normalize_column_name(name)

    # Prefer an exact case-insensitive match first.
    for col in columns:
        current = unicodedata.normalize("NFKC", str(col)).replace("\ufeff", "").strip().casefold()
        if current == wanted_exact:
            return col

    # Then allow harmless formatting differences such as spaces, '_' and '-'.
    normalized_matches = [col for col in columns if _normalize_column_name(col) == wanted_norm]
    if len(normalized_matches) == 1:
        return normalized_matches[0]
    if len(normalized_matches) > 1:
        print(f"[WARN] Ambiguous normalized match for '{name}': {normalized_matches}")
    return None


def _load_feature_sets_from_params(
    params: Dict[str, Any],
    csv_columns: List[str],
) -> List[Tuple[str, str, List[str]]]:
    """
    Read targets and feature sets automatically from params.yaml.

    Expected structure:

    TARGETS:
      AC:
        fs1: [Ucs_class, PI/FF]
        fs2: [Ucs_class, PI, FF]
        fs3: [Ucs_class, LL, PL, FF]
        fs4: [Ucs_class, PI]

    Returns:
        [
            (target_column, "AC_fs1", [feature columns...]),
            ...
        ]
    """
    targets_cfg = params.get("TARGETS", params.get("targets"))

    if not isinstance(targets_cfg, dict) or not targets_cfg:
        raise ValueError(
            "No non-empty TARGETS mapping was found in params.yaml."
        )

    feature_sets: List[Tuple[str, str, List[str]]] = []

    for configured_target, fs_dict in targets_cfg.items():
        if not isinstance(fs_dict, dict) or not fs_dict:
            print(
                f"[WARN] Target '{configured_target}' has no valid "
                "feature-set mapping; skipped."
            )
            continue

        target_col = _find_case_insensitive(
            str(configured_target),
            csv_columns,
        )

        if target_col is None:
            print(
                f"[WARN] Target '{configured_target}' is defined in params.yaml "
                "but absent from the CSV; its feature sets are skipped."
            )
            continue

        for fs_name, configured_features in fs_dict.items():
            display_name = f"{configured_target}_{fs_name}"

            if not isinstance(configured_features, (list, tuple)):
                print(
                    f"[SKIP] {display_name}: feature definition must be a list."
                )
                continue

            resolved_features: List[str] = []
            missing_features: List[str] = []

            for feature in configured_features:
                actual = _find_case_insensitive(
                    str(feature),
                    csv_columns,
                )

                if actual is None:
                    missing_features.append(str(feature))
                elif actual not in resolved_features:
                    resolved_features.append(actual)

            if missing_features:
                print(
                    f"[WARN] {display_name}: missing configured features "
                    f"{missing_features}; continuing with available features "
                    f"{resolved_features}."
                )

            if not resolved_features:
                print(
                    f"[SKIP] {display_name}: none of its configured features "
                    "exist in the CSV."
                )
                continue

            feature_sets.append(
                (target_col, display_name, resolved_features)
            )

    if not feature_sets:
        configured_targets = [str(k) for k in targets_cfg.keys()]
        configured_features = []
        for fs_dict in targets_cfg.values():
            if isinstance(fs_dict, dict):
                for values in fs_dict.values():
                    if isinstance(values, (list, tuple)):
                        configured_features.extend(str(v) for v in values)
        configured_features = list(dict.fromkeys(configured_features))

        target_status = {
            t: _find_case_insensitive(t, csv_columns)
            for t in configured_targets
        }
        feature_status = {
            f: _find_case_insensitive(f, csv_columns)
            for f in configured_features
        }

        print("[WARN] No usable target/feature-set combinations were found.")
        print(f"[WARN] Configured target matches: {target_status}")
        print(f"[WARN] Configured feature matches: {feature_status}")
        print(f"[WARN] CSV columns ({len(csv_columns)}): {csv_columns}")
        print("[WARN] Nothing will be analyzed, but no exception is raised.")

    return feature_sets


def _quote_formula_name(name: str) -> str:
    """
    Quote a column name for statsmodels/patsy formulas.

    This is required for names such as PI/FF because '/' has a special
    meaning in formula syntax.
    """
    escaped = str(name).replace("\\", "\\\\").replace('"', '\\"')
    return f'Q("{escaped}")'


# -----------------------------------------------------------------------------
# Main analysis
# -----------------------------------------------------------------------------
def main(
    data_path: str,
    out_path: str,
    params_path: str = "params.yaml",
) -> None:

    # -------------------------------------------------------------------------
    # Load dataset
    # -------------------------------------------------------------------------
    file_path = Path(data_path)
    df = pd.read_csv(file_path)

    # Clean column labels.
    df.columns = [unicodedata.normalize("NFKC", str(c)).replace("\ufeff", "").strip() for c in df.columns]

    # Replace infinities and remove fully empty rows.
    df = (
        df.replace([np.inf, -np.inf], np.nan)
        .dropna(how="all")
    )

    # -------------------------------------------------------------------------
    # Read target(s) + feature sets automatically from params.yaml
    # -------------------------------------------------------------------------
    params = _load_params(params_path)

    feature_sets = _load_feature_sets_from_params(
        params=params,
        csv_columns=list(df.columns),
    )

    if not feature_sets:
        print(
            "[DONE] No usable configured target/feature combinations "
            "were available; exiting cleanly."
        )
        return

    targets = list(
        dict.fromkeys(
            target for target, _, _ in feature_sets
        )
    )

    all_features = list(
        dict.fromkeys(
            feature
            for _, _, features in feature_sets
            for feature in features
        )
    )

    print(f"[INFO] Parameters loaded from: {params_path}")
    print(f"[INFO] Target column(s): {targets}")
    print(f"[INFO] Feature columns: {all_features}")
    print(
        "[INFO] Feature sets: "
        f"{[fs_name for _, fs_name, _ in feature_sets]}"
    )

    # -------------------------------------------------------------------------
    # Output containers
    # -------------------------------------------------------------------------
    results_reg = []
    results_mw = []
    results_vif = []
    results_anova = []

    out_dir = Path(out_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Loop over feature sets exactly as defined in params.yaml
    # -------------------------------------------------------------------------
    for target_name, fs_name, features in feature_sets:

        # Convert target and features together so X/y always remain aligned.
        model_data = df[[target_name] + features].copy()

        for col in [target_name] + features:
            model_data[col] = pd.to_numeric(
                model_data[col],
                errors="coerce",
            )

        model_data = (
            model_data
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        if model_data.empty:
            print(
                f"[SKIP] {fs_name}: no complete numeric rows."
            )
            continue

        X = model_data[features]
        y = model_data[target_name]

        print(
            f"[PROCESSING] {fs_name} | "
            f"Target={target_name} | "
            f"Features={features} | "
            f"n={len(model_data)}"
        )

        # ---------------------------------------------------------------------
        # Regression: Linear Regression and GPR
        # ---------------------------------------------------------------------
        try:
            kernel = (
                ConstantKernel()
                * RBF()
                + WhiteKernel()
            )

            gpr = GaussianProcessRegressor(
                kernel=kernel,
                random_state=0,
            )
            gpr.fit(X, y)

            y_pred_gpr = gpr.predict(X)
            r2_gpr = r2_score(y, y_pred_gpr)

            lr = LinearRegression()
            lr.fit(X, y)

            y_pred_lr = lr.predict(X)
            r2_lr = r2_score(y, y_pred_lr)

            results_reg.append(
                {
                    "FeatureSet": fs_name,
                    "Target": target_name,
                    "n": len(model_data),
                    "R2_LR": r2_lr,
                    "R2_GPR": r2_gpr,
                }
            )

        except Exception as exc:
            print(
                f"[WARN] Regression failed for {fs_name}: {exc}"
            )
            y_pred_gpr = None

        # ---------------------------------------------------------------------
        # Mann-Whitney U
        # Split by the median of the YAML-defined target
        # ---------------------------------------------------------------------
        try:
            target_median = model_data[target_name].median()

            group_low = model_data[
                model_data[target_name] <= target_median
            ]
            group_high = model_data[
                model_data[target_name] > target_median
            ]

            for feat in features:
                low_values = group_low[feat].dropna()
                high_values = group_high[feat].dropna()

                if low_values.empty or high_values.empty:
                    print(
                        f"[WARN] Mann-Whitney skipped for "
                        f"{fs_name}/{feat}: empty group."
                    )
                    continue

                u_stat, p_val = mannwhitneyu(
                    low_values,
                    high_values,
                    alternative="two-sided",
                )

                n1 = len(low_values)
                n2 = len(high_values)

                mean_u = n1 * n2 / 2
                std_u = np.sqrt(
                    n1 * n2 * (n1 + n2 + 1) / 12
                )

                z_val = (
                    np.nan
                    if std_u == 0
                    else (u_stat - mean_u) / std_u
                )

                results_mw.append(
                    {
                        "FeatureSet": fs_name,
                        "Target": target_name,
                        "Feature": feat,
                        "U": u_stat,
                        "Z": z_val,
                        "p": p_val,
                        "n_low": n1,
                        "n_high": n2,
                    }
                )

        except Exception as exc:
            print(
                f"[WARN] Mann-Whitney failed for {fs_name}: {exc}"
            )

        # ---------------------------------------------------------------------
        # ANOVA
        # Q("...") makes names such as PI/FF safe.
        # ---------------------------------------------------------------------
        try:
            target_formula = _quote_formula_name(target_name)

            feature_formula = " + ".join(
                _quote_formula_name(feat)
                for feat in features
            )

            formula = (
                f"{target_formula} ~ {feature_formula}"
            )

            model_ols = ols(
                formula,
                data=model_data,
            ).fit()

            anova_table = anova_lm(
                model_ols,
                typ=2,
            )

            anova_table["FeatureSet"] = fs_name
            anova_table["Target"] = target_name

            anova_result = (
                anova_table
                .reset_index()
                .rename(columns={"index": "Feature"})
            )

            # Convert Q("PI/FF") back to readable names where possible.
            feature_name_map = {
                _quote_formula_name(feat): feat
                for feat in features
            }

            anova_result["Feature"] = (
                anova_result["Feature"]
                .replace(feature_name_map)
            )

            results_anova.append(
                anova_result
            )

        except Exception as exc:
            print(
                f"[WARN] ANOVA failed for {fs_name}: {exc}"
            )

        # ---------------------------------------------------------------------
        # VIF
        # ---------------------------------------------------------------------
        if len(features) > 1:
            try:
                X_const = sm.add_constant(
                    X,
                    has_constant="add",
                )

                for j, feat in enumerate(
                    features,
                    start=1,
                ):
                    try:
                        vif = variance_inflation_factor(
                            X_const.values,
                            j,
                        )
                    except Exception as exc:
                        print(
                            f"[WARN] VIF failed for "
                            f"{fs_name}/{feat}: {exc}"
                        )
                        vif = np.nan

                    results_vif.append(
                        {
                            "FeatureSet": fs_name,
                            "Target": target_name,
                            "Feature": feat,
                            "VIF": vif,
                        }
                    )

            except Exception as exc:
                print(
                    f"[WARN] VIF setup failed for {fs_name}: {exc}"
                )

        # ---------------------------------------------------------------------
        # GPR actual-vs-predicted plot
        # ---------------------------------------------------------------------
        if y_pred_gpr is not None:
            try:
                plt.figure(figsize=(7, 5))

                plt.scatter(
                    y,
                    y_pred_gpr,
                    alpha=0.6,
                    label="GPR Predictions",
                )

                min_value = min(
                    y.min(),
                    np.min(y_pred_gpr),
                )
                max_value = max(
                    y.max(),
                    np.max(y_pred_gpr),
                )

                plt.plot(
                    [min_value, max_value],
                    [min_value, max_value],
                    "r--",
                    label="Perfect Fit",
                )

                plt.title(
                    f"GPR Performance - {fs_name}\n"
                    f"({target_name} vs {', '.join(features)})"
                )

                plt.xlabel(
                    f"Actual {target_name}"
                )
                plt.ylabel(
                    f"Predicted {target_name}"
                )

                plt.legend()
                plt.grid(alpha=0.3)
                plt.tight_layout()

                plt.savefig(
                    out_dir / f"GPR_plot_{fs_name}.png",
                    dpi=250,
                    bbox_inches="tight",
                )

                plt.close()

            except Exception as exc:
                print(
                    f"[WARN] Plot failed for {fs_name}: {exc}"
                )

    # -------------------------------------------------------------------------
    # Save CSV outputs
    # -------------------------------------------------------------------------
    if results_reg:
        pd.DataFrame(
            results_reg
        ).to_csv(
            out_dir / "Regression_Results.csv",
            index=False,
        )

    if results_mw:
        pd.DataFrame(
            results_mw
        ).to_csv(
            out_dir / "MannWhitney_Results.csv",
            index=False,
        )

    if results_vif:
        pd.DataFrame(
            results_vif
        ).to_csv(
            out_dir / "VIF_Results.csv",
            index=False,
        )

    if results_anova:
        pd.concat(
            results_anova,
            ignore_index=True,
        ).to_csv(
            out_dir / "ANOVA_Results.csv",
            index=False,
        )

    # -------------------------------------------------------------------------
    # Merge CSV outputs into one Excel workbook
    # -------------------------------------------------------------------------
    excel_path = (
        out_dir
        / "Multicollinearity_Results.xlsx"
    )

    csv_files = sorted(
        out_dir.glob("*.csv")
    )

    if not csv_files:
        print(
            "[WARN] No CSV files found to merge into Excel."
        )
    else:
        with pd.ExcelWriter(
            excel_path,
            engine="openpyxl",
        ) as writer:

            for csv_file in csv_files:
                try:
                    df_csv = pd.read_csv(
                        csv_file
                    )

                    sheet_name = (
                        csv_file.stem[:31]
                    )

                    df_csv.to_excel(
                        writer,
                        sheet_name=sheet_name,
                        index=False,
                    )

                    print(
                        f"[INFO] Added {csv_file.name} "
                        f"-> sheet {sheet_name}"
                    )

                except Exception as exc:
                    print(
                        f"[WARN] Could not add "
                        f"{csv_file.name}: {exc}"
                    )

        print(
            f"[INFO] All CSVs merged into {excel_path}"
        )

    print("[DONE] Multicollinearity analysis completed.")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=(
            "Multicollinearity, VIF, Mann-Whitney, ANOVA, LR, "
            "and GPR analysis using targets and feature sets "
            "defined automatically in params.yaml."
        )
    )

    ap.add_argument(
        "--data",
        required=True,
        help="Path to input CSV file.",
    )

    ap.add_argument(
        "--out",
        required=True,
        help="Output directory for results.",
    )

    ap.add_argument(
        "--params",
        default="params.yaml",
        help=(
            "Path to params.yaml. Targets and feature sets are "
            "read automatically from TARGETS."
        ),
    )

    args = ap.parse_args()

    main(
        args.data,
        args.out,
        args.params,
    )
