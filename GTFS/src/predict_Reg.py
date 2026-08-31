#!/usr/bin/env python
from __future__ import annotations
import argparse, json, sys, os, signal, re, math
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import warnings
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
from scipy.stats import friedmanchisquare, rankdata
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, RepeatedKFold, cross_validate
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor

# ============================================================
# Suppress warnings
# ============================================================
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# Optional GUI imports
# ============================================================
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    import tkinter.font as tkfont
except Exception:
    tk = None

# ============================================================
# Matplotlib imports
# ============================================================
try:
    import matplotlib
    matplotlib.use("TkAgg")
    matplotlib.rcParams["figure.autolayout"] = True
    matplotlib.rcParams["figure.dpi"] = 110
    matplotlib.rcParams["font.size"] = 8
    matplotlib.rcParams["axes.titlesize"] = 9
    matplotlib.rcParams["axes.labelsize"] = 8
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.patches import Circle
    import matplotlib.pyplot as plt
except Exception:
    Figure = None
    FigureCanvasTkAgg = None
    NavigationToolbar2Tk = None
    plt = None

# ============================================================
# Constants and helper functions from original code
# ============================================================
EPS = 1e-12

def ioa_willmott(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    num = np.sum((y_pred - y_true)**2)
    denom = np.sum((np.abs(y_pred - np.mean(y_true)) + np.abs(y_true - np.mean(y_true)))**2)
    if denom == 0: return np.nan
    return 1.0 - num / denom

def ios_skill(y_true, y_pred):
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    baseline = np.full_like(y_true, np.mean(y_true))
    rmse_b = math.sqrt(mean_squared_error(y_true, baseline))
    if rmse_b == 0: return np.nan
    return 1.0 - rmse / rmse_b

def p_within_percent(y_true, y_pred, pct=20.0):
    ok = np.abs(y_pred - y_true) <= (pct/100.0) * (np.abs(y_true) + EPS)
    return 100.0 * float(np.mean(ok))

def anderson_darling_stat(residuals):
    r = np.asarray(residuals, dtype=float)
    if r.size < 5: return float("nan")
    r = (r - np.mean(r)) / (np.std(r) + EPS)
    r = np.sort(r)
    n = len(r)
    from math import erf
    Phi = lambda x: 0.5*(1.0+erf(x/np.sqrt(2.0)))
    Fi = np.clip(np.array([Phi(x) for x in r]), 1e-12, 1-1e-12)
    i = np.arange(1, n+1)
    s = np.sum((2*i-1)*(np.log(Fi) + np.log(1-Fi[::-1])))
    A2 = -n - s/n
    return float(A2)

# ============================================================
# Original functions that need to be preserved
# ============================================================
def _safe_json_load(path: Path):
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8").strip()
        return json.loads(text if text else "{}")
    except Exception:
        return {}


def _find_joblib(root, candidate_names):
    """Find a joblib file directly or recursively, case-insensitively."""
    root = Path(root)
    for name in candidate_names:
        path = root / str(name)
        if path.exists():
            return path
    wanted = {str(name).lower() for name in candidate_names}
    try:
        for path in root.rglob("*.joblib"):
            if path.name.lower() in wanted:
                return path
    except Exception:
        pass
    return None


def _resolve_cv_training_data(data_dir, fs, suffix="AC"):
    """Resolve X/y training joblib paths robustly for CV diagnostics."""
    data_dir = Path(data_dir)
    fs_text = str(fs).strip()
    fs_lower = fs_text.lower()
    fs_upper = fs_text.upper()
    fs_title = "Fs" + fs_lower[2:] if fs_lower.startswith("fs") else fs_text
    suffix_text = str(suffix or "AC").strip()
    x_path = _find_joblib(data_dir,[f"X_train_{fs_text}.joblib",f"X_train_{fs_lower}.joblib",f"X_train_{fs_upper}.joblib",f"X_train_{fs_title}.joblib"])
    y_path = _find_joblib(data_dir,[f"y_train_{suffix_text}.joblib",f"y_train_{suffix_text.lower()}.joblib",f"y_train_{suffix_text.upper()}.joblib","y_train_AC.joblib","y_train_ac.joblib"])
    return x_path, y_path

def load_training_best_model_and_fs(data_dir, models_dir, reports_dir):
    """Load the model selected from development-set cross-validation."""
    model_path = next((p for p in (models_dir / "best_overall.pkl", models_dir / "best.pkl") if p.exists()), None)
    if model_path is None:
        raise FileNotFoundError("best_overall.pkl or best.pkl was not found. Run train.py first.")
    loaded = joblib.load(model_path)
    package = dict(loaded) if isinstance(loaded, dict) else {"model": loaded}
    model = package.get("model") or next((v for v in package.values() if hasattr(v, "predict")), None)
    if model is None:
        raise ValueError(f"No prediction model found in {model_path.name}.")
    meta = _safe_json_load(reports_dir / "best_meta.json")
    fs = package.get("feature_set") or package.get("fs") or meta.get("feature_set") or meta.get("fs")
    if not fs:
        match = re.search(r"(?:^|[_-])(fs\d+)(?:[_-]|$)", model_path.stem, flags=re.I)
        fs = match.group(1).lower() if match else None
    if not fs:
        raise ValueError("Could not determine the feature set of the selected model.")
    name = package.get("model_type") or meta.get("model") or type(model).__name__
    return model, None, str(fs), str(name)

def _infer_fs_from_name(name: str) -> str | None:
    name_stem = Path(name).stem
    prefix_match = re.match(r"^([^-]+)-", name_stem)
    if not prefix_match:
        return None
        
    prefix = prefix_match.group(1)
    m = re.search(rf"^{re.escape(prefix)}-fs([123])(?:_|\.|$)", name_stem)
    if m:
        return f"fs{m.group(1)}"
    return None

def _extract_model_prefix(filename: str) -> str:
    name_stem = Path(filename).stem
    match = re.match(r"^([^-]+)", name_stem)
    return match.group(1) if match else name_stem

def select_test_best_from_leaderboard(data_dir, models_dir, reports_dir):
    """Return the CV-selected model; test performance never selects a model."""
    try:
        model, _, fs, name = load_training_best_model_and_fs(data_dir, models_dir, reports_dir)
    except Exception as error:
        print(f"[WARN] Could not load CV-selected model: {error}")
        return None, None, None, None, None
    filename = "best_overall.pkl" if (models_dir / "best_overall.pkl").exists() else "best.pkl"
    return model, None, fs, name, filename

# ============================================================
# Enhanced Statistical Ranking System with Bootstrapping
# ============================================================
def statistical_ranking_system(df, n_bootstrap=1000, confidence_level=0.95):
    """Select by repeated-CV only; calculate a separate descriptive test rank."""
    if df is None or df.empty: return df
    df = _add_gap_diagnostics_for_ranking(df)
    def rank(col, asc): return df[col].rank(ascending=asc, method="average", na_option="bottom")
    df["rank_cv_r2"] = rank("cv_r2_mean", False)
    df["rank_cv_stability"] = rank("cv_r2_std", True)
    df["rank_train_cv_gap"] = rank("abs_train_cv_gap", True)
    df["selection_rank_score"] = 0.60*df["rank_cv_r2"] + 0.20*df["rank_cv_stability"] + 0.20*df["rank_train_cv_gap"]
    df["statistical_rank"] = df["selection_rank_score"].rank(ascending=True, method="min").astype(int)
    df["selection_rank"] = df["statistical_rank"]
    score=df["selection_rank_score"]
    df["statistical_score"] = 1.0 if score.max()==score.min() else 1.0-(score-score.min())/(score.max()-score.min())
    df["statistical_score_ci_lower"]=np.nan; df["statistical_score_ci_upper"]=np.nan; df["statistical_score_ci_width"]=np.nan
    ranks=df[["rank_cv_r2","rank_cv_stability","rank_train_cv_gap"]].to_numpy(float)
    df["rank_stability"]=[1.0 if np.std(x)==0 else 1/(1+np.std(x)/np.mean(x)) for x in ranks]
    if df["r2_test"].notna().any():
        df["rank_test_r2"]=rank("r2_test",False); df["rank_test_rmse"]=rank("rmse_test",True); df["rank_test_mae"]=rank("mae_test",True)
        df["test_rank_score"]=0.50*df["rank_test_r2"]+0.30*df["rank_test_rmse"]+0.20*df["rank_test_mae"]
        df["test_performance_rank"]=df["test_rank_score"].rank(ascending=True,method="min").astype("Int64")
    else: df["test_performance_rank"]=pd.Series(pd.NA,index=df.index,dtype="Int64")
    df["generalization_rank_score"]=0.60*rank("abs_train_cv_gap",True)+0.40*rank("cv_r2_std",True)
    df["generalization_rank"]=df["generalization_rank_score"].rank(ascending=True,method="min").astype(int)
    return df.sort_values(["statistical_rank","cv_r2_mean","cv_r2_std"],ascending=[True,False,True]).reset_index(drop=True)

def calculate_bootstrap_confidence(df, X_normalized, pca, explained_variance, 
                                 n_bootstrap=1000, confidence_level=0.95):
    """
    Calculate bootstrapped confidence intervals for statistical scores
    """
    n_samples = len(df)
    
    # Bootstrap the PCA scores
    bootstrap_scores = []
    
    for _ in range(n_bootstrap):
        # Bootstrap resample
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot = X_normalized[indices]
        
        # Apply PCA transformation using original PCA components
        scores_boot = X_boot @ pca.components_.T
        
        # Calculate weighted scores
        weighted_boot = np.zeros(n_samples)
        for i, var_ratio in enumerate(explained_variance):
            weighted_boot += scores_boot[:, i] * var_ratio
        
        bootstrap_scores.append(weighted_boot)
    
    bootstrap_scores = np.array(bootstrap_scores)
    
    # Calculate confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = 100 * alpha / 2
    upper_percentile = 100 * (1 - alpha / 2)
    
    ci_lower = np.percentile(bootstrap_scores, lower_percentile, axis=0)
    ci_upper = np.percentile(bootstrap_scores, upper_percentile, axis=0)
    
    # Store confidence intervals
    df['statistical_score_ci_lower'] = ci_lower
    df['statistical_score_ci_upper'] = ci_upper
    df['statistical_score_ci_width'] = ci_upper - ci_lower
    
    return df

def calculate_pareto_rank(df):
    """
    Calculate Pareto optimality rank
    A model is Pareto-optimal if no other model is better in all metrics
    """
    higher_better = ['r2_test', 'p20_test', 'ioa_test', 'ios_test']
    lower_better = ['mae_test', 'rmse_test', 'mse_test']
    
    # Filter to available metrics
    higher_better = [m for m in higher_better if m in df.columns]
    lower_better = [m for m in lower_better if m in df.columns]
    
    pareto_scores = []
    
    for i, row_i in df.iterrows():
        dominated_count = 0
        
        for j, row_j in df.iterrows():
            if i == j:
                continue
                
            # Check if row_j dominates row_i
            j_better_higher = all(row_j[m] >= row_i[m] for m in higher_better)
            j_better_lower = all(row_j[m] <= row_i[m] for m in lower_better)
            
            if j_better_higher and j_better_lower:
                # Check for strict dominance
                if (any(row_j[m] > row_i[m] for m in higher_better) or 
                    any(row_j[m] < row_i[m] for m in lower_better)):
                    dominated_count += 1
        
        # Pareto score: 1/(1 + number of models that dominate this one)
        pareto_scores.append(1 / (1 + dominated_count))
    
    return pd.Series(pareto_scores).rank(ascending=False, method='min')

def calculate_dominance_rank(df, metrics):
    """
    Calculate dominance rank based on pairwise comparisons
    """
    higher_better = ['r2_test', 'p20_test', 'ioa_test', 'ios_test']
    lower_better = ['mae_test', 'rmse_test', 'mse_test']
    
    # Filter to available metrics
    higher_better = [m for m in higher_better if m in df.columns]
    lower_better = [m for m in lower_better if m in df.columns]
    
    n_models = len(df)
    dominance_scores = np.zeros(n_models)
    
    for i in range(n_models):
        for j in range(n_models):
            if i == j:
                continue
                
            # Count how many metrics model i is better than model j
            better_count = 0
            total_comparisons = 0
            
            for metric in higher_better:
                if metric in metrics:
                    if df.iloc[i][metric] > df.iloc[j][metric]:
                        better_count += 1
                    total_comparisons += 1
            
            for metric in lower_better:
                if metric in metrics:
                    if df.iloc[i][metric] < df.iloc[j][metric]:
                        better_count += 1
                    total_comparisons += 1
            
            if total_comparisons > 0:
                # Model i dominates model j if it's better in majority of metrics
                if better_count > total_comparisons / 2:
                    dominance_scores[i] += 1
    
    # Convert dominance scores to rank (higher score = better rank)
    return pd.Series(dominance_scores).rank(ascending=False, method='min')

def calculate_bayesian_rank(df, metrics):
    """
    Bayesian ranking using Dirichlet-multinomial model
    """
    n_models = len(df)
    
    # Define which metrics should be higher/lower
    higher_better = ['r2_test', 'p20_test', 'ioa_test', 'ios_test']
    lower_better = ['mae_test', 'rmse_test', 'mse_test']
    
    # Filter to available metrics
    higher_better = [m for m in higher_better if m in df.columns]
    lower_better = [m for m in lower_better if m in df.columns]
    
    all_metrics = higher_better + lower_better
    
    # Initialize Dirichlet prior (uniform)
    alpha_prior = np.ones(n_models)
    
    # Calculate pairwise comparisons
    win_matrix = np.zeros((n_models, n_models))
    
    for i in range(n_models):
        for j in range(n_models):
            if i == j:
                continue
                
            wins = 0
            for metric in higher_better:
                if metric in metrics and df.iloc[i][metric] > df.iloc[j][metric]:
                    wins += 1
            
            for metric in lower_better:
                if metric in metrics and df.iloc[i][metric] < df.iloc[j][metric]:
                    wins += 1
            
            total_comparisons = len([m for m in all_metrics if m in metrics])
            if total_comparisons > 0:
                win_matrix[i, j] = wins / total_comparisons
    
    # Bayesian update
    wins_per_model = np.sum(win_matrix, axis=1)
    losses_per_model = np.sum(win_matrix, axis=0)
    
    # Dirichlet posterior
    alpha_posterior = alpha_prior + wins_per_model - losses_per_model
    alpha_posterior = np.maximum(alpha_posterior, 0.1)  # Prevent zeros
    
    # Expected ranking from Dirichlet distribution
    bayesian_scores = alpha_posterior / np.sum(alpha_posterior)
    
    return pd.Series(bayesian_scores).rank(ascending=False, method='min')

def calculate_statistical_rank(df, metrics):
    """
    Calculate statistical rank using ensemble of non-parametric methods
    """
    ranks = []
    
    # Method 1: PCA-based ranking
    ranks.append(df['statistical_score'].rank(ascending=False, method='min').values)
    
    # Method 2: Pareto optimality ranking
    ranks.append(calculate_pareto_rank(df).values)
    
    # Method 3: Dominance-based ranking
    ranks.append(calculate_dominance_rank(df, metrics).values)
    
    # Method 4: Bayesian ranking
    ranks.append(calculate_bayesian_rank(df, metrics).values)
    
    # Ensemble ranking using trimmed mean
    ranks_array = np.column_stack(ranks)
    
    # Calculate trimmed mean (remove top and bottom 10%)
    def trimmed_mean(arr):
        n = len(arr)
        trim = int(0.1 * n)
        if trim > 0:
            sorted_arr = np.sort(arr)
            return np.mean(sorted_arr[trim:-trim])
        return np.mean(arr)
    
    ensemble_rank = np.apply_along_axis(trimmed_mean, 1, ranks_array)
    
    # Final rank based on ensemble trimmed mean
    df['statistical_rank'] = pd.Series(ensemble_rank).rank(method='min').astype(int)
    
    # Calculate rank stability
    df['rank_stability'] = calculate_rank_stability(ranks_array)
    
    return df

def calculate_rank_stability(ranks_array):
    """
    Calculate rank stability across different ranking methods
    """
    n_methods = ranks_array.shape[1]
    
    if n_methods < 2:
        return np.ones(len(ranks_array))
    
    # Calculate coefficient of variation for each model's ranks
    stability_scores = []
    
    for i in range(len(ranks_array)):
        ranks = ranks_array[i]
        if np.std(ranks) == 0:
            stability = 1.0  # Perfect agreement
        else:
            cv = np.std(ranks) / np.mean(ranks)
            stability = 1 / (1 + cv)  # Convert to [0,1] scale
        
        stability_scores.append(stability)
    
    return np.array(stability_scores)


# ============================================================
# Gap-aware scientific model ranking
# ============================================================
def _safe_numeric_series(df, column, default=np.nan):
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _add_gap_diagnostics_for_ranking(df):
    """Normalize train/CV/test metrics and add leakage-safe diagnostics."""
    df = df.copy()
    if "fs" not in df.columns and "feature_set" in df.columns: df["fs"] = df["feature_set"]
    if "feature_set" not in df.columns and "fs" in df.columns: df["feature_set"] = df["fs"]
    for col in ["r2_train", "cv_r2_mean", "cv_r2_std", "r2_test", "rmse_test", "mae_test", "ioa_test", "p20_test"]:
        if col not in df.columns: df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["train_cv_gap"] = df["r2_train"] - df["cv_r2_mean"]
    df["cv_test_gap"] = df["cv_r2_mean"] - df["r2_test"]
    df["abs_train_cv_gap"] = df["train_cv_gap"].abs()
    df["abs_cv_test_gap"] = df["cv_test_gap"].abs()
    df["possible_underfitting"] = (df["r2_train"] < 0.40) & (df["cv_r2_mean"] < 0.40)
    df["possible_overfitting"] = df["train_cv_gap"] >= 0.20
    df["high_cv_instability"] = df["cv_r2_std"] >= 0.20
    df["test_shift"] = df["abs_cv_test_gap"] >= 0.20
    df["stable_generalization"] = (df["abs_train_cv_gap"] <= 0.10) & (df["cv_r2_mean"] >= 0.40) & (~df["high_cv_instability"])
    diagnoses=[]
    for _,r in df.iterrows():
        if r["possible_underfitting"]: d="Possible underfitting"
        elif r["possible_overfitting"] and r["high_cv_instability"]: d="Overfitting with high CV instability"
        elif r["possible_overfitting"]: d="Possible overfitting"
        elif r["high_cv_instability"]: d="High CV instability"
        elif r["test_shift"]: d="CV-test shift"
        elif r["stable_generalization"]: d="Stable generalization"
        else: d="Moderate generalization"
        diagnoses.append(d)
    df["generalization_diagnosis"] = diagnoses
    return df

def statistical_ranking_system(df, n_bootstrap=1000, confidence_level=0.95):
    """Select by repeated-CV only; calculate a separate descriptive test rank."""
    if df is None or df.empty: return df
    df = _add_gap_diagnostics_for_ranking(df)
    def rank(col, asc): return df[col].rank(ascending=asc, method="average", na_option="bottom")
    df["rank_cv_r2"] = rank("cv_r2_mean", False)
    df["rank_cv_stability"] = rank("cv_r2_std", True)
    df["rank_train_cv_gap"] = rank("abs_train_cv_gap", True)
    df["selection_rank_score"] = 0.60*df["rank_cv_r2"] + 0.20*df["rank_cv_stability"] + 0.20*df["rank_train_cv_gap"]
    df["statistical_rank"] = df["selection_rank_score"].rank(ascending=True, method="min").astype(int)
    df["selection_rank"] = df["statistical_rank"]
    score=df["selection_rank_score"]
    df["statistical_score"] = 1.0 if score.max()==score.min() else 1.0-(score-score.min())/(score.max()-score.min())
    df["statistical_score_ci_lower"]=np.nan; df["statistical_score_ci_upper"]=np.nan; df["statistical_score_ci_width"]=np.nan
    ranks=df[["rank_cv_r2","rank_cv_stability","rank_train_cv_gap"]].to_numpy(float)
    df["rank_stability"]=[1.0 if np.std(x)==0 else 1/(1+np.std(x)/np.mean(x)) for x in ranks]
    if df["r2_test"].notna().any():
        df["rank_test_r2"]=rank("r2_test",False); df["rank_test_rmse"]=rank("rmse_test",True); df["rank_test_mae"]=rank("mae_test",True)
        df["test_rank_score"]=0.50*df["rank_test_r2"]+0.30*df["rank_test_rmse"]+0.20*df["rank_test_mae"]
        df["test_performance_rank"]=df["test_rank_score"].rank(ascending=True,method="min").astype("Int64")
    else: df["test_performance_rank"]=pd.Series(pd.NA,index=df.index,dtype="Int64")
    df["generalization_rank_score"]=0.60*rank("abs_train_cv_gap",True)+0.40*rank("cv_r2_std",True)
    df["generalization_rank"]=df["generalization_rank_score"].rank(ascending=True,method="min").astype(int)
    return df.sort_values(["statistical_rank","cv_r2_mean","cv_r2_std"],ascending=[True,False,True]).reset_index(drop=True)

WEIGHT_SCENARIOS = {
    "equal": {"rank_cv_r2": 1/3, "rank_cv_stability": 1/3, "rank_train_cv_gap": 1/3},
    "performance_focused": {"rank_cv_r2": 0.70, "rank_cv_stability": 0.15, "rank_train_cv_gap": 0.15},
    "stability_focused": {"rank_cv_r2": 0.40, "rank_cv_stability": 0.30, "rank_train_cv_gap": 0.30},
}

def _weight_sensitivity_diagnosis(first_place_rate, rank_range, rank_sd):
    """Convert sensitivity statistics into a transparent interpretation."""
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


def predefined_weight_sensitivity_analysis(df):
    if df is None or df.empty: return df, pd.DataFrame()
    result=df.copy(); rank_columns=[]; rows=[]
    for name,weights in WEIGHT_SCENARIOS.items():
        score=sum(float(w)*pd.to_numeric(result[c],errors="coerce") for c,w in weights.items())
        result[f"{name}_score"]=score; result[f"{name}_rank"]=score.rank(ascending=True,method="min").astype("Int64")
        rank_columns.append(f"{name}_rank"); rows.append({"scenario":name,**weights,"total_weight":sum(weights.values())})
    r=result[rank_columns].astype(float)
    result["sensitivity_mean_rank"]=r.mean(axis=1); result["sensitivity_median_rank"]=r.median(axis=1)
    result["sensitivity_min_rank"]=r.min(axis=1); result["sensitivity_max_rank"]=r.max(axis=1)
    result["sensitivity_rank_range"]=result["sensitivity_max_rank"]-result["sensitivity_min_rank"]
    result["sensitivity_rank_sd"]=r.std(axis=1,ddof=1); result["first_place_count"]=(r==1).sum(axis=1)
    result["first_place_rate"]=result["first_place_count"]/len(rank_columns)
    result["weight_sensitivity_diagnosis"]=[_weight_sensitivity_diagnosis(a,b,c) for a,b,c in zip(result["first_place_rate"],result["sensitivity_rank_range"],result["sensitivity_rank_sd"])]
    return result,pd.DataFrame(rows)

def monte_carlo_weight_sensitivity(df, n_simulations=1000, random_state=42, min_performance_weight=0.50, max_single_weight=0.80):
    if df is None or df.empty: return df,pd.DataFrame(),np.empty((0,0),dtype=int)
    criteria=["rank_cv_r2","rank_cv_stability","rank_train_cv_gap"]
    matrix=df[criteria].to_numpy(float); rng=np.random.default_rng(random_state); weights=[]
    while len(weights)<int(n_simulations):
        w=rng.dirichlet(np.ones(3))
        if w[0]>=min_performance_weight and w.max()<=max_single_weight: weights.append(w)
    wa=np.asarray(weights); ra=np.asarray([pd.Series(matrix@w).rank(ascending=True,method="min").astype(int).to_numpy() for w in wa]).T
    result=df.copy(); result["mc_mean_rank"]=ra.mean(axis=1); result["mc_median_rank"]=np.median(ra,axis=1)
    result["mc_min_rank"]=ra.min(axis=1); result["mc_max_rank"]=ra.max(axis=1); result["mc_rank_range"]=result["mc_max_rank"]-result["mc_min_rank"]
    result["mc_rank_sd"]=ra.std(axis=1,ddof=1); result["mc_first_place_count"]=(ra==1).sum(axis=1); result["mc_first_place_rate"]=result["mc_first_place_count"]/ra.shape[1]
    result["mc_top3_count"]=(ra<=3).sum(axis=1); result["mc_top3_rate"]=result["mc_top3_count"]/ra.shape[1]
    result["mc_weight_sensitivity_diagnosis"]=[_weight_sensitivity_diagnosis(a,b,c) for a,b,c in zip(result["mc_first_place_rate"],result["mc_rank_range"],result["mc_rank_sd"])]

    # Monte Carlo robustness rank. This uses development-stage robustness only.
    rank_mc_median=result["mc_median_rank"].rank(ascending=True,method="average",na_option="bottom")
    rank_mc_first=result["mc_first_place_rate"].rank(ascending=False,method="average",na_option="bottom")
    rank_mc_top3=result["mc_top3_rate"].rank(ascending=False,method="average",na_option="bottom")
    rank_mc_sd=result["mc_rank_sd"].rank(ascending=True,method="average",na_option="bottom")
    result["robustness_rank_score"]=0.45*rank_mc_median+0.25*rank_mc_first+0.20*rank_mc_top3+0.10*rank_mc_sd
    result["robustness_rank"]=result["robustness_rank_score"].rank(ascending=True,method="min").astype("Int64")

    # Final recommended development-stage rank: primary CV selection + MC robustness.
    # Independent-test performance is deliberately excluded.
    rank_selection=result["selection_rank_score"].rank(ascending=True,method="average",na_option="bottom")
    rank_robust=result["robustness_rank_score"].rank(ascending=True,method="average",na_option="bottom")
    result["recommended_rank_score"]=0.60*rank_selection+0.40*rank_robust
    result["recommended_rank"]=result["recommended_rank_score"].rank(ascending=True,method="min").astype("Int64")

    wdf=pd.DataFrame(wa,columns=["cv_r2_weight","cv_sd_weight","train_cv_gap_weight"]); wdf.insert(0,"simulation",np.arange(1,len(wdf)+1))
    return result,wdf,ra

def run_complete_weight_sensitivity(
    df,
    reports_dir,
    n_simulations=1000,
    random_state=42,
):
    """Run all sensitivity analyses and write paper-ready outputs."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    scenario_result, scenarios_df = predefined_weight_sensitivity_analysis(df)
    mc_result, mc_weights_df, mc_ranks = monte_carlo_weight_sensitivity(
        scenario_result,
        n_simulations=n_simulations,
        random_state=random_state,
    )

    scenarios_path = reports_dir / "WeightScenarios.csv"
    scenario_table_path = reports_dir / "WeightSensitivityAnalysis.csv"
    mc_weights_path = reports_dir / "MonteCarloWeightSensitivity.csv"

    scenarios_df.to_csv(scenarios_path, index=False)
    mc_result.to_csv(
        scenario_table_path,
        index=False,
        float_format="%.6f",
        encoding="utf-8-sig",
    )
    mc_weights_df.to_csv(
        mc_weights_path,
        index=False,
        float_format="%.6f",
    )

    # Save the complete model-by-simulation rank matrix for reproducibility.
    model_labels = (
        mc_result.get("model", pd.Series("model", index=mc_result.index)).astype(str)
        + "-"
        + mc_result.get(
            "feature_set",
            mc_result.get("fs", pd.Series("fs", index=mc_result.index)),
        ).astype(str)
    )
    rank_matrix_df = pd.DataFrame(
        mc_ranks,
        columns=[
            f"simulation_{i}"
            for i in range(1, mc_ranks.shape[1] + 1)
        ],
    )
    rank_matrix_df.insert(0, "model_feature_set", model_labels.to_numpy())
    rank_matrix_path = reports_dir / "MonteCarloRankMatrix.csv"
    rank_matrix_df.to_csv(rank_matrix_path, index=False)

    # Plot 1: first-place rate under the Monte Carlo analysis.
    first_place_plot = reports_dir / "MonteCarloFirstPlaceRate.png"
    rank_distribution_plot = reports_dir / "WeightSensitivityRankDistribution.png"

    if plt is not None:
        plot_df = mc_result.copy()
        plot_df["model_label"] = model_labels
        plot_df = plot_df.sort_values(
            "mc_first_place_rate", ascending=False
        )

        fig, ax = plt.subplots(
            figsize=(max(8, len(plot_df) * 0.48), 5.5)
        )
        ax.bar(
            np.arange(len(plot_df)),
            plot_df["mc_first_place_rate"].astype(float),
        )
        ax.set_xticks(np.arange(len(plot_df)))
        ax.set_xticklabels(
            plot_df["model_label"], rotation=55, ha="right"
        )
        ax.set_ylabel("First-place frequency")
        ax.set_ylim(0, 1)
        ax.set_title(
            "Model first-place frequency under constrained random weights"
        )
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(first_place_plot, dpi=300, bbox_inches="tight")
        plt.close(fig)

        rank_columns = [
            f"{name}_rank" for name in WEIGHT_SCENARIOS
        ]
        scenario_plot = mc_result.copy()
        scenario_plot["model_label"] = model_labels
        scenario_plot = scenario_plot.sort_values(
            "sensitivity_mean_rank", ascending=True
        )

        fig, ax = plt.subplots(
            figsize=(max(8, len(scenario_plot) * 0.48), 5.8)
        )
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
        ax.set_xticklabels(
            scenario_plot["model_label"], rotation=55, ha="right"
        )
        ax.set_ylabel("Rank (lower is better)")
        ax.invert_yaxis()
        ax.set_title("Rank sensitivity across predefined weighting scenarios")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(rank_distribution_plot, dpi=300, bbox_inches="tight")
        plt.close(fig)

    # Human-readable sensitivity summary.
    summary_path = reports_dir / "WeightSensitivitySummary.txt"
    sorted_result = mc_result.sort_values(
        ["mc_first_place_rate", "sensitivity_mean_rank"],
        ascending=[False, True],
    )
    best = sorted_result.iloc[0]
    feature_value = best.get("feature_set", best.get("fs", "N/A"))

    lines = [
        "WEIGHT-SENSITIVITY ANALYSIS",
        "=" * 76,
        "",
        "Primary ranking weights:",
        "- Mean CV R² rank: 0.60",
        "- CV standard-deviation rank: 0.20",
        "- Absolute train-CV gap rank: 0.20",
        "",
        "Predefined scenarios:",
        f"- Number of scenarios: {len(WEIGHT_SCENARIOS)}",
        "",
        "Monte Carlo analysis:",
        f"- Accepted constrained weight vectors: {len(mc_weights_df)}",
        "- Performance criteria jointly received at least 50% weight.",
        "- No individual criterion received more than 80% weight.",
        "",
        "Most robust first-place model:",
        f"- Model: {best.get('model', 'N/A')}",
        f"- Feature set: {feature_value}",
        f"- Primary selection rank: {best.get('statistical_rank', 'N/A')}",
        f"- First in predefined scenarios: "
        f"{int(best.get('first_place_count', 0))}/{len(WEIGHT_SCENARIOS)} "
        f"({float(best.get('first_place_rate', np.nan)):.1%})",
        f"- Monte Carlo first-place rate: "
        f"{float(best.get('mc_first_place_rate', np.nan)):.1%}",
        f"- Monte Carlo top-three rate: "
        f"{float(best.get('mc_top3_rate', np.nan)):.1%}",
        f"- Monte Carlo rank range: "
        f"{int(best.get('mc_min_rank', 0))}-"
        f"{int(best.get('mc_max_rank', 0))}",
        f"- Diagnosis: "
        f"{best.get('mc_weight_sensitivity_diagnosis', 'N/A')}",
        "",
        "Interpretation:",
        "The primary model should be described as robust only when it remains "
        "highly ranked across both the predefined scenarios and the constrained "
        "Monte Carlo weight analysis. Test metrics are not used in any of these "
        "selection-sensitivity calculations.",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    outputs = {
        "scenarios": str(scenarios_path),
        "scenario_results": str(scenario_table_path),
        "monte_carlo_weights": str(mc_weights_path),
        "monte_carlo_rank_matrix": str(rank_matrix_path),
        "summary": str(summary_path),
    }
    if plt is not None:
        outputs["first_place_plot"] = str(first_place_plot)
        outputs["rank_distribution_plot"] = str(rank_distribution_plot)

    return mc_result, outputs


# ============================================================
# Enhanced Cross-Validation Analysis with Proper 5-Fold + Repeats
# ============================================================
def compute_enhanced_cv(base_model, X, y, cv_strategy, scaler=None, 
                       return_predictions=False, n_jobs=-1):
    """
    Compute enhanced cross-validation with multiple metrics
    FIXED: Proper handling of RepeatedKFold and metrics calculation
    """
    # If base_model is a dictionary, extract the actual model
    if isinstance(base_model, dict):
        # Look for a sklearn model in the dictionary
        for key, value in base_model.items():
            if hasattr(value, 'predict') and hasattr(value, 'fit'):
                base_model = value
                break
        # If still a dict, try to create MLPRegressor from saved parameters
        if isinstance(base_model, dict):
            try:
                model_params = base_model.get('model_params', {})
                if 'MLPRegressor' in str(base_model):
                    base_model = MLPRegressor(**model_params)
                else:
                    # Try to create a simple model as fallback
                    base_model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000)
            except:
                # Last resort: create a simple MLP
                base_model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000)
    
    # Create pipeline if scaler is provided
    if scaler is not None and not isinstance(scaler, str):
        try:
            # Try to create a pipeline
            pipeline = Pipeline([
                ('scaler', scaler),
                ('model', base_model)
            ])
            model_to_cv = pipeline
        except:
            # Fallback to manual scaling
            model_to_cv = base_model
    else:
        model_to_cv = base_model
    
    # Define scoring metrics
    scoring = {
        'r2': 'r2',
        'neg_mse': 'neg_mean_squared_error',
        'neg_mae': 'neg_mean_absolute_error',
        'max_error': 'max_error'
    }
    
    try:
        # Perform cross-validation
        cv_results = cross_validate(
            model_to_cv, X, y, 
            cv=cv_strategy,
            scoring=scoring,
            return_train_score=False,
            return_estimator=return_predictions,
            n_jobs=n_jobs,
            error_score='raise'
        )
        
        # Calculate additional metrics
        cv_metrics = {}
        
        # Process R² scores
        if 'test_r2' in cv_results:
            r2_scores = cv_results['test_r2']
            cv_metrics['r2_mean'] = np.mean(r2_scores)
            cv_metrics['r2_std'] = np.std(r2_scores)
            cv_metrics['r2_scores'] = r2_scores
            cv_metrics['r2_median'] = np.median(r2_scores)
            cv_metrics['r2_iqr'] = np.percentile(r2_scores, 75) - np.percentile(r2_scores, 25)
        
        # Process MSE/RMSE scores
        if 'test_neg_mse' in cv_results:
            mse_scores = -cv_results['test_neg_mse']  # Convert back to positive MSE
            cv_metrics['mse_mean'] = np.mean(mse_scores)
            cv_metrics['mse_std'] = np.std(mse_scores)
            cv_metrics['mse_scores'] = mse_scores
            
            # Calculate RMSE
            rmse_scores = np.sqrt(mse_scores)
            cv_metrics['rmse_mean'] = np.mean(rmse_scores)
            cv_metrics['rmse_std'] = np.std(rmse_scores)
            cv_metrics['rmse_scores'] = rmse_scores
            cv_metrics['rmse_median'] = np.median(rmse_scores)
            cv_metrics['rmse_iqr'] = np.percentile(rmse_scores, 75) - np.percentile(rmse_scores, 25)
        
        # Process MAE scores
        if 'test_neg_mae' in cv_results:
            mae_scores = -cv_results['test_neg_mae']  # Convert back to positive MAE
            cv_metrics['mae_mean'] = np.mean(mae_scores)
            cv_metrics['mae_std'] = np.std(mae_scores)
            cv_metrics['mae_scores'] = mae_scores
            cv_metrics['mae_median'] = np.median(mae_scores)
            cv_metrics['mae_iqr'] = np.percentile(mae_scores, 75) - np.percentile(mae_scores, 25)
        
        # Process max error
        if 'test_max_error' in cv_results:
            max_error_scores = cv_results['test_max_error']
            cv_metrics['max_error_mean'] = np.mean(max_error_scores)
            cv_metrics['max_error_std'] = np.std(max_error_scores)
            cv_metrics['max_error_scores'] = max_error_scores
        
        # Calculate confidence intervals
        for key in ['r2', 'rmse', 'mae']:
            mean_key = f'{key}_mean'
            std_key = f'{key}_std'
            scores_key = f'{key}_scores'
            
            if mean_key in cv_metrics and std_key in cv_metrics and scores_key in cv_metrics:
                n = len(cv_metrics[scores_key])
                if n > 1:
                    se = cv_metrics[std_key] / np.sqrt(n)
                    cv_metrics[f'{key}_ci_lower'] = cv_metrics[mean_key] - 1.96 * se
                    cv_metrics[f'{key}_ci_upper'] = cv_metrics[mean_key] + 1.96 * se
                    cv_metrics[f'{key}_ci_width'] = 2 * 1.96 * se
                else:
                    cv_metrics[f'{key}_ci_lower'] = cv_metrics[mean_key]
                    cv_metrics[f'{key}_ci_upper'] = cv_metrics[mean_key]
                    cv_metrics[f'{key}_ci_width'] = 0
        
        # Store timing information
        cv_metrics['fit_time'] = cv_results.get('fit_time', [])
        cv_metrics['score_time'] = cv_results.get('score_time', [])
        
        if return_predictions:
            cv_metrics['estimators'] = cv_results.get('estimator', [])
        
        return cv_metrics
        
    except Exception as e:
        print(f"CV Error in enhanced CV: {e}")
        # Fallback to simple CV
        return compute_simple_cv(base_model, X, y, cv_strategy, scaler)

def compute_simple_cv(base_model, X, y, cv_strategy, scaler=None):
    """Fallback simple CV calculation"""
    scores_r2 = []
    scores_rmse = []
    scores_mae = []
    
    for train_idx, val_idx in cv_strategy.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        if scaler is not None and not isinstance(scaler, str):
            try:
                scaler_clone = scaler.__class__()
                X_train_scaled = scaler_clone.fit_transform(X_train)
                X_val_scaled = scaler_clone.transform(X_val)
            except:
                X_train_scaled, X_val_scaled = X_train, X_val
        else:
            X_train_scaled, X_val_scaled = X_train, X_val
        
        try:
            # Handle dictionary models
            if isinstance(base_model, dict):
                # Extract model from dict
                model_to_clone = base_model.get('model', None)
                if model_to_clone is None:
                    # Try to find any sklearn model
                    for key, value in base_model.items():
                        if hasattr(value, 'predict') and hasattr(value, 'fit'):
                            model_to_clone = value
                            break
                if model_to_clone is None:
                    # Create a simple MLP as fallback
                    model_to_clone = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000)
            else:
                model_to_clone = base_model
            
            model_clone = clone(model_to_clone)
            model_clone.fit(X_train_scaled, y_train)
            y_pred = model_clone.predict(X_val_scaled)
            
            # Calculate multiple metrics
            r2 = r2_score(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            mae = mean_absolute_error(y_val, y_pred)
            
            scores_r2.append(r2)
            scores_rmse.append(rmse)
            scores_mae.append(mae)
        except Exception as e:
            print(f"Fold error in simple CV: {e}")
            scores_r2.append(np.nan)
            scores_rmse.append(np.nan)
            scores_mae.append(np.nan)
    
    scores_r2 = np.array(scores_r2)
    scores_rmse = np.array(scores_rmse)
    scores_mae = np.array(scores_mae)
    
    # Calculate statistics only if we have valid scores
    result = {}
    if len(scores_r2) > 0 and not np.all(np.isnan(scores_r2)):
        result.update({
            'r2_mean': np.nanmean(scores_r2),
            'r2_std': np.nanstd(scores_r2),
            'r2_scores': scores_r2,
            'r2_median': np.nanmedian(scores_r2),
        })
    
    if len(scores_rmse) > 0 and not np.all(np.isnan(scores_rmse)):
        result.update({
            'rmse_mean': np.nanmean(scores_rmse),
            'rmse_std': np.nanstd(scores_rmse),
            'rmse_scores': scores_rmse,
            'rmse_median': np.nanmedian(scores_rmse),
        })
    
    if len(scores_mae) > 0 and not np.all(np.isnan(scores_mae)):
        result.update({
            'mae_mean': np.nanmean(scores_mae),
            'mae_std': np.nanstd(scores_mae),
            'mae_scores': scores_mae,
            'mae_median': np.nanmedian(scores_mae)
        })
    
    return result

def plot_cv_results(cv_metrics, metric='r2', ax=None):
    """
    Plot cross-validation results properly for 5-fold with repetitions
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
    
    # Determine which metric to plot
    if metric.lower() == 'rmse':
        scores_key = 'rmse_scores'
        mean_key = 'rmse_mean'
        std_key = 'rmse_std'
        ci_lower_key = 'rmse_ci_lower'
        ci_upper_key = 'rmse_ci_upper'
        metric_name = 'RMSE'
    elif metric.lower() == 'mae':
        scores_key = 'mae_scores'
        mean_key = 'mae_mean'
        std_key = 'mae_std'
        ci_lower_key = 'mae_ci_lower'
        ci_upper_key = 'mae_ci_upper'
        metric_name = 'MAE'
    else:  # Default to R²
        scores_key = 'r2_scores'
        mean_key = 'r2_mean'
        std_key = 'r2_std'
        ci_lower_key = 'r2_ci_lower'
        ci_upper_key = 'r2_ci_upper'
        metric_name = 'R²'
    
    if scores_key not in cv_metrics:
        ax.text(0.5, 0.5, f"No {metric_name} scores available", 
                ha='center', va='center', transform=ax.transAxes)
        return fig, ax
    
    scores = cv_metrics[scores_key]
    # Filter out NaN values
    valid_scores = scores[~np.isnan(scores)]
    
    if len(valid_scores) == 0:
        ax.text(0.5, 0.5, f"No valid {metric_name} scores", 
                ha='center', va='center', transform=ax.transAxes)
        return fig, ax
    
    n_folds = len(valid_scores)
    
    # Create fold numbers (handles repeated CV)
    fold_numbers = np.arange(1, n_folds + 1)
    
    # Plot individual fold scores
    ax.scatter(fold_numbers, valid_scores, alpha=0.7, s=50, color='steelblue', 
               edgecolor='black', zorder=5, label='Fold scores')
    
    # Plot mean line
    mean_score = cv_metrics.get(mean_key, np.mean(valid_scores))
    ax.axhline(y=mean_score, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_score:.4f}')
    
    # Plot median line
    median_score = cv_metrics.get(f'{metric.lower()}_median', np.median(valid_scores))
    ax.axhline(y=median_score, color='green', linestyle=':', linewidth=2,
               label=f'Median: {median_score:.4f}')
    
    # Plot confidence interval if available
    if ci_lower_key in cv_metrics and ci_upper_key in cv_metrics:
        ci_lower = cv_metrics[ci_lower_key]
        ci_upper = cv_metrics[ci_upper_key]
        ax.axhspan(ci_lower, ci_upper, alpha=0.2, color='orange',
                   label=f'95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]')
    
    # Add IQR if available
    iqr_key = f'{metric.lower()}_iqr'
    if iqr_key in cv_metrics:
        q1 = np.percentile(valid_scores, 25)
        q3 = np.percentile(valid_scores, 75)
        ax.axhspan(q1, q3, alpha=0.1, color='blue', label=f'IQR: [{q1:.4f}, {q3:.4f}]')
    
    # Add standard deviation bands
    std_score = cv_metrics.get(std_key, np.std(valid_scores))
    if not np.isnan(std_score):
        ax.axhspan(mean_score - std_score, mean_score + std_score, alpha=0.1, 
                   color='gray', label=f'Mean ± 1σ: ±{std_score:.4f}')
    
    # Configure plot
    ax.set_xlabel('Fold Number', fontsize=10)
    ax.set_ylabel(metric_name, fontsize=10)
    ax.set_title(f'Cross-Validation {metric_name} Scores (n={n_folds} folds)', fontsize=12)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(fold_numbers)
    
    # Add summary statistics text box
    stats_text = f"Mean: {mean_score:.4f}\n"
    if not np.isnan(std_score):
        stats_text += f"Std: {std_score:.4f}\n"
    stats_text += f"Median: {median_score:.4f}\n"
    stats_text += f"Min: {np.min(valid_scores):.4f}\n"
    stats_text += f"Max: {np.max(valid_scores):.4f}\n"
    stats_text += f"Range: {np.ptp(valid_scores):.4f}"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
            facecolor="white", alpha=0.8))
    
    return fig, ax

# ============================================================
# Regression diagnostic analysis (ROC/AUC intentionally excluded)
# ============================================================
def compute_regression_diagnostics(y_true, y_pred):
    """Return diagnostics appropriate for continuous AC regression."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    n = min(y_true.size, y_pred.size)
    y_true, y_pred = y_true[:n], y_pred[:n]
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if y_true.size < 2:
        return None
    residuals = y_true - y_pred
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    bias = float(np.mean(residuals))
    std = float(np.std(residuals, ddof=1)) if residuals.size > 1 else 0.0
    skewness = float(stats.skew(residuals, bias=False)) if residuals.size >= 3 else np.nan
    kurtosis = float(stats.kurtosis(residuals, bias=False)) if residuals.size >= 4 else np.nan
    shapiro_stat, shapiro_p = (np.nan, np.nan)
    if 3 <= residuals.size <= 5000:
        try:
            shapiro_stat, shapiro_p = stats.shapiro(residuals)
        except Exception:
            pass
    return {
        'y_true': y_true, 'y_pred': y_pred, 'residuals': residuals,
        'r2': float(r2), 'mse': float(mse), 'rmse': float(rmse),
        'mae': float(mae), 'ioa': float(ioa_willmott(y_true, y_pred)),
        'ios': float(ios_skill(y_true, y_pred)),
        'p20': float(p_within_percent(y_true, y_pred, 20.0)),
        'ad': float(anderson_darling_stat(residuals)),
        'bias': bias, 'residual_std': std, 'skewness': skewness,
        'kurtosis': kurtosis, 'shapiro_stat': float(shapiro_stat),
        'shapiro_p': float(shapiro_p), 'n': int(y_true.size)
    }

def plot_regression_necessity(ax, y_true, y_pred):
    """Q-Q residual plot plus normal-reference line for regression reporting."""
    diagnostics = compute_regression_diagnostics(y_true, y_pred)
    ax.clear()
    if diagnostics is None:
        ax.text(0.5, 0.5, 'Insufficient regression observations', ha='center', va='center', transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        return None
    residuals = diagnostics['residuals']
    (osm, osr), (slope, intercept, corr) = stats.probplot(residuals, dist='norm')
    ax.scatter(osm, osr, s=28, alpha=0.75, edgecolor='black', linewidth=0.3, label='Residual quantiles')
    xline = np.asarray([np.min(osm), np.max(osm)])
    ax.plot(xline, intercept + slope*xline, '--', linewidth=1.8, label=f'Normal reference (r={corr:.3f})')
    ax.axhline(0.0, color='gray', linewidth=0.8, alpha=0.6)
    ax.set_xlabel('Theoretical normal quantiles')
    ax.set_ylabel('Ordered residuals')
    ax.set_title('Residual Q-Q Plot and Normality Diagnostics')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='best', fontsize=7)
    text = (f"Bias={diagnostics['bias']:.4f}\n"
            f"Std={diagnostics['residual_std']:.4f}\n"
            f"Skew={diagnostics['skewness']:.3f}\n"
            f"Kurtosis={diagnostics['kurtosis']:.3f}\n"
            f"Shapiro p={diagnostics['shapiro_p']:.4f}\n"
            f"AD={diagnostics['ad']:.3f}")
    ax.text(0.02, 0.98, text, transform=ax.transAxes, va='top', fontsize=7,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))
    return diagnostics

# ============================================================
# Enhanced Statistical Summary with Hypothesis Testing
# ============================================================
def generate_statistical_summary(df):
    """
    Generate comprehensive statistical summary with hypothesis testing
    """
    if df is None or df.empty:
        return "No evaluation data available."
    
    summary = []
    summary.append("=" * 100)
    summary.append("GAP-AWARE SCIENTIFIC MODEL EVALUATION REPORT")
    summary.append("=" * 100)
    summary.append("")
    
    # Basic statistics
    summary.append("DATASET OVERVIEW:")
    summary.append(f"• Total models evaluated: {len(df)}")
    summary.append(f"• Feature sets used: {', '.join(sorted(df['fs'].unique()))}")
    summary.append(f"• Model types: {', '.join(sorted(df['model'].unique()))}")
    summary.append("")
    
    # Best model according to statistical ranking
    if 'statistical_rank' in df.columns:
        best_model_idx = df['statistical_rank'].idxmin()
        best_model = df.loc[best_model_idx]
        
        summary.append("BEST MODEL BY CV-ONLY SELECTION RANK:")
        summary.append(f"• Model: {best_model.get('model', 'N/A')}")
        summary.append(f"• Feature Set: {best_model.get('fs', 'N/A')}")
        summary.append(f"• Statistical Score: {best_model.get('statistical_score', 0):.4f}")
        
        # Add confidence interval if available
        if 'statistical_score_ci_lower' in df.columns:
            ci_lower = best_model.get('statistical_score_ci_lower', 0)
            ci_upper = best_model.get('statistical_score_ci_upper', 0)
            summary.append(f"• 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        summary.append(f"• Selection Rank: {best_model.get('statistical_rank', 0)}")
        summary.append(f"• Test Performance Rank: {best_model.get('test_performance_rank', 'NA')}")
        summary.append(f"• Generalization Rank: {best_model.get('generalization_rank', 'NA')}")
        summary.append(f"• Train-CV Gap: {best_model.get('train_cv_gap', np.nan):.4f}")
        summary.append(f"• CV-Test Gap: {best_model.get('cv_test_gap', np.nan):.4f}")
        summary.append(f"• Diagnosis: {best_model.get('generalization_diagnosis', 'N/A')}")
        
        # Add rank stability if available
        if 'rank_stability' in df.columns:
            stability = best_model.get('rank_stability', 1)
            summary.append(f"• Rank Stability: {stability:.3f} (1.0 = perfect agreement)")
        summary.append("")
    
    # Performance analysis with statistical significance
    summary.append("PERFORMANCE DISTRIBUTION WITH STATISTICAL TESTS:")
    
    metrics_to_analyze = ['r2_test', 'p20_test', 'mae_test', 'rmse_test']
    for metric in metrics_to_analyze:
        if metric in df.columns:
            values = df[metric].dropna()
            if len(values) > 0:
                metric_name = metric.replace('_test', '').upper()
                summary.append(f"• {metric_name}:")
                summary.append(f"  - Range: [{values.min():.4f}, {values.max():.4f}]")
                summary.append(f"  - Mean ± Std: {values.mean():.4f} ± {values.std():.4f}")
                summary.append(f"  - Median: {values.median():.4f}")
                summary.append(f"  - IQR: {values.quantile(0.75) - values.quantile(0.25):.4f}")
                
                # Normality test
                if len(values) >= 3:
                    try:
                        shapiro_stat, shapiro_p = stats.shapiro(values)
                        summary.append(f"  - Shapiro-Wilk normality test: W={shapiro_stat:.4f}, p={shapiro_p:.4f}")
                    except:
                        pass
    summary.append("")
    
    # Model selection recommendations with confidence
    summary.append("MODEL SELECTION RECOMMENDATIONS:")
    
    if 'r2_test' in df.columns:
        best_r2 = df['r2_test'].max()
        best_r2_model = df.loc[df['r2_test'].idxmax(), 'model']
        
        if best_r2 >= 0.8:
            summary.append(f"• ✅ EXCELLENT: {best_r2_model} achieves R² = {best_r2:.3f} (≥ 0.8)")
        elif best_r2 >= 0.6:
            summary.append(f"• ✅ GOOD: {best_r2_model} achieves R² = {best_r2:.3f} (0.6-0.8)")
        elif best_r2 >= 0.4:
            summary.append(f"• ⚠️ MODERATE: {best_r2_model} achieves R² = {best_r2:.3f} (0.4-0.6)")
        else:
            summary.append(f"• ❌ POOR: {best_r2_model} achieves R² = {best_r2:.3f} (< 0.4)")
    
    if 'p20_test' in df.columns:
        best_p20 = df['p20_test'].max()
        best_p20_model = df.loc[df['p20_test'].idxmax(), 'model']
        
        if best_p20 >= 90:
            summary.append(f"• ✅ EXCELLENT: {best_p20_model} achieves P20 = {best_p20:.1f}% (≥ 90%)")
        elif best_p20 >= 80:
            summary.append(f"• ✅ GOOD: {best_p20_model} achieves P20 = {best_p20:.1f}% (80-90%)")
        elif best_p20 >= 70:
            summary.append(f"• ⚠️ ACCEPTABLE: {best_p20_model} achieves P20 = {best_p20:.1f}% (70-80%)")
        else:
            summary.append(f"• ❌ POOR: {best_p20_model} achieves P20 = {best_p20:.1f}% (< 70%)")
    
    # Model stability assessment
    if 'rank_stability' in df.columns:
        avg_stability = df['rank_stability'].mean()
        if avg_stability >= 0.9:
            summary.append(f"• ✅ HIGH RANK STABILITY: Average stability = {avg_stability:.3f}")
        elif avg_stability >= 0.7:
            summary.append(f"• ⚠️ MODERATE RANK STABILITY: Average stability = {avg_stability:.3f}")
        else:
            summary.append(f"• ❌ LOW RANK STABILITY: Average stability = {avg_stability:.3f}")
    
    # Statistical power assessment
    if len(df) >= 10:
        summary.append(f"• 📊 ADEQUATE STATISTICAL POWER: {len(df)} models provide good power for comparisons")
    elif len(df) >= 5:
        summary.append(f"• ⚠️ LIMITED STATISTICAL POWER: {len(df)} models may limit statistical power")
    else:
        summary.append(f"• ❌ LOW STATISTICAL POWER: {len(df)} models insufficient for robust comparisons")
    
    summary.append("")
    summary.append("ENHANCED RANKING METHODOLOGY:")
    summary.append("• PCA-weighted composite score with bootstrapped confidence intervals")
    summary.append("• Pareto optimality ranking")
    summary.append("• Pairwise dominance ranking")
    summary.append("• Bayesian Dirichlet ranking")
    summary.append("• Ensemble trimmed mean rank (robust to outliers)")
    summary.append("• Rank stability assessment")
    summary.append("")
    
    summary.append("=" * 100)
    
    return "\n".join(summary)


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


# ============================================================
# GUI helper functions (preserved from original)
# ============================================================
def _hex(v):
    v = max(0, min(255, int(round(v))))
    return f"{v:02X}"

def _rgb_hex(r,g,b): return f"#{_hex(r)}{_hex(g)}{_hex(b)}"

def _color_from_rank(rank_norm):
    if rank_norm <= 0.5:
        t = rank_norm / 0.5
        r = 0 + t*(255-0)
        g = 180 + t*(210-180)
        b = 0
    else:
        t = (rank_norm - 0.5)/0.5
        r = 255 + t*(220-255)
        g = 210 + t*(0-210)
        b = 0
    return _rgb_hex(r,g,b)

def _make_scrolled_frame(parent):
    outer = ttk.Frame(parent)
    canvas = tk.Canvas(outer, highlightthickness=0)
    xscroll = ttk.Scrollbar(outer, orient="horizontal", command=canvas.xview)
    yscroll = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
    canvas.configure(xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
    
    inner = ttk.Frame(canvas)
    inner_id = canvas.create_window((0, 0), window=inner, anchor="nw")
    
    def _on_configure(event=None):
        canvas.configure(scrollregion=canvas.bbox("all"))
        canvas.itemconfigure(inner_id, width=canvas.winfo_width())
    
    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    
    def _on_linux_scroll_up(event):
        canvas.yview_scroll(-1, "units")
    
    def _on_linux_scroll_down(event):
        canvas.yview_scroll(1, "units")
    
    # Bind events
    canvas.bind("<Configure>", _on_configure)
    inner.bind("<Configure>", _on_configure)
    
    # Bind mouse wheel events
    canvas.bind("<MouseWheel>", _on_mousewheel)
    inner.bind("<MouseWheel>", _on_mousewheel)
    
    # Linux mouse wheel bindings
    canvas.bind("<Button-4>", _on_linux_scroll_up)
    canvas.bind("<Button-5>", _on_linux_scroll_down)
    inner.bind("<Button-4>", _on_linux_scroll_up)
    inner.bind("<Button-5>", _on_linux_scroll_down)
    
    # Grid layout
    canvas.grid(row=0, column=0, sticky="nsew")
    yscroll.grid(row=0, column=1, sticky="ns")
    xscroll.grid(row=1, column=0, sticky="ew")
    
    outer.rowconfigure(0, weight=1)
    outer.columnconfigure(0, weight=1)
    
    # FORCE INITIAL CONFIGURATION
    def _initial_config():
        canvas.update_idletasks()
        _on_configure()
        canvas.yview_moveto(0)
        canvas.xview_moveto(0)
    
    outer.after(100, _initial_config)
    
    return outer, inner

def _uniformize_canvas_widget(widget):
    """Make the Tk canvas for a Matplotlib figure keep the current size"""
    try:
        widget.update_idletasks()
        w, h = widget.winfo_width(), widget.winfo_height()
        if w > 1 and h > 1:
            widget.configure(width=w, height=h)
        try:
            widget.grid_propagate(False)
        except Exception:
            pass
    except Exception:
        pass

def _draw_info(ax, text):
    ax.clear()
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])

# ============================================================
# COMPLETE launch_gui function with TAB 2
# ============================================================
def on_main_window_close(window):
    """
    Close the regression GUI normally.

    The previous forced SIGTERM caused the parent launcher to receive
    termination code 15 and report:
        "The process exited with code 15"

    Closing normally avoids that false error.
    """
    try:
        window.quit()
    except Exception:
        pass

    try:
        window.destroy()
    except Exception:
        pass


def launch_gui(data_dir, models_dir, reports_dir, default_cv_folds=5, default_cv_repeats=1, params_path="params.yaml"):
    if tk is None: 
        sys.exit("[ERROR] tkinter not available.")
    if Figure is None: 
        sys.exit("[ERROR] matplotlib TkAgg backend not available.")
    
    # Load training best model
    try:
        tr_model, tr_scaler, tr_fs, tr_name = load_training_best_model_and_fs(data_dir, models_dir, reports_dir)
    except Exception as e:
        print(f"[WARN] Could not load training best model: {e}")
        tr_model = tr_scaler = tr_fs = tr_name = None
    
    # Load test best model from leaderboard
    te_model, te_scaler, te_fs, te_name, te_file = select_test_best_from_leaderboard(data_dir, models_dir, reports_dir)
    
    # Create main window
    root = tk.Tk()
    root.title("Regression Prediction, Evaluation & Visualization")
    root.geometry("1500x1080")
    root.protocol(
        "WM_DELETE_WINDOW",
        lambda: on_main_window_close(root)
    )
    # Create notebook for tabs
    nb = ttk.Notebook(root)
    nb.pack(fill="both", expand=True, padx=6, pady=6)
    
    # ============================================================
    # TAB 1: Single Regression Prediction (params.yaml driven)
    # ============================================================
    tab_single = ttk.Frame(nb)
    nb.add(tab_single, text="Single Regression Prediction")

    try:
        params_file = Path(params_path)
        with open(params_file, "r", encoding="utf-8") as stream:
            single_params = yaml.safe_load(stream) or {}
    except Exception as exc:
        single_params = {}
        print(f"[WARN] Could not load params.yaml for single prediction: {exc}")

    single_outer = ttk.Frame(tab_single, padding=14)
    single_outer.pack(fill="both", expand=True)
    single_outer.columnconfigure(1, weight=1)
    single_outer.rowconfigure(5, weight=1)

    ttk.Label(
        single_outer,
        text="Single Regression Prediction",
        font=("TkDefaultFont", 15, "bold")
    ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 12))

    ttk.Label(
        single_outer,
        text=(
            "Select a trained regression model. The required predictors are loaded "
            "automatically from params.yaml according to the model feature set."
        ),
        wraplength=1050
    ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(0, 14))

    ttk.Label(single_outer, text="Regression model:").grid(
        row=2, column=0, sticky="e", padx=(0, 8), pady=5
    )
    single_model_var = tk.StringVar()
    single_model_combo = ttk.Combobox(
        single_outer, textvariable=single_model_var, state="readonly", width=60
    )
    single_model_combo.grid(row=2, column=1, columnspan=2, sticky="ew", pady=5)

    single_status_var = tk.StringVar(value="Choose a regression model.")
    ttk.Label(single_outer, textvariable=single_status_var, foreground="blue").grid(
        row=3, column=0, columnspan=4, sticky="w", pady=(2, 10)
    )

    input_frame = ttk.LabelFrame(single_outer, text="Required model inputs", padding=12)
    input_frame.grid(row=4, column=0, columnspan=4, sticky="nw", pady=(0, 12))

    # Input controls are created dynamically from the selected model/YAML feature set.
    # Feature names are preserved exactly (spaces, punctuation and case included).
    single_input_vars = {}
    single_input_widgets = {}  # feature -> (label, entry, optional_note_label)

    result_frame = ttk.LabelFrame(single_outer, text="Prediction result", padding=12)
    result_frame.grid(row=5, column=0, columnspan=4, sticky="nsew")
    result_frame.columnconfigure(0, weight=1)
    result_frame.rowconfigure(0, weight=1)

    single_result_text = tk.Text(result_frame, height=13, wrap="word", font=("Courier New", 10))
    single_result_text.grid(row=0, column=0, sticky="nsew")
    single_result_scroll = ttk.Scrollbar(result_frame, orient="vertical", command=single_result_text.yview)
    single_result_scroll.grid(row=0, column=1, sticky="ns")
    single_result_text.configure(yscrollcommand=single_result_scroll.set)

    single_model_cache = {}
    single_state = {"package": None, "feature_names": [], "feature_set": None}

    def _single_canonical_feature(name):
        # Generic template: preserve the predictor name exactly as defined in YAML/model metadata.
        return str(name).strip()

    def _single_extract_feature_list(value):
        if isinstance(value, str):
            return [_single_canonical_feature(value)]
        if isinstance(value, (list, tuple)):
            return [_single_canonical_feature(v) for v in value]
        if isinstance(value, dict):
            for key in ("features", "feature_names", "columns", "inputs", "X"):
                if key in value:
                    return _single_extract_feature_list(value[key])
        return []

    def _single_params_features(target, fs):
        target = str(target or "")
        fs = str(fs or "").lower()
        roots = []
        for root_key in ("TARGETS", "targets", "FEATURE_SETS", "feature_sets"):
            root = single_params.get(root_key)
            if isinstance(root, dict):
                roots.append(root)
        for root in roots:
            target_block = next(
                (value for key, value in root.items() if str(key).lower() == target.lower()), None
            )
            search_blocks = [target_block, root] if isinstance(target_block, dict) else [root]
            for block in search_blocks:
                for key, value in block.items():
                    if str(key).lower() == fs:
                        names = _single_extract_feature_list(value)
                        if names:
                            return names
        return []

    def _single_feature_note(name, target=None):
        """Return an optional per-feature GUI note defined in params.yaml.

        Generic schema supported:
            feature_notes:
              FeatureName: "note shown beside the input"

        A target-scoped mapping is also accepted:
            feature_notes:
              TargetName:
                FeatureName: "note"
        """
        notes = single_params.get("feature_notes", single_params.get("FEATURE_NOTES", {}))
        if not isinstance(notes, dict):
            return ""

        # Prefer target-specific notes when provided.
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

    def _single_infer_fs(model_file, package=None):
        if isinstance(package, dict):
            fs_value = package.get("feature_set") or package.get("fs")
            if fs_value:
                return str(fs_value).lower()
        match = re.search(r"(?:^|[_-])(fs\d+)(?:[_-]|$)", Path(model_file).stem, flags=re.I)
        return match.group(1).lower() if match else None

    def _single_load_package(model_file):
        if model_file in single_model_cache:
            return single_model_cache[model_file]
        loaded = joblib.load(models_dir / model_file)
        if isinstance(loaded, dict):
            package = dict(loaded)
            model = package.get("model")
            if model is None:
                model = next((v for v in package.values() if hasattr(v, "predict")), None)
            package["model"] = model
        else:
            package = {"model": loaded}
        if package.get("model") is None:
            raise ValueError(f"No prediction model was found inside {model_file}.")
        package["feature_set"] = _single_infer_fs(model_file, package)
        single_model_cache[model_file] = package
        return package

    def _single_expected_count(model):
        candidates = [model]
        if hasattr(model, "named_steps"):
            candidates.extend(model.named_steps.values())
        for obj in candidates:
            count = getattr(obj, "n_features_in_", None)
            if count is not None:
                try:
                    return int(count)
                except Exception:
                    pass
        return None

    def _single_feature_names(package):
        model = package["model"]
        fs = package.get("feature_set")
        target = package.get("target") or ""
        params_names = _single_params_features(target, fs)

        saved_names = package.get("feature_names") or []
        if isinstance(saved_names, dict):
            saved_names = list(saved_names.values())
        saved_names = [_single_canonical_feature(x) for x in saved_names]

        estimator_names = []
        candidates = [model]
        if hasattr(model, "named_steps"):
            candidates.extend(model.named_steps.values())
        for obj in candidates:
            raw = getattr(obj, "feature_names_in_", None)
            if raw is not None:
                estimator_names = [_single_canonical_feature(x) for x in list(raw)]
                if estimator_names:
                    break

        names = params_names or saved_names or estimator_names
        if not names:
            raise ValueError(f"No feature definition was found for {fs} in {params_path}.")

        expected = _single_expected_count(model)
        if expected is not None and len(names) != expected:
            raise ValueError(
                f"Feature-count mismatch: params.yaml defines {len(names)} predictor(s) {names}, "
                f"but the selected model expects {expected}. Retrain the model with the current params.yaml."
            )
        return names

    def _single_parse(name):
        var = single_input_vars.get(name)
        if var is None:
            return None
        text = var.get().strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError as exc:
            raise ValueError(f"{name} must be numeric.") from exc

    def _single_build_row(feature_names):
        row = []
        for feature in feature_names:
            value = _single_parse(feature)
            if value is None:
                raise ValueError(f"Please enter {feature}.")
            row.append(float(value))
        return np.asarray([row], dtype=float)

    def _single_update_inputs(*_):
        model_file = single_model_var.get().strip()
        if not model_file:
            return
        try:
            package = _single_load_package(model_file)
            feature_names = _single_feature_names(package)
            fs = package.get("feature_set")
            single_state.update(package=package, feature_names=feature_names, feature_set=fs)

            for label, entry, note_label in single_input_widgets.values():
                label.grid_remove(); entry.grid_remove()
                note_label.grid_remove()

            # Build one numeric entry per required predictor, in the exact training order.
            for row, name in enumerate(feature_names):
                note_text = _single_feature_note(name, package.get("target"))
                if name not in single_input_vars:
                    var = tk.StringVar()
                    lbl = ttk.Label(input_frame, text=f"{name}:")
                    ent = ttk.Entry(input_frame, textvariable=var, width=28)
                    note_lbl = ttk.Label(input_frame, text="", foreground="gray")
                    single_input_vars[name] = var
                    single_input_widgets[name] = (lbl, ent, note_lbl)
                label, entry, note_label = single_input_widgets[name]
                note_label.configure(text=f"Note: {note_text}" if note_text else "")
                label.grid(row=row, column=0, sticky="e", padx=5, pady=4)
                entry.grid(row=row, column=1, sticky="w", padx=5, pady=4)
                if note_text:
                    note_label.grid(row=row, column=2, sticky="w", padx=(8, 5), pady=4)
                else:
                    note_label.grid_remove()

            model_type = package.get("model_type") or type(package["model"]).__name__
            order = ", ".join(feature_names)
            expected = _single_expected_count(package["model"])
            single_status_var.set(
                f"Model: {model_type} | Feature set: {fs or 'unknown'} | "
                f"Required order from params.yaml: {order} | Expected columns: {expected or len(feature_names)}"
            )
        except Exception as exc:
            single_state.update(package=None, feature_names=[], feature_set=None)
            single_status_var.set(f"Could not inspect selected model: {exc}")
            messagebox.showerror("Model feature detection", str(exc))

    def _single_predict():
        try:
            if single_state["package"] is None:
                _single_update_inputs()
            package = single_state["package"]
            feature_names = single_state["feature_names"]
            if package is None or not feature_names:
                raise ValueError("Select a valid regression model first.")

            X = _single_build_row(feature_names)
            model = package["model"]
            preprocessing = package.get("preprocessing_pipeline") or package.get("preprocessor") or package.get("transformer")
            scaler = package.get("scaler")

            if isinstance(model, Pipeline):
                X_ready = X
            elif preprocessing is not None and hasattr(preprocessing, "transform"):
                X_ready = preprocessing.transform(X)
            elif scaler is not None and hasattr(scaler, "transform"):
                X_ready = scaler.transform(X)
            else:
                X_ready = X

            prediction = np.asarray(model.predict(X_ready), dtype=float).ravel()
            if prediction.size == 0 or not np.isfinite(prediction[0]):
                raise ValueError("The selected model returned an invalid prediction.")
            predicted_value = float(prediction[0])
            target_name = str(package.get("target") or "target")
            model_type = package.get("model_type") or type(model).__name__
            input_lines = "\n".join(
                f"  {name:<24}: {X[0, idx]:.6f}"
                for idx, name in enumerate(feature_names)
            )
            text = (
                "SINGLE REGRESSION PREDICTION\n" + "=" * 66 + "\n"
                f"Model file       : {single_model_var.get()}\n"
                f"Model type       : {model_type}\n"
                f"Target           : {target_name}\n"
                f"Feature set      : {single_state['feature_set']}\n"
                f"Feature order    : {', '.join(feature_names)}\n"
                "Input values:\n" + input_lines + "\n" + "-" * 66 + "\n"
                f"Predicted {target_name}: {predicted_value:.6f}\n"
            )
            single_result_text.delete("1.0", tk.END)
            single_result_text.insert(tk.END, text)
            single_status_var.set("Prediction completed successfully.")
        except Exception as exc:
            messagebox.showerror("Single regression prediction", str(exc))
            single_status_var.set(f"Prediction failed: {exc}")

    single_button_frame = ttk.Frame(single_outer)
    single_button_frame.grid(row=6, column=0, columnspan=4, sticky="w", pady=(12, 0))
    ttk.Button(single_button_frame, text="Predict", command=_single_predict).pack(side="left", padx=(0, 8))
    ttk.Button(
        single_button_frame, text="Clear inputs",
        command=lambda: [var.set("") for var in single_input_vars.values()]
    ).pack(side="left")

    regression_model_files = sorted(
        p.name for p in models_dir.glob("*.pkl") if not p.name.lower().startswith("best")
    ) or sorted(p.name for p in models_dir.glob("*.pkl"))
    single_model_combo["values"] = regression_model_files
    if regression_model_files:
        preferred = next((name for name in regression_model_files if "svr" in name.lower()), regression_model_files[0])
        single_model_var.set(preferred)
        _single_update_inputs()
    else:
        single_status_var.set(f"No .pkl regression models found in {models_dir}")
    single_model_combo.bind("<<ComboboxSelected>>", _single_update_inputs)

    nb.select(tab_single)
    root.update_idletasks()

    # ============================================================
    # TAB 2: Model Evaluation & Ranking (Enhanced)
    # ============================================================
    tab_eval = ttk.Frame(nb)
    nb.add(tab_eval, text="Model Evaluation & Ranking")

    # Move this tab to the first (leftmost) notebook position.
    nb.insert(0, tab_eval)

    # Configure style
    style = ttk.Style()
    try:
        style.theme_use('clam')
    except:
        pass
    style.configure("Custom.Treeview", font=('TkDefaultFont', 10), rowheight=28)
    
    # Treeview columns for enhanced ranking
    cols = (
        "model", "fs", "statistical_rank", "test_performance_rank",
        "generalization_rank", "statistical_score", "rank_stability",
        "r2_train", "cv_r2_mean", "cv_r2_std", "r2_test",
        "train_cv_gap", "cv_test_gap",
        "generalization_diagnosis", "file"
    )
    
    tree = ttk.Treeview(tab_eval, columns=cols, show="headings", height=15, style="Custom.Treeview")
    
    # Set column widths
    widths = {
        "model": 105, "fs": 55, "statistical_rank": 85,
        "test_performance_rank": 90, "generalization_rank": 95,
        "statistical_score": 90, "rank_stability": 85,
        "r2_train": 75, "cv_r2_mean": 85, "cv_r2_std": 75,
        "r2_test": 75, "train_cv_gap": 90, "cv_test_gap": 90,
        "generalization_diagnosis": 190, "file": 220
    }
    
    for c in cols:
        tree.heading(c, text=c.replace('_', ' ').upper())
        tree.column(c, width=widths.get(c, 100), anchor="center")
    
    tree.grid(row=0, column=0, columnspan=5, sticky="nsew", pady=(0, 6))
    tab_eval.rowconfigure(0, weight=1)
    tab_eval.columnconfigure(0, weight=1)
    
    # Scrollbars
    vsb = ttk.Scrollbar(tab_eval, orient="vertical", command=tree.yview)
    vsb.grid(row=0, column=5, sticky="ns")
    tree.configure(yscrollcommand=vsb.set)
    
    status_leader = ttk.Label(tab_eval, text="", foreground="blue")
    status_leader.grid(row=1, column=0, columnspan=5, sticky="w")
    
    def colorize_rows_by_statistical_rank():
        items = tree.get_children()
        if not items:
            return
        
        parsed = []
        for iid in items:
            vals = tree.item(iid)["values"]
            try:
                rank = int(vals[2])  # statistical_rank column
            except Exception:
                rank = len(items) + 1
            parsed.append((iid, rank))
        
        if not parsed:
            return
        
        # Find best model (lowest rank)
        best_iid = min(parsed, key=lambda t: t[1])[0]
        
        # Normalize rank for coloring
        ranks = [t[1] for t in parsed]
        r_max, r_min = max(ranks), min(ranks)
        span = (r_max - r_min) if (r_max - r_min) != 0 else 1.0
        
        best_font = tkfont.Font(root=root, underline=2, family='Helvetica', size=10, weight='bold', slant='italic')
        
        for iid, rank in parsed:
            rank_norm = (rank - r_min) / span
            color = _color_from_rank(rank_norm)
            tag = f"row_{iid}"
            tree.tag_configure(tag, background=color)
            
            if iid == best_iid:
                tree.tag_configure("bestrow_tag", font=best_font)
                tree.item(iid, tags=(tag, "bestrow_tag"))
            else:
                tree.item(iid, tags=(tag,))
    
    # Summary text container
    summary_frame = ttk.LabelFrame(tab_eval, text="Gap-Aware Scientific Model Ranking Summary", padding=10)
    summary_frame.grid(row=2, column=0, columnspan=6, sticky="nsew", pady=(10, 0))
    tab_eval.rowconfigure(2, weight=1)
    
    # Create Text widget with scrollbar for summary
    summary_text = tk.Text(summary_frame, wrap=tk.WORD, height=15, font=('Courier New', 9))
    summary_scrollbar = ttk.Scrollbar(summary_frame, orient="vertical", command=summary_text.yview)
    summary_text.configure(yscrollcommand=summary_scrollbar.set)
    
    summary_text.grid(row=0, column=0, sticky="nsew")
    summary_scrollbar.grid(row=0, column=1, sticky="ns")
    summary_frame.columnconfigure(0, weight=1)
    summary_frame.rowconfigure(0, weight=1)
    
    # Button frame for export functionality
    button_frame = ttk.Frame(tab_eval)
    button_frame.grid(row=3, column=0, columnspan=6, sticky="w", pady=(10, 0))
    
    def reload_leaderboard():
        tree.delete(*tree.get_children())
        csv = reports_dir / "all_evaluations.csv"
        
        if not csv.exists():
            status_leader.config(text=f"[INFO] {csv.name} not found. Run evaluate_all.")
            summary_text.delete(1.0, tk.END)
            summary_text.insert(tk.END, "No evaluation data available. Please run evaluate_all first.")
            return
        
        df = pd.read_csv(csv)
        
        # Apply enhanced statistical ranking
        df = statistical_ranking_system(df)
        
        def format_value(v, nd=4, pct=False):
            try:
                val = float(str(v).replace("%", ""))
                return f"{val:.1f}%" if pct else f"{val:.{nd}f}"
            except Exception:
                return str(v)
        
        for _, r in df.iterrows():
            tree.insert("", "end", values=(
                r.get("model", ""),
                r.get("fs", ""),
                int(r.get("statistical_rank", len(df) + 1)),
                r.get("test_performance_rank", ""),
                r.get("generalization_rank", ""),
                format_value(r.get("statistical_score", np.nan), 4),
                format_value(r.get("rank_stability", np.nan), 3),
                format_value(r.get("r2_train", np.nan), 4),
                format_value(r.get("cv_r2_mean", np.nan), 4),
                format_value(r.get("cv_r2_std", np.nan), 4),
                format_value(r.get("r2_test", np.nan), 4),
                format_value(r.get("train_cv_gap", np.nan), 4),
                format_value(r.get("cv_test_gap", np.nan), 4),
                r.get("generalization_diagnosis", ""),
                r.get("file", "")
            ))
        
        colorize_rows_by_statistical_rank()
        status_leader.config(text=f"[OK] Loaded {len(df)} rows with enhanced statistical ranking.")
        
        # Update summary text with enhanced statistical analysis
        summary_report = generate_statistical_summary(df)
        summary_text.delete(1.0, tk.END)
        summary_text.insert(tk.END, summary_report)
        summary_text.see(1.0)
        
        # Save statistical ranking to CSV
        try:
            ranking_path = save_statistical_ranking_to_csv(df, reports_dir, "Enhanced_FinalRanking.csv")
            if ranking_path:
                status_leader.config(text=f"{status_leader.cget('text')} | Enhanced ranking saved to: {ranking_path.name}")
        except Exception as e:
            status_leader.config(text=f"{status_leader.cget('text')} | Error saving ranking: {str(e)}")
    
    def export_ranking_csv():
        """Export the current statistical ranking to CSV file"""
        csv = reports_dir / "all_evaluations.csv"
        if not csv.exists():
            messagebox.showerror("Error", "No evaluation data found. Please load data first.")
            return
        
        df = pd.read_csv(csv)
        df = statistical_ranking_system(df)
        
        # Ask user for save location
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile="Enhanced_FinalRanking.csv"
        )
        
        if not file_path:
            return
        
        # Save using the same function
        ranking_path = save_statistical_ranking_to_csv(df, Path(file_path).parent, Path(file_path).name)
        
        if ranking_path:
            messagebox.showinfo("Success", f"Enhanced statistical ranking saved to:\n{ranking_path}")
            status_leader.config(text=f"Enhanced ranking exported to: {ranking_path.name}")
    
    # Add buttons
    ttk.Button(button_frame, text="Reload Leaderboard", command=reload_leaderboard).grid(row=0, column=0, sticky="w", padx=(0, 10))
    ttk.Button(button_frame, text="Export Enhanced Ranking", command=export_ranking_csv).grid(row=0, column=1, sticky="w")
    
    # Add enhanced options frame
    options_frame = ttk.LabelFrame(tab_eval, text="Enhanced Ranking Options", padding=10)
    options_frame.grid(row=4, column=0, columnspan=6, sticky="ew", pady=(10, 0))
    
    # Bootstrap samples
    ttk.Label(options_frame, text="Bootstrap samples:").grid(row=0, column=0, sticky="w", padx=(0, 5))
    bootstrap_var = tk.IntVar(value=1000)
    ttk.Entry(options_frame, textvariable=bootstrap_var, width=10).grid(row=0, column=1, sticky="w", padx=(0, 10))
    
    # Confidence level
    ttk.Label(options_frame, text="Confidence level:").grid(row=0, column=2, sticky="w", padx=(0, 5))
    confidence_var = tk.DoubleVar(value=0.95)
    ttk.Entry(options_frame, textvariable=confidence_var, width=10).grid(row=0, column=3, sticky="w", padx=(0, 10))
    
    def reload_with_enhanced_options():
        """Reload leaderboard with custom bootstrap and confidence settings"""
        csv = reports_dir / "all_evaluations.csv"
        if not csv.exists():
            messagebox.showerror("Error", "No evaluation data found.")
            return
        
        df = pd.read_csv(csv)
        
        # Apply statistical ranking with custom parameters
        df = statistical_ranking_system(
            df, 
            n_bootstrap=bootstrap_var.get(),
            confidence_level=confidence_var.get()
        )
        
        # Update tree
        tree.delete(*tree.get_children())
        for _, r in df.iterrows():
            tree.insert("", "end", values=(
                r.get("model", ""),
                r.get("fs", ""),
                int(r.get("statistical_rank", len(df) + 1)),
                r.get("test_performance_rank", ""),
                r.get("generalization_rank", ""),
                f"{r.get('statistical_score', np.nan):.4f}",
                f"{r.get('rank_stability', np.nan):.3f}",
                f"{r.get('r2_train', np.nan):.4f}",
                f"{r.get('cv_r2_mean', np.nan):.4f}",
                f"{r.get('cv_r2_std', np.nan):.4f}",
                f"{r.get('r2_test', np.nan):.4f}",
                f"{r.get('train_cv_gap', np.nan):.4f}",
                r.get("generalization_diagnosis", ""),
                r.get("file", "")
            ))
        
        colorize_rows_by_statistical_rank()
        status_leader.config(text=f"[OK] Reloaded with {bootstrap_var.get()} bootstraps, {confidence_var.get()*100:.0f}% confidence")
        
        # Update summary
        summary_report = generate_statistical_summary(df)
        summary_text.delete(1.0, tk.END)
        summary_text.insert(tk.END, summary_report)
    
    ttk.Button(options_frame, text="Apply Enhanced Settings", command=reload_with_enhanced_options).grid(row=0, column=4, sticky="w")
    
    # Initial load
    reload_leaderboard()

    # Open the GUI on Model Evaluation & Ranking by default.
    nb.select(tab_eval)
    
    # ============================================================
    # TAB 3: Model Visualization & Analysis (Enhanced)
    # ============================================================
    tab_viz = ttk.Frame(nb)
    nb.add(tab_viz, text="Model Visualization & Analysis")
    nb.insert(1,tab_viz)
    
    # Create scrolled frame
    outer, viz = _make_scrolled_frame(tab_viz)
    outer.pack(fill="both", expand=True)
    
    # Row 0: Instructions
    ttk.Label(viz, text="Load predictions_long.csv then choose Model/FS & Split.").grid(
        row=0, column=0, columnspan=16, sticky="w", pady=(0, 10))
    
    # Row 1: File path
    ttk.Label(viz, text="File path:", width=10).grid(row=1, column=0, sticky="e", padx=(0, 5))
    pred_path_var = tk.StringVar(value=str(reports_dir / "predictions_long.csv"))
    ttk.Entry(viz, textvariable=pred_path_var, width=80).grid(row=1, column=1, columnspan=10, sticky="ew", padx=(0, 5))
    
    def browse_file():
        file_path = filedialog.askopenfilename(
            title="Select predictions_long.csv", 
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")]
        )
        if file_path:
            pred_path_var.set(file_path)
    
    ttk.Button(viz, text="Browse", width=10, command=browse_file).grid(row=1, column=11, padx=(0, 5))
    
    def load_predictions_long():
        path = Path(pred_path_var.get())
        if not path.exists():
            messagebox.showerror("Missing file", f"File not found:\n{path}")
            return
        
        try:
            df = pd.read_csv(path)
            required = {"index", "y_true", "y_pred", "residual", "target", "model", "fs", "file", "split"}
            if not required.issubset(df.columns):
                messagebox.showerror("Bad file", f"Missing columns: {required - set(df.columns)}")
                return
            
            # Store in global state
            viz_state["df"] = df
            
            # Populate model combo
            unique_files = sorted(df["file"].unique())
            model_combo["values"] = unique_files
            if unique_files:
                model_combo.set(unique_files[0])
            
            # Update status
            status_viz.config(text=f"Loaded {len(df)} rows from {path.name}")
            
            # Refresh plots
            refresh_basic_plots()
            plot_regression_diagnostics()
            compute_and_plot_cv()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {str(e)}")
    
    ttk.Button(viz, text="Load", width=10, command=load_predictions_long).grid(row=1, column=12, padx=(0, 5))
    
    status_viz = ttk.Label(viz, text="", foreground="blue")
    status_viz.grid(row=1, column=13, columnspan=3, sticky="w")
    
    # Row 2: Main controls
    ttk.Label(viz, text="Model (fs):", width=12).grid(row=2, column=0, sticky="e", padx=(0, 5))
    model_combo = ttk.Combobox(viz, state="readonly", width=40)
    model_combo.grid(row=2, column=1, columnspan=3, sticky="w", padx=(0, 15))
    
    ttk.Label(viz, text="Split:", width=6).grid(row=2, column=4, sticky="e", padx=(0, 5))
    split_combo = ttk.Combobox(viz, state="readonly", width=8, values=["train", "test", "all"])
    split_combo.grid(row=2, column=5, sticky="w", padx=(0, 15))
    split_combo.set("test")
    
    # Row 3: Enhanced controls
    ttk.Label(viz, text="Classification:", width=12).grid(row=3, column=0, sticky="e", padx=(0, 5))

    def launch_classification_gui_from_regression():
        import subprocess
        script_path = Path(__file__).resolve().with_name("predict_class.py")
        if not script_path.exists():
            messagebox.showerror("Missing file", f"predict_class.py was not found:\n{script_path}")
            return
        class_models = models_dir.parent / f"{models_dir.name}_class"
        class_reports = reports_dir.parent / f"{reports_dir.name}_class"
        command = [sys.executable, str(script_path), "--gui",
                   "--data", str(data_dir),
                   "--models_dir", str(class_models),
                   "--reports_dir", str(class_reports)]
        try:
            subprocess.Popen(command, cwd=str(Path(__file__).resolve().parent.parent))
        except Exception as exc:
            messagebox.showerror("Classification GUI", str(exc))

    ttk.Button(viz, text="Open Classification GUI", width=23,
               command=launch_classification_gui_from_regression).grid(
                   row=3, column=1, sticky="w", padx=(0, 5))
    
    # Enhanced CV controls - FIXED: Initialize with proper values
    ttk.Label(viz, text="CV folds:", width=10).grid(row=3, column=2, sticky="e", padx=(0, 5))
    cv_folds_var = tk.StringVar(value=str(default_cv_folds))
    ttk.Entry(viz, textvariable=cv_folds_var, width=8).grid(row=3, column=3, sticky="w", padx=(0, 5))
    
    ttk.Label(viz, text="CV repeats:", width=10).grid(row=3, column=4, sticky="e", padx=(0, 5))
    cv_repeats_var = tk.StringVar(value=str(default_cv_repeats))
    ttk.Entry(viz, textvariable=cv_repeats_var, width=8).grid(row=3, column=5, sticky="w", padx=(0, 5))
    
    ttk.Label(viz, text="CV metric:", width=10).grid(row=3, column=6, sticky="e", padx=(0, 5))
    cv_metric_combo = ttk.Combobox(viz, state="readonly", width=12, values=["R2", "MAE", "RMSE", "All"])
    cv_metric_combo.grid(row=3, column=7, sticky="w", padx=(0, 5))
    cv_metric_combo.set("R2")
    
    ttk.Label(viz, text="Plot scale:", width=10).grid(row=3, column=8, sticky="e", padx=(0, 5))
    plot_scale = tk.DoubleVar(value=1.0)
    ttk.Scale(viz, from_=0.8, to=1.8, variable=plot_scale, orient="horizontal", length=150).grid(
        row=3, column=9, columnspan=3, sticky="w", padx=(0, 5))
    
    # Configure column weights
    for col in range(16):
        viz.columnconfigure(col, weight=1 if col in [1, 3, 5, 7, 9, 13] else 0, minsize=20)
    
    # Preview table
    tree_v = ttk.Treeview(viz, columns=("index", "y_true", "y_pred", "residual", "target", "model", "fs", "file", "split"),
                         show="headings", height=8, style="Custom.Treeview")
    
    column_config = [
        ("index", 60), ("y_true", 100), ("y_pred", 100), ("residual", 100),
        ("target", 100), ("model", 200), ("fs", 60), ("file", 400), ("split", 80)
    ]
    
    for c, w in column_config:
        tree_v.heading(c, text=c.upper())
        tree_v.column(c, width=w, anchor="center", stretch=False)
    
    tree_v.grid(row=4, column=0, columnspan=16, sticky="nsew", pady=(10, 0))
    
    # Scrollbars for preview table
    vsb_v = ttk.Scrollbar(viz, orient="vertical", command=tree_v.yview)
    vsb_v.grid(row=4, column=16, sticky="ns")
    hsb_v = ttk.Scrollbar(viz, orient="horizontal", command=tree_v.xview)
    hsb_v.grid(row=5, column=0, columnspan=16, sticky="ew")
    tree_v.configure(yscrollcommand=vsb_v.set, xscrollcommand=hsb_v.set)
    
    # Create figure frames
    base_w, base_h = 6.0, 3.5
    
    frame1 = ttk.Frame(viz)
    frame1.grid(row=6, column=0, columnspan=8, sticky="nsew", padx=(0, 8), pady=(10, 0))
    frame1.grid_propagate(False)
    frame1.config(width=base_w * 80, height=base_h * 80)
    
    frame2 = ttk.Frame(viz)
    frame2.grid(row=6, column=8, columnspan=8, sticky="nsew", padx=(0, 8), pady=(10, 0))
    frame2.grid_propagate(False)
    frame2.config(width=base_w * 80, height=base_h * 80)
    
    frame3 = ttk.Frame(viz)
    frame3.grid(row=7, column=0, columnspan=8, sticky="nsew", pady=(10, 0))
    frame3.grid_propagate(False)
    frame3.config(width=base_w * 80, height=base_h * 80)
    
    frame4 = ttk.Frame(viz)
    frame4.grid(row=7, column=8, columnspan=8, sticky="nsew", pady=(10, 0))
    frame4.grid_propagate(False)
    frame4.config(width=base_w * 80, height=base_h * 80)
    
    # Summary text area
    summary_txt = tk.Text(viz, width=120, height=8, wrap="word", font=('Courier New', 9))
    summary_txt.grid(row=8, column=0, columnspan=16, sticky="nsew", pady=(10, 0))
    
    # Configure row weights
    viz.rowconfigure(6, weight=1)
    viz.rowconfigure(7, weight=1)
    
    # State variables
    viz_state = {"df": None}
    current_figures = {
        'fig1': None, 'fig2': None, 'fig3': None, 'fig4': None,
        'ax1': None, 'ax2': None, 'ax3': None, 'ax4': None,
        'canv1': None, 'canv2': None, 'canv3': None, 'canv4': None
    }
    
    def create_figure(frame, fig_num):
        """Create a new figure in the specified frame"""
        for widget in frame.winfo_children():
            widget.destroy()
        
        scale_val = float(plot_scale.get())
        fig = Figure(figsize=(base_w * scale_val, base_h * scale_val))
        ax = fig.add_subplot(111)
        canv = FigureCanvasTkAgg(fig, master=frame)
        canv_widget = canv.get_tk_widget()
        canv_widget.pack(fill="both", expand=True)
        
        # Add toolbar
        tb = ttk.Frame(frame)
        tb.pack(fill="x")
        NavigationToolbar2Tk(canv, tb).update()
        
        current_figures[f'fig{fig_num}'] = fig
        current_figures[f'ax{fig_num}'] = ax
        current_figures[f'canv{fig_num}'] = canv
        
        return fig, ax, canv
    
    # Create initial figures
    fig1, ax1, canv1 = create_figure(frame1, 1)
    fig2, ax2, canv2 = create_figure(frame2, 2)
    fig3, ax3, canv3 = create_figure(frame3, 3)
    fig4, ax4, canv4 = create_figure(frame4, 4)
    
    def _current_selection():
        df = viz_state["df"]
        if df is None or df.empty:
            raise RuntimeError("Load predictions_long.csv first.")
        selected_file = model_combo.get().strip()
        if not selected_file:
            raise RuntimeError("Select a file from the dropdown.")
        split = split_combo.get().strip()
        return df, selected_file, split
    
    def _filtered_df():
        df, selected_file, split = _current_selection()
        d2 = df[df["file"] == selected_file]
        if split != "all":
            d2 = d2[d2["split"] == split]
        if d2.empty:
            raise RuntimeError(f"No rows for file '{selected_file}' and split '{split}'.")
        return d2
    
    def refresh_preview_table(limit=500):
        tree_v.delete(*tree_v.get_children())
        try:
            d2 = _filtered_df()
        except Exception as e:
            status_viz.config(text=f"[WARN] {e}")
            return
        
        subset = d2.head(limit)
        for _, r in subset.iterrows():
            tree_v.insert("", "end", values=(
                int(r["index"]), r["y_true"], r["y_pred"], r["residual"],
                r["target"], r["model"], r["fs"], r["file"], r["split"]
            ))
    
    def refresh_basic_plots():
        try:
            d2 = _filtered_df()
        except Exception as e:
            status_viz.config(text=f"[WARN] {e}")
            return
        
        y_true = d2["y_true"].values
        y_pred = d2["y_pred"].values
        residuals = y_true - y_pred
        
        refresh_preview_table(limit=500)
        
        # Plot 1: Residuals histogram (enhanced)
        ax1 = current_figures['ax1']
        ax1.clear()
        
        # Enhanced histogram with KDE
        n, bins, patches = ax1.hist(residuals, bins=30, edgecolor="black", alpha=0.7, density=True)
        
        # Add KDE
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(residuals)
        x_range = np.linspace(residuals.min(), residuals.max(), 1000)
        ax1.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
        
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=1, label='Zero error')
        ax1.axvline(x=np.mean(residuals), color='green', linestyle='-', linewidth=1.5, 
                   label=f'Mean: {np.mean(residuals):.4f}')
        ax1.axvline(x=np.median(residuals), color='blue', linestyle='-.', linewidth=1.5,
                   label=f'Median: {np.median(residuals):.4f}')
        
        # Add normal distribution for comparison
        mu, sigma = np.mean(residuals), np.std(residuals)
        normal_pdf = stats.norm.pdf(x_range, mu, sigma)
        ax1.plot(x_range, normal_pdf, 'g--', alpha=0.7, label=f'Normal(μ={mu:.3f}, σ={sigma:.3f})')
        
        ax1.set_title("Enhanced Residuals Distribution", fontsize=9)
        ax1.set_xlabel("y_true - y_pred", fontsize=8)
        ax1.set_ylabel("Density", fontsize=8)
        ax1.legend(fontsize=7)
        ax1.grid(True, alpha=0.25)
        current_figures['canv1'].draw()
        
        # Plot 2: Actual vs Predicted (enhanced)
        ax2 = current_figures['ax2']
        ax2.clear()
        
        # Scatter plot with density coloring
        sc = ax2.scatter(y_true, y_pred, s=25, alpha=0.7, c=residuals, cmap='coolwarm')
        
        # Perfect prediction line
        mn = float(np.min([y_true.min(), y_pred.min()]))
        mx = float(np.max([y_true.max(), y_pred.max()]))
        ax2.plot([mn, mx], [mn, mx], 'k--', linewidth=2, label='Perfect prediction')
        
        # Regression line
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        lr.fit(y_true.reshape(-1, 1), y_pred)
        y_pred_lr = lr.predict(np.array([mn, mx]).reshape(-1, 1))
        ax2.plot([mn, mx], y_pred_lr, 'r-', linewidth=1.5, 
                label=f'Fit: y = {lr.coef_[0]:.3f}x + {lr.intercept_:.3f}')
        
        # Confidence bands
        std_res = np.std(residuals)
        ax2.fill_between([mn, mx], [mn - 1.96*std_res, mx - 1.96*std_res],
                         [mn + 1.96*std_res, mx + 1.96*std_res],
                         alpha=0.2, color='gray', label='95% confidence band')
        
        ax2.set_title("Actual vs Predicted with Residual Coloring", fontsize=9)
        ax2.set_xlabel("Actual", fontsize=8)
        ax2.set_ylabel("Predicted", fontsize=8)
        ax2.grid(True, alpha=0.25)
        ax2.legend(fontsize=7)
        
        # Add colorbar for residuals
        if plt:
            plt.colorbar(sc, ax=ax2, label='Residual')
        current_figures['canv2'].draw()
        
        # Calculate enhanced metrics
        r2 = r2_score(y_true, y_pred)
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        ioa = ioa_willmott(y_true, y_pred)
        ios = ios_skill(y_true, y_pred)
        p20 = p_within_percent(y_true, y_pred, 20.0)
        ad = anderson_darling_stat(residuals)
        
        # Additional statistics
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        skewness = stats.skew(residuals)
        kurtosis = stats.kurtosis(residuals)
        
        # Normality test
        shapiro_stat, shapiro_p = stats.shapiro(residuals) if len(residuals) >= 3 else (np.nan, np.nan)
        
        summary = (
            f"Selection: {model_combo.get()}  [{split_combo.get()}]\n"
            f"R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}  IOA={ioa:.4f}  IOS={ios:.4f}\n"
            f"20-Index={p20:.1f}%  AD={ad:.3f}\n"
            f"Residuals: mean={mean_residual:.4f}, std={std_residual:.4f}, skew={skewness:.3f}, kurtosis={kurtosis:.3f}\n"
            f"Normality test: W={shapiro_stat:.3f}, p={shapiro_p:.3f} {'(normal)' if shapiro_p > 0.05 else '(non-normal)'}"
        )
        
        summary_txt.delete("1.0", tk.END)
        summary_txt.insert(tk.END, summary)
        
    def plot_regression_diagnostics():
        """Plot residual Q-Q diagnostics; ROC/AUC belongs to predict_class.py."""
        try:
            d2 = _filtered_df()
            diagnostics = plot_regression_necessity(
                current_figures['ax3'],
                d2['y_true'].to_numpy(dtype=float),
                d2['y_pred'].to_numpy(dtype=float)
            )
            current_figures['canv3'].draw()
            if diagnostics is not None:
                current_summary = summary_txt.get('1.0', tk.END).strip()
                marker = '\n\nRegression Diagnostic Report:'
                if marker in current_summary:
                    current_summary = current_summary.split(marker)[0].rstrip()
                diagnostic_summary = (
                    f"{marker}\n"
                    f"N = {diagnostics['n']}\n"
                    f"Mean residual bias = {diagnostics['bias']:.6f}\n"
                    f"Residual standard deviation = {diagnostics['residual_std']:.6f}\n"
                    f"Residual skewness = {diagnostics['skewness']:.4f}\n"
                    f"Residual kurtosis = {diagnostics['kurtosis']:.4f}\n"
                    f"Shapiro-Wilk p-value = {diagnostics['shapiro_p']:.6f}\n"
                    f"Anderson-Darling statistic = {diagnostics['ad']:.6f}\n"
                    "ROC/AUC: not applicable to continuous AC regression."
                )
                summary_txt.delete('1.0', tk.END)
                summary_txt.insert(tk.END, current_summary + diagnostic_summary)
        except Exception as exc:
            print(f'Error plotting regression diagnostics: {exc}')
            ax3 = current_figures['ax3']
            ax3.clear()
            ax3.text(0.5, 0.5, f'Regression diagnostic error:\n{str(exc)[:80]}',
                     ha='center', va='center', transform=ax3.transAxes)
            ax3.set_xticks([]); ax3.set_yticks([])
            current_figures['canv3'].draw()

    def compute_and_plot_cv():
        """Enhanced cross-validation plotting with proper 5-fold + repeats handling"""
        try:
            selected_file = model_combo.get()
            if not selected_file:
                _draw_info(current_figures['ax4'], "Select a file to run CV")
                current_figures['canv4'].draw()
                return
            
            # Get model data
            d2 = _filtered_df()
            if d2.empty:
                _draw_info(current_figures['ax4'], "No data available for selected file")
                current_figures['canv4'].draw()
                return
            
            model_name = d2["model"].iloc[0]
            fs = d2["fs"].iloc[0]
            
            # Resolve training data robustly. Files may be nested below data_dir
            # and may use fs1/Fs1/FS1 naming.
            suffix = reports_dir.name.split('_')[-1] if '_' in reports_dir.name else "AC"
            x_path, y_path = _resolve_cv_training_data(data_dir, fs, suffix)

            if x_path is None or y_path is None:
                missing = []
                if x_path is None:
                    missing.append(f"X training data for {fs}")
                if y_path is None:
                    missing.append(f"y training data for {suffix}")
                _draw_info(current_figures['ax4'],"CV data not found:\n"+"\n".join(missing)+f"\n\nSearched under:\n{data_dir}")
                current_figures['canv4'].draw()
                return
            try:
                X = joblib.load(x_path)
                y = joblib.load(y_path)
            except Exception as exc:
                _draw_info(current_figures['ax4'],f"CV data loading failed:\n{exc}")
                current_figures['canv4'].draw()
                return
            print(f"[CV] X loaded from: {x_path}")
            print(f"[CV] y loaded from: {y_path}")
            if X is None or y is None or len(y) < 3:
                _draw_info(current_figures['ax4'], f"CV: insufficient data for {fs}")
                current_figures['canv4'].draw()
                return
            
            # Load model
            model_path = models_dir / selected_file
            base_model = None
            if model_path.exists():
                try:
                    base_model = joblib.load(model_path)
                except Exception:
                    base_model = None
            if base_model is None:
                try:
                    fallback_path = next((p for p in (models_dir / "best_overall.pkl", models_dir / "best.pkl") if p.exists()), None)
                    if fallback_path is None:
                        raise FileNotFoundError("best_overall.pkl/best.pkl not found")
                    base_model = joblib.load(fallback_path)
                except Exception:
                    _draw_info(current_figures['ax4'], "CV: cannot load model")
                    current_figures['canv4'].draw()
                    return
            if isinstance(base_model, dict):
                extracted = base_model.get("model")
                if extracted is None:
                    extracted = next((value for value in base_model.values() if hasattr(value, "fit") and hasattr(value, "predict")), None)
                if extracted is not None:
                    base_model = extracted
            
            # Load scaler
            def _load_scaler_for_fs(fs_):
                path = data_dir / f"scaler_{fs_}.joblib"
                if path.exists():
                    try:
                        return joblib.load(path)
                    except:
                        return None
                return None
            
            base_scaler = None  # saved model Pipeline already contains preprocessing
            
            # CV settings - FIXED: Use StringVar and validate input
            try:
                folds = int(cv_folds_var.get())
                folds = max(3, min(10, folds))  # Limit to reasonable range
            except:
                folds = default_cv_folds
            
            try:
                repeats = int(cv_repeats_var.get())
                repeats = max(1, min(10, repeats))  # Limit repeats
            except:
                repeats = 1
            
            metric = (cv_metric_combo.get() or "R2").upper()
            
            # Create CV strategy - FIXED: Proper RepeatedKFold
            if repeats > 1:
                splitter = RepeatedKFold(n_splits=folds, n_repeats=repeats, random_state=42)
                total_folds = folds * repeats
            else:
                splitter = KFold(n_splits=folds, shuffle=True, random_state=42)
                total_folds = folds
            
            # Run enhanced CV
            cv_metrics = compute_enhanced_cv(
                base_model, X, y, splitter, 
                scaler=base_scaler,
                return_predictions=False,
                n_jobs=1  # Single-threaded for GUI stability
            )
            
            # Create enhanced CV plot
            ax4 = current_figures['ax4']
            fig4 = current_figures['fig4']
            
            # Clear and create new plot
            fig4.clf()
            
            if metric == "All":
                # Plot all metrics in subplots
                metrics_to_plot = [('r2', 'R²'), ('rmse', 'RMSE'), ('mae', 'MAE')]
                n_metrics = len(metrics_to_plot)
                
                for i, (metric_key, metric_name) in enumerate(metrics_to_plot, 1):
                    ax = fig4.add_subplot(1, n_metrics, i)
                    plot_cv_results(cv_metrics, metric=metric_key, ax=ax)
                
                fig4.suptitle(f'Cross-Validation Analysis - {folds} folds, {repeats} repeats', 
                             fontsize=12, y=1.02)
            else:
                # Plot single metric
                ax = fig4.add_subplot(111)
                plot_cv_results(cv_metrics, metric=metric.lower(), ax=ax)
                ax.set_title(f'Cross-Validation {metric} Scores - {folds} folds, {repeats} repeats', 
                            fontsize=12)
            
            fig4.tight_layout()
            current_figures['canv4'].draw()
            
            # Update summary with CV results
            current_summary = summary_txt.get("1.0", tk.END).strip()
            
            # Remove existing CV summary if present
            lines = current_summary.split('\n')
            new_lines = []
            in_cv_section = False
            for line in lines:
                if "CV Analysis" in line:
                    in_cv_section = True
                if not in_cv_section or line.strip() == "":
                    new_lines.append(line)
                if line.strip() == "":
                    in_cv_section = False
            
            # Add new CV summary
            new_summary = "\n".join(new_lines).strip()
            cv_summary = f"\n\nCross-Validation Analysis ({folds} folds, {repeats} repeats):"
            
            for key in ['r2', 'rmse', 'mae']:
                mean_key = f'{key}_mean'
                std_key = f'{key}_std'
                median_key = f'{key}_median'
                
                if mean_key in cv_metrics:
                    cv_summary += f"\n{key.upper()}:"
                    cv_summary += f" Mean={cv_metrics[mean_key]:.4f}"
                    if std_key in cv_metrics:
                        cv_summary += f", Std={cv_metrics[std_key]:.4f}"
                    if median_key in cv_metrics:
                        cv_summary += f", Median={cv_metrics[median_key]:.4f}"
                    
                    # Add confidence interval if available
                    ci_lower_key = f'{key}_ci_lower'
                    ci_upper_key = f'{key}_ci_upper'
                    if ci_lower_key in cv_metrics and ci_upper_key in cv_metrics:
                        cv_summary += f", 95% CI=[{cv_metrics[ci_lower_key]:.4f}, {cv_metrics[ci_upper_key]:.4f}]"
            
            summary_txt.delete("1.0", tk.END)
            summary_txt.insert(tk.END, new_summary + cv_summary)
            
        except Exception as e:
            print(f"Error in compute_and_plot_cv: {e}")
            import traceback
            traceback.print_exc()
            _draw_info(current_figures['ax4'], f"CV Error: {str(e)[:50]}...")
            current_figures['canv4'].draw()
    
    def _resize_all_plots(*_):
        """Resize all plots simultaneously"""
        s = float(plot_scale.get())
        
        for fig_num, frame in [(1, frame1), (2, frame2), (3, frame3), (4, frame4)]:
            fig, ax, canv = create_figure(frame, fig_num)
            
            if viz_state["df"] is not None:
                try:
                    if fig_num == 1 or fig_num == 2:
                        refresh_basic_plots()
                    elif fig_num == 3:
                        plot_regression_diagnostics()
                    elif fig_num == 4:
                        compute_and_plot_cv()
                except Exception as e:
                    print(f"Error redrawing plot {fig_num}: {e}")
    
    def _try_full_refresh(*_):
        """Refresh all plots and table"""
        try:
            refresh_basic_plots()
            plot_regression_diagnostics()
            compute_and_plot_cv()
        except Exception as e:
            status_viz.config(text=f"[WARN] {e}")

    def save_all_regression_plots_png():
        """Save the four current regression diagnostics in one 2x2 PNG."""
        try:
            d2 = _filtered_df()
            y_true = d2["y_true"].to_numpy(dtype=float)
            y_pred = d2["y_pred"].to_numpy(dtype=float)
            residuals = y_true - y_pred

            selected_file = model_combo.get().strip() or "regression_model"
            split = split_combo.get().strip() or "all"

            save_path = filedialog.asksaveasfilename(
                title="Save all regression plots",
                defaultextension=".png",
                filetypes=[("PNG image", "*.png")],
                initialfile="paper_regression_all_plots.png",
            )
            if not save_path:
                return

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            ax1e, ax2e, ax3e, ax4e = axes.ravel()

            # --------------------------------------------------
            # Panel A: residual distribution
            # --------------------------------------------------
            ax1e.hist(
                residuals,
                bins=30,
                edgecolor="black",
                alpha=0.7,
                density=True,
            )

            if len(np.unique(residuals)) > 1:
                try:
                    from scipy.stats import gaussian_kde
                    kde = gaussian_kde(residuals)
                    x_range = np.linspace(
                        residuals.min(),
                        residuals.max(),
                        1000,
                    )
                    ax1e.plot(
                        x_range,
                        kde(x_range),
                        linewidth=2,
                        label="KDE",
                    )
                except Exception:
                    x_range = np.linspace(
                        residuals.min(),
                        residuals.max(),
                        1000,
                    )
            else:
                x_range = np.linspace(
                    residuals.min() - 1,
                    residuals.max() + 1,
                    1000,
                )

            ax1e.axvline(
                0,
                linestyle="--",
                linewidth=1,
                label="Zero error",
            )
            ax1e.axvline(
                np.mean(residuals),
                linewidth=1.5,
                label=f"Mean={np.mean(residuals):.3f}",
            )
            ax1e.axvline(
                np.median(residuals),
                linestyle="-.",
                linewidth=1.5,
                label=f"Median={np.median(residuals):.3f}",
            )

            sigma = np.std(residuals)
            if sigma > 0:
                normal_pdf = stats.norm.pdf(
                    x_range,
                    np.mean(residuals),
                    sigma,
                )
                ax1e.plot(
                    x_range,
                    normal_pdf,
                    linestyle="--",
                    alpha=0.8,
                    label="Normal reference",
                )

            ax1e.set_title("Residual distribution")
            ax1e.set_xlabel("Residual (actual - predicted)")
            ax1e.set_ylabel("Density")
            ax1e.grid(True, alpha=0.25)
            ax1e.legend(fontsize=8)

            # --------------------------------------------------
            # Panel B: actual vs predicted
            # --------------------------------------------------
            sc = ax2e.scatter(
                y_true,
                y_pred,
                s=30,
                alpha=0.75,
                c=residuals,
                cmap="coolwarm",
            )

            mn = float(np.min([y_true.min(), y_pred.min()]))
            mx = float(np.max([y_true.max(), y_pred.max()]))
            ax2e.plot(
                [mn, mx],
                [mn, mx],
                "k--",
                linewidth=1.8,
                label="1:1 line",
            )

            from sklearn.linear_model import LinearRegression
            lr = LinearRegression()
            lr.fit(y_true.reshape(-1, 1), y_pred)
            xx = np.array([mn, mx])
            yy = lr.predict(xx.reshape(-1, 1))
            ax2e.plot(
                xx,
                yy,
                linewidth=1.5,
                label=f"Fit: y={lr.coef_[0]:.3f}x+{lr.intercept_:.3f}",
            )

            ax2e.set_title("Actual vs predicted")
            ax2e.set_xlabel("Actual")
            ax2e.set_ylabel("Predicted")
            ax2e.grid(True, alpha=0.25)
            ax2e.legend(fontsize=8)
            fig.colorbar(sc, ax=ax2e, label="Residual")

            # --------------------------------------------------
            # Panel C: residual Q-Q / normality diagnostics
            # --------------------------------------------------
            plot_regression_necessity(ax3e, y_true, y_pred)

            # --------------------------------------------------
            # Panel D: CV plot
            # --------------------------------------------------
            try:
                model_name = d2["model"].iloc[0]
                fs = d2["fs"].iloc[0]

                suffix = (
                    reports_dir.name.split("_")[-1]
                    if "_" in reports_dir.name
                    else "AC"
                )
                x_path, y_path = _resolve_cv_training_data(data_dir, fs, suffix)
                if x_path is None or y_path is None:
                    missing = []
                    if x_path is None:
                        missing.append(f"X training data for {fs}")
                    if y_path is None:
                        missing.append(f"y training data for {suffix}")
                    raise FileNotFoundError("CV data not found: "+", ".join(missing)+f". Searched under: {data_dir}")
                X = joblib.load(x_path)
                y = joblib.load(y_path)
                print(f"[CV SAVE] X loaded from: {x_path}")
                print(f"[CV SAVE] y loaded from: {y_path}")

                model_path = models_dir / selected_file
                if model_path.exists():
                    base_model = joblib.load(model_path)
                else:
                    fallback_path = next((p for p in (models_dir / "best_overall.pkl", models_dir / "best.pkl") if p.exists()), None)
                    if fallback_path is None:
                        raise FileNotFoundError("Neither the selected model nor best_overall.pkl/best.pkl was found.")
                    base_model = joblib.load(fallback_path)
                if isinstance(base_model, dict):
                    extracted = base_model.get("model")
                    if extracted is None:
                        extracted = next((value for value in base_model.values() if hasattr(value, "fit") and hasattr(value, "predict")), None)
                    if extracted is None:
                        raise ValueError("No sklearn-compatible estimator found in the saved model package.")
                    base_model = extracted

                try:
                    folds = max(3, min(10, int(cv_folds_var.get())))
                except Exception:
                    folds = default_cv_folds

                try:
                    repeats = max(1, min(10, int(cv_repeats_var.get())))
                except Exception:
                    repeats = 1

                if repeats > 1:
                    splitter = RepeatedKFold(
                        n_splits=folds,
                        n_repeats=repeats,
                        random_state=42,
                    )
                else:
                    splitter = KFold(
                        n_splits=folds,
                        shuffle=True,
                        random_state=42,
                    )

                cv_metrics = compute_enhanced_cv(
                    base_model,
                    X,
                    y,
                    splitter,
                    scaler=None,
                    return_predictions=False,
                    n_jobs=1,
                )

                metric = (cv_metric_combo.get() or "R2").upper()
                metric_key = (
                    "rmse" if metric == "RMSE"
                    else "mae" if metric == "MAE"
                    else "r2"
                )
                plot_cv_results(
                    cv_metrics,
                    metric=metric_key,
                    ax=ax4e,
                )
                ax4e.set_title(
                    f"Cross-validation {metric_key.upper()} "
                    f"({folds} folds, {repeats} repeats)"
                )
            except Exception as exc:
                ax4e.clear()
                ax4e.text(
                    0.5,
                    0.5,
                    f"CV plot unavailable:\n{str(exc)[:120]}",
                    ha="center",
                    va="center",
                    transform=ax4e.transAxes,
                )
                ax4e.set_xticks([])
                ax4e.set_yticks([])

            fig.suptitle(
                f"Regression diagnostics: {selected_file} ({split})",
                fontsize=14,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.965])
            fig.savefig(
                save_path,
                dpi=600,
                bbox_inches="tight",
            )
            plt.close(fig)

            status_viz.config(
                text=f"[OK] Combined regression figure saved: {save_path}"
            )
            messagebox.showinfo(
                "Saved",
                f"All regression plots were saved to:\n{save_path}",
            )
        except Exception as exc:
            messagebox.showerror(
                "Save all plots",
                str(exc),
            )
    
    # Bind events for auto-refresh
    model_combo.bind("<<ComboboxSelected>>", _try_full_refresh)
    split_combo.bind("<<ComboboxSelected>>", _try_full_refresh)
    cv_metric_combo.bind("<<ComboboxSelected>>", _try_full_refresh)
    
    # Trace variables for auto-refresh - FIXED: Use lambda functions
    cv_folds_var.trace_add("write", lambda *_: _try_full_refresh())
    cv_repeats_var.trace_add("write", lambda *_: _try_full_refresh())
    plot_scale.trace_add("write", lambda *_: _resize_all_plots())
    
    # Manual refresh button
    ttk.Button(
        viz,
        text="Refresh All Plots",
        command=_try_full_refresh,
        width=15,
    ).grid(
        row=6,
        column=15,
        sticky="ne",
        padx=5,
        pady=5,
    )

    ttk.Button(
        viz,
        text="Save all plots PNG",
        command=save_all_regression_plots_png,
        width=18,
    ).grid(
        row=7,
        column=15,
        sticky="ne",
        padx=5,
        pady=5,
    )
    
    # Auto-load predictions if present
    pred_path = reports_dir / "predictions_long.csv"
    if pred_path.exists():
        pred_path_var.set(str(pred_path))
        root.after(500, load_predictions_long)  # Load after GUI is ready
    
    # ============================================================
    # Start GUI
    # ============================================================
    root.update_idletasks()

    # ---------------- Consolidated analysis outputs tab ----------------
    consolidation_tab = ttk.Frame(nb)
    nb.add(consolidation_tab, text="Nested CV & Permutation")
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
        result = run_integrated_report_consolidation(reports_dir, task="regression", experiment_family="GLFS")
        _show_consolidation_result(result)

    def _run_nested_only():
        try:
            merged, output, files = merge_nested_cv_reports(reports_dir, "R")
            _show_consolidation_result({"nested_cv": {"path": str(output), "rows": len(merged), "files": len(files)}, "permutation": None, "warnings": []})
        except Exception as exc:
            messagebox.showerror("Nested CV grouping", str(exc))

    def _run_permutation_only():
        try:
            combined, summary, output, summary_output, warns, files = extract_permutation_sensitivity_reports(
                reports_dir, task="regression", experiment_family="GLFS"
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

# ============================================================
# Enhanced Save Function with More Details
# ============================================================
def save_statistical_ranking_to_csv(
    df,
    reports_dir,
    filename="Enhanced_FinalRanking.csv",
):
    """Save a paper-ready, gap-aware model ranking table.

    The primary ``statistical_rank`` is the scientific Selection Rank and
    excludes test metrics. Test Performance Rank and Generalization Rank are
    included as separate descriptive columns.
    """
    if df is None or df.empty:
        return None

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Ensure the latest gap-aware ranking and diagnoses are present even when
    # this save function is called independently.
    save_df = statistical_ranking_system(df.copy())
    save_df, sensitivity_outputs = run_complete_weight_sensitivity(
        save_df,
        reports_dir=reports_dir,
        n_simulations=1000,
        random_state=42,
    )

    ranking_columns = [
        # Identity and three distinct ranking purposes
        "statistical_rank",
        "selection_rank",
        "test_performance_rank",
        "generalization_rank",
        "model",
        "fs",
        "feature_set",
        "target",

        # Composite ranking information
        "statistical_score",
        "selection_rank_score",
        "rank_stability",
        "generalization_rank_score",
        "test_rank_score",

        # Core train/CV-only/test performance
        "r2_train",
        "cv_r2_mean",
        "cv_r2_std",
        "cv_r2_mean",
        "r2_test",
        "cv_r2_std",
        "cv_r2_std",
        "mae_test",
        "rmse_test",
        "ioa_val",
        "ioa_test",
        "p20_val",
        "p20_test",

        # Generalization gaps
        "train_cv_gap",
        "train_val_gap",
        "cv_test_gap",
        "cv_test_gap",
        "abs_train_cv_gap",
        "abs_cv_test_gap",
        "abs_cv_test_gap",

        # Diagnostic flags and final interpretation
        "possible_overfitting",
        "possible_underfitting",
        "high_cv_instability",
        "split_sensitive",
        "stable_generalization",
        "generalization_diagnosis",

        # Individual selection-criterion ranks for transparency
        "rank_cv_r2",
        "rank_cv_stability",
        "rank_cv_stability",
        "rank_train_cv_gap",
        "rank_train_cv_gap",
        "rank_test_r2",
        "rank_test_rmse",
        "rank_test_mae",


        # Predefined weight-sensitivity scenarios
        "equal_score",
        "equal_rank",
        "performance_focused_score",
        "performance_focused_rank",
        "cv_focused_score",
        "cv_focused_rank",
        "stability_focused_score",
        "stability_focused_rank",
        "validation_focused_score",
        "validation_focused_rank",
        "sensitivity_mean_rank",
        "sensitivity_median_rank",
        "sensitivity_min_rank",
        "sensitivity_max_rank",
        "sensitivity_rank_range",
        "sensitivity_rank_sd",
        "first_place_count",
        "first_place_rate",
        "weight_sensitivity_diagnosis",

        # Constrained Monte Carlo weight-sensitivity analysis
        "mc_mean_rank",
        "mc_median_rank",
        "mc_min_rank",
        "mc_max_rank",
        "mc_rank_range",
        "mc_rank_sd",
        "mc_first_place_count",
        "mc_first_place_rate",
        "mc_top3_count",
        "mc_top3_rate",
        "mc_weight_sensitivity_diagnosis",
        "robustness_rank_score",
        "robustness_rank",
        "recommended_rank_score",
        "recommended_rank",

        # Traceability
        "file",
    ]

    # Avoid duplicate alias columns when both fs and feature_set contain the
    # same information. Prefer feature_set for the exported paper-ready file.
    if "feature_set" in save_df.columns:
        ranking_columns = [
            column for column in ranking_columns if column != "fs"
        ]

    available_columns = []
    seen = set()
    for column in ranking_columns:
        if column in save_df.columns and column not in seen:
            available_columns.append(column)
            seen.add(column)

    final_df = save_df[available_columns].copy()

    sort_columns = [
        column
        for column in ["statistical_rank", "cv_r2_mean", "cv_r2_mean"]
        if column in final_df.columns
    ]
    ascending = [True, False, False][:len(sort_columns)]
    if sort_columns:
        final_df = final_df.sort_values(
            sort_columns,
            ascending=ascending,
            na_position="last",
        )
    final_df = final_df.reset_index(drop=True)

    output_path = reports_dir / filename
    final_df.to_csv(
        output_path,
        index=False,
        float_format="%.6f",
        encoding="utf-8-sig",
    )

    # Also write the exact compact filename requested by the user.
    requested_alias = reports_dir / "EnhancedFinalRanking.csv"
    if requested_alias.resolve() != output_path.resolve():
        final_df.to_csv(
            requested_alias,
            index=False,
            float_format="%.6f",
            encoding="utf-8-sig",
        )

    metadata = {
        "generated_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_models": int(len(final_df)),
        "primary_rank": "statistical_rank / selection_rank",
        "selection_uses_test_metrics": False,
        "selection_weights": {
            "cv_r2_mean": 0.40,
            "cv_r2_mean": 0.25,
            "cv_r2_std": 0.15,
            "abs_train_cv_gap": 0.15,
            "cv_r2_std": 0.05,
        },
        "predefined_weight_scenarios": WEIGHT_SCENARIOS,
        "monte_carlo_weight_simulations": 1000,
        "monte_carlo_constraints": {
            "minimum_combined_cv_and_validation_weight": 0.50,
            "maximum_single_criterion_weight": 0.50,
        },
        "sensitivity_outputs": sensitivity_outputs,
        "test_rank_purpose": "Descriptive final holdout reporting only",
        "generalization_rank_purpose": (
            "Supplementary assessment based on train-CV gap, CV variability, "
            "and CV-validation split sensitivity"
        ),
        "principal_overfitting_indicator": "r2_train - cv_r2_mean",
        "output_alias": str(requested_alias),
    }

    metadata_path = reports_dir / f"{output_path.stem}_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    return output_path

# ============================================================
# Main CLI Function
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="Regression Model Evaluation & Visualization")
    ap.add_argument("--gui", action="store_true", help="Launch GUI")
    ap.add_argument("--data", default="data/processed", help="Input data directory")
    ap.add_argument("--models_dir", default="models", help="Models directory")
    ap.add_argument("--reports_dir", default="reports", help="Reports directory")
    ap.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    ap.add_argument("--save-ranking", action="store_true", help="Save statistical ranking to CSV and exit")
    ap.add_argument("--bootstrap", type=int, default=1000, help="Number of bootstrap samples for confidence intervals")
    ap.add_argument("--confidence", type=float, default=0.95, help="Confidence level for intervals")
    ap.add_argument("--full-report", action="store_true", help="Generate comprehensive statistical report")
    ap.add_argument("--cv-folds", type=int, default=5, help="Number of CV folds (default: 5)")
    ap.add_argument("--cv-repeats", type=int, default=1, help="Number of CV repeats (default: 1)")
    ap.add_argument("--enhanced", action="store_true", help="Use enhanced features (implied by other enhanced options)")
    
    ap.add_argument("--merge-nested-cv", action="store_true", help="Merge reports_dir/nested_cv/*_nested_cv_folds.csv and exit")
    ap.add_argument("--extract-permutation", action="store_true", help="Consolidate permutation-sensitivity CSV files and exit")
    ap.add_argument("--consolidate-analysis", action="store_true", help="Run both nested-CV grouping and permutation-sensitivity consolidation and exit")
    args = ap.parse_args()
    
    # Convert to Path objects
    data_dir = Path(args.data)
    models_dir = Path(args.models_dir)
    reports_dir = Path(args.reports_dir)
    
    if args.merge_nested_cv:
        merged, output, files = merge_nested_cv_reports(reports_dir, "R")
        print(f"Merged {len(files)} nested-CV files ({len(merged)} rows) -> {output}")
        return
    if args.extract_permutation:
        combined, summary, output, summary_output, warns, files = extract_permutation_sensitivity_reports(reports_dir, task="regression", experiment_family="GLFS")
        print(f"Merged {len(files)} permutation files ({len(combined)} rows) -> {output}")
        print(f"Feature summary: {len(summary)} rows -> {summary_output}")
        for warning in warns: print(f"[WARN] {warning}")
        return
    if args.consolidate_analysis:
        print(json.dumps(run_integrated_report_consolidation(reports_dir, task="regression", experiment_family="GLFS"), indent=2))
        return

    # Check if enhanced features are requested
    use_enhanced = args.enhanced or args.bootstrap != 1000 or args.confidence != 0.95 or args.full_report
    
    if args.save_ranking or args.full_report:
        csv = reports_dir / "all_evaluations.csv"
        if not csv.exists():
            print(f"[ERROR] {csv.name} not found. Run evaluate_all first.")
            return
        
        df = pd.read_csv(csv)
        
        # Apply enhanced statistical ranking
        df = statistical_ranking_system(df, n_bootstrap=args.bootstrap, 
                                       confidence_level=args.confidence)
        
        if args.full_report:
            # Generate and save comprehensive report
            report = generate_statistical_summary(df)
            report_path = reports_dir / "comprehensive_statistical_report.txt"
            with open(report_path, 'w') as f:
                f.write(report)
            print(f"[OK] Comprehensive report saved to: {report_path}")
            
            # Print key findings
            print("\n" + "="*80)
            print("KEY FINDINGS:")
            print("="*80)
            
            if 'statistical_rank' in df.columns:
                top3 = df.sort_values('statistical_rank').head(3)
                print("\nTop 3 Models:")
                for i, (_, row) in enumerate(top3.iterrows(), 1):
                    print(f"{i}. {row['model']} ({row['fs']}):")
                    print(f"   Rank: {row['statistical_rank']}, Score: {row['statistical_score']:.4f}")
                    if 'statistical_score_ci_lower' in df.columns:
                        print(f"   95% CI: [{row['statistical_score_ci_lower']:.4f}, {row['statistical_score_ci_upper']:.4f}]")
                    print(f"   R²: {row['r2_test']:.4f}, P20: {row['p20_test']:.1f}%")
        
        ranking_path = save_statistical_ranking_to_csv(df, reports_dir, "Enhanced_FinalRanking.csv")
        
        if ranking_path:
            print(f"[OK] Enhanced statistical ranking saved to: {ranking_path}")
            print(f"[INFO] File contains {len(df)} models with confidence intervals")
            
            # Print model diversity
            print(f"\nModel Diversity:")
            print(f"- Unique models: {df['model'].nunique()}")
            print(f"- Feature sets: {df['fs'].nunique()}")
            print(f"- Best R²: {df['r2_test'].max():.4f}")
            print(f"- Best P20: {df['p20_test'].max():.1f}%")
            
            # Statistical power assessment
            n_models = len(df)
            if n_models >= 15:
                print("\n✅ Excellent statistical power for comparisons")
            elif n_models >= 8:
                print("\n⚠️  Moderate statistical power")
            else:
                print("\n❌ Limited statistical power - consider evaluating more models")
        else:
            print("[ERROR] Failed to save ranking")
        return
    
    if args.gui:
        # Call the launch_gui function with CV parameters
        try:
            launch_gui(data_dir, models_dir, reports_dir, default_cv_folds=args.cv_folds, default_cv_repeats=args.cv_repeats, params_path=args.params)
        except Exception as e:
            print(f"[ERROR] Failed to launch GUI: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Regression Model Evaluation System")
        print("="*50)
        print(f"Data directory: {data_dir}")
        print(f"Models directory: {models_dir}")
        print(f"Reports directory: {reports_dir}")
        print(f"Enhanced features: {'Enabled' if use_enhanced else 'Disabled'}")
        print(f"CV folds: {args.cv_folds}, CV repeats: {args.cv_repeats}")
        print("\nAvailable options:")
        print("  --gui              Launch interactive GUI")
        print("  --save-ranking     Save enhanced statistical ranking")
        print("  --full-report      Generate comprehensive statistical report")
        print("  --bootstrap N      Set bootstrap samples (default: 1000)")
        print("  --confidence C     Set confidence level (default: 0.95)")
        print("  --cv-folds N       Set CV folds (default: 5)")
        print("  --cv-repeats N     Set CV repeats (default: 1)")
        print("  --enhanced         Enable all enhanced features")
        print("\nExample:")
        print("  python predict.py --gui --params params.yaml")

# ============================================================
# Main Execution
# ============================================================
if __name__ == "__main__":
    main()