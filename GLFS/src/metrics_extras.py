# Full metrics_extras implementation (IOA, IOS, etc.)
from __future__ import annotations
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import (
    mean_squared_error, roc_auc_score
)
import math

EPS = 1e-12

# ---------------- Core metrics ----------------
def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))

def rmse(y_true, y_pred):
    return math.sqrt(mse(y_true, y_pred))

def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / (ss_tot + EPS))

# ---------------- Agreement metrics ----------------
def ioa_willmott(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    num = np.sum((y_pred - y_true)**2)
    denom = np.sum((np.abs(y_pred - np.mean(y_true)) + np.abs(y_true - np.mean(y_true)))**2)
    if denom == 0: return np.nan
    return 1.0 - num/denom

def ios_skill(y_true, y_pred):
    rmse_val = rmse(y_true, y_pred)
    baseline = np.full_like(y_true, np.mean(y_true))
    rmse_b = rmse(y_true, baseline)
    if rmse_b == 0: return np.nan
    return 1.0 - rmse_val/rmse_b

# ---------------- Practical accuracy ----------------
def p_within_percent(y_true, y_pred, pct=20.0):
    ok = np.abs(y_pred - y_true) <= (pct/100.0) * (np.abs(y_true) + EPS)
    return 100.0 * float(np.mean(ok))

# ---------------- Anderson-Darling statistic ----------------
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

# ---------------- Band-based discrimination & AUC ----------------
def band_labels(y_true, y_pred, low, high):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return ((y_pred >= low) & (y_pred <= high)).astype(int)

def auc_score_from_band(y_true, y_pred, low, high):
    labels = band_labels(y_true, y_pred, low, high)
    rel_err = np.abs(y_pred - y_true) / (np.abs(y_true) + EPS)
    score   = 1.0 - rel_err
    try:
        return float(roc_auc_score(labels, score))
    except Exception:
        return float("nan")

def discrimination_and_accuracy(y_true, y_pred, thresholds):
    low, high = thresholds
    lab = band_labels(y_true, y_pred, low, high)
    pred = lab
    acc = float(np.mean(lab == pred))
    pos = lab == 1
    neg = lab == 0
    tpr = float(np.mean(pred[pos] == 1)) if np.any(pos) else np.nan
    tnr = float(np.mean(pred[neg] == 0)) if np.any(neg) else np.nan
    bacc = np.nanmean([tpr, tnr])
    return acc, bacc

# ---------------- CV AUC ----------------
def cv_auc_for_regressor(X, y, model_factory, thresholds, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs = []
    for tr_idx, va_idx in kf.split(X):
        Xtr, Xva = X[tr_idx], X[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]
        try:
            model = model_factory()
            model.fit(Xtr, ytr)
            yhat = model.predict(Xva)
            aucs.append(auc_score_from_band(yva, yhat, *thresholds))
        except Exception:
            continue
    aucs = np.asarray(aucs, dtype=float)
    aucs = aucs[np.isfinite(aucs)]
    return float(np.mean(aucs)) if aucs.size else float("nan")

# ---------------- Sensitivity (Permutation Importance) ----------------
def permutation_sensitivity(model, X, y, feature_names=None, n_repeats=5):
    base_pred = model.predict(X)
    base_err = mean_squared_error(y, base_pred)
    sens = {}
    for i in range(X.shape[1]):
        losses = []
        for _ in range(n_repeats):
            Xp = X.copy()
            rng = np.random.default_rng()
            rng.shuffle(Xp[:,i])
            yp = model.predict(Xp)
            losses.append(mean_squared_error(y, yp) - base_err)
        sens[feature_names[i] if feature_names else f"f{i}"] = float(np.mean(losses))
    return sens
