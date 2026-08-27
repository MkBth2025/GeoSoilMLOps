"""Model-specific classifier construction.

params.yaml decides which classifiers are active. This registry only provides
construction recipes and default fallback grids.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None


@dataclass(frozen=True)
class ClassifierSpec:
    name: str
    build_estimator: Callable[[int], Any]
    default_grid: dict[str, list[Any]]
    explanation: str
    optional_dependency: str | None = None


def _build_xgb(seed: int):
    if XGBClassifier is None:
        return None
    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=1,
        verbosity=0,
    )


def _specs() -> dict[str, ClassifierSpec]:
    return {
        "LOGREG": ClassifierSpec(
            "LOGREG",
            lambda seed: LogisticRegression(max_iter=3000, random_state=seed),
            {
                "penalty": ["l2"],
                "C": [0.01, 0.1, 1.0, 10.0, 100.0],
                "solver": ["lbfgs"],
                "class_weight": [None, "balanced"],
                "max_iter": [3000],
            },
            "Multinomial logistic-regression baseline.",
        ),
        "SVC": ClassifierSpec(
            "SVC",
            lambda seed: SVC(probability=True, random_state=seed),
            {
                "C": [0.1, 1.0, 10.0, 100.0],
                "kernel": ["rbf", "linear"],
                "gamma": ["scale", "auto"],
                "class_weight": [None, "balanced"],
            },
            "Support-vector classifier for nonlinear small-sample data.",
        ),
        "KNN": ClassifierSpec(
            "KNN",
            lambda seed: KNeighborsClassifier(),
            {
                "n_neighbors": [3, 5, 7, 11, 15],
                "weights": ["uniform", "distance"],
                "p": [1, 2],
            },
            "Distance-based nearest-neighbour classifier.",
        ),
        "DT": ClassifierSpec(
            "DT",
            lambda seed: DecisionTreeClassifier(random_state=seed),
            {
                "criterion": ["gini", "entropy", "log_loss"],
                "max_depth": [None, 3, 5, 8, 12],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "class_weight": [None, "balanced"],
            },
            "Interpretable nonlinear decision-tree classifier.",
        ),
        "RF": ClassifierSpec(
            "RF",
            lambda seed: RandomForestClassifier(random_state=seed, n_jobs=1),
            {
                "n_estimators": [100, 300, 500],
                "max_depth": [None, 5, 10, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "class_weight": [None, "balanced"],
            },
            "Random-forest classifier.",
        ),
        "ET": ClassifierSpec(
            "ET",
            lambda seed: ExtraTreesClassifier(random_state=seed, n_jobs=1),
            {
                "n_estimators": [100, 300, 500],
                "max_depth": [None, 5, 10, 20],
                "min_samples_leaf": [1, 2, 4],
                "class_weight": [None, "balanced"],
            },
            "Extra-Trees classifier for nonlinear tabular data.",
        ),
        "GBC": ClassifierSpec(
            "GBC",
            lambda seed: GradientBoostingClassifier(random_state=seed),
            {
                "n_estimators": [100, 200],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [2, 3],
            },
            "Gradient-boosting classifier.",
        ),
        "XGB": ClassifierSpec(
            "XGB",
            _build_xgb,
            {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 6, 9],
                "learning_rate": [0.01, 0.1, 0.3],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
            },
            "XGBoost multiclass classifier.",
            optional_dependency="xgboost",
        ),
        "ANN": ClassifierSpec(
            "ANN",
            lambda seed: MLPClassifier(
                max_iter=3000,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=seed,
            ),
            {
                "hidden_layer_sizes": [(16,), (32,), (32, 16), (64, 32)],
                "activation": ["relu", "tanh"],
                "solver": ["adam"],
                "alpha": [0.0001, 0.001, 0.01],
                "learning_rate_init": [0.0005, 0.001, 0.005],
                "batch_size": [8, 16, 32],
                "early_stopping": [True],
                "max_iter": [3000],
            },
            "Feed-forward neural-network classifier.",
        ),
    }


def available_classifiers() -> list[str]:
    return list(_specs())


def validate_requested_classifiers(names: list[str]) -> list[str]:
    requested = [str(name).strip().upper() for name in names]
    specs = _specs()
    unknown = [name for name in requested if name not in specs]
    if unknown:
        raise KeyError(
            "Unknown classifier name(s) in params.yaml: "
            f"{', '.join(unknown)}. Supported recipes: {', '.join(specs)}"
        )
    return requested


def load_classifier_spec(name: str) -> ClassifierSpec:
    key = str(name).strip().upper()
    specs = _specs()
    if key not in specs:
        raise KeyError(
            f"Unknown classifier '{name}'. Available: {', '.join(specs)}"
        )
    spec = specs[key]
    if key == "XGB" and XGBClassifier is None:
        raise ImportError("XGB classifier requires the optional 'xgboost' package.")
    return spec
