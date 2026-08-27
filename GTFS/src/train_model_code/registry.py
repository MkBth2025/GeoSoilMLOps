from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
except ImportError:
    GaussianProcessRegressor = None
    ConstantKernel = RBF = WhiteKernel = None


@dataclass(frozen=True)
class ModelSpec:
    name: str
    builder: Callable[[int], Any]
    explanation: str
    optional_dependency: str | None = None

    def build_estimator(self, random_state: int):
        return self.builder(random_state)


def _build_xgb(seed: int):
    if XGBRegressor is None:
        return None
    return XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        random_state=seed,
        n_jobs=1,
        verbosity=0,
    )


def _build_gpr(seed: int):
    if GaussianProcessRegressor is None:
        return None
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))
        + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-8, 1e1))
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        random_state=seed,
        n_restarts_optimizer=1,
    )


_MODEL_SPECS: dict[str, ModelSpec] = {
    "MLR": ModelSpec(
        "MLR",
        lambda seed: LinearRegression(),
        "Ordinary multiple linear regression baseline.",
    ),
    "RIDGE": ModelSpec(
        "RIDGE",
        lambda seed: Ridge(),
        "L2-regularized linear regression.",
    ),
    "LASSO": ModelSpec(
        "LASSO",
        lambda seed: Lasso(random_state=seed, max_iter=10000),
        "L1-regularized sparse linear regression.",
    ),
    "EN": ModelSpec(
        "EN",
        lambda seed: ElasticNet(random_state=seed, max_iter=10000),
        "Elastic Net regression combining L1 and L2 regularization.",
    ),
    "SVR": ModelSpec(
        "SVR",
        lambda seed: SVR(),
        "Support Vector Regression for nonlinear small-sample tabular data.",
    ),
    "KNN": ModelSpec(
        "KNN",
        lambda seed: KNeighborsRegressor(),
        "Instance-based nonlinear regression.",
    ),
    "DT": ModelSpec(
        "DT",
        lambda seed: DecisionTreeRegressor(random_state=seed),
        "Interpretable nonlinear decision-tree baseline.",
    ),
    "RF": ModelSpec(
        "RF",
        lambda seed: RandomForestRegressor(
            random_state=seed,
            n_jobs=1,
        ),
        "Bagged decision-tree ensemble.",
    ),
    "ET": ModelSpec(
        "ET",
        lambda seed: ExtraTreesRegressor(
            random_state=seed,
            n_jobs=1,
        ),
        "Highly randomized tree ensemble suitable for nonlinear tabular data.",
    ),
    "GBR": ModelSpec(
        "GBR",
        lambda seed: GradientBoostingRegressor(random_state=seed),
        "Sequential boosting model effective on small tabular datasets.",
    ),
    "XGB": ModelSpec(
        "XGB",
        _build_xgb,
        "Extreme Gradient Boosting regressor.",
        optional_dependency="xgboost",
    ),
    "ANN": ModelSpec(
        "ANN",
        lambda seed: MLPRegressor(
            random_state=seed,
            max_iter=3000,
            early_stopping=True,
        ),
        "Feed-forward artificial neural network regressor.",
    ),
    "GPR": ModelSpec(
        "GPR",
        _build_gpr,
        "Gaussian Process Regression.",
    ),
    "KRR": ModelSpec(
        "KRR",
        lambda seed: KernelRidge(),
        "Kernel Ridge Regression.",
    ),
}


def available_models() -> list[str]:
    return sorted(_MODEL_SPECS)


def load_model_spec(name: str) -> ModelSpec:
    key = str(name).strip().upper()
    if key not in _MODEL_SPECS:
        supported = ", ".join(available_models())
        raise KeyError(f"Unknown model '{key}'. Supported models: {supported}")
    return _MODEL_SPECS[key]


def validate_requested_models(model_names: list[str]) -> list[str]:
    """Validate models enabled in params.yaml.

    ``params.yaml`` determines the active model set. The registry is only a
    catalog of construction recipes for model identifiers that the code knows
    how to build.
    """
    requested = [str(name).strip().upper() for name in model_names]
    unknown = [name for name in requested if name not in _MODEL_SPECS]
    if unknown:
        supported = ", ".join(available_models())
        raise KeyError(
            "Unknown model name(s) in params.yaml: "
            f"{', '.join(unknown)}. Supported registry recipes: {supported}"
        )
    return requested


def load_requested_model_specs(model_names: list[str]) -> dict[str, ModelSpec]:
    """Return only model specifications explicitly enabled in params.yaml."""
    requested = validate_requested_models(model_names)
    return {name: _MODEL_SPECS[name] for name in requested}
