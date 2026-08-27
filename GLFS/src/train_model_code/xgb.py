"""XGBoost regression model definition (optional dependency)."""
from .base import ModelSpec


def build_estimator(random_state: int):
    try:
        from xgboost import XGBRegressor
    except ImportError:
        return None
    return XGBRegressor(random_state=random_state, n_jobs=1, objective="reg:squarederror")


MODEL_SPEC = ModelSpec(
    name="XGB",
    build_estimator=build_estimator,
    optional_dependency="xgboost",
    explanation=(
        "XGBoost builds trees sequentially to correct prior residuals. Learning "
        "rate, depth, sampling, and regularization jointly control accuracy and overfitting."
    ),
)
