"""Ridge regression model definition."""
from sklearn.linear_model import Ridge
from .base import ModelSpec


def build_estimator(random_state: int):
    return Ridge()


MODEL_SPEC = ModelSpec(
    name="RIDGE",
    build_estimator=build_estimator,
    explanation=(
        "Ridge is linear regression with L2 regularization. Alpha controls the "
        "penalty strength and can stabilize correlated soil predictors."
    ),
)
