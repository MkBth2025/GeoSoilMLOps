"""Multiple linear regression model definition."""
from sklearn.linear_model import LinearRegression
from .base import ModelSpec


def build_estimator(random_state: int):
    return LinearRegression()


MODEL_SPEC = ModelSpec(
    name="MLR",
    build_estimator=build_estimator,
    explanation=(
        "MLR estimates a linear relationship between the selected features and AC. "
        "It provides an interpretable baseline against nonlinear models."
    ),
)
