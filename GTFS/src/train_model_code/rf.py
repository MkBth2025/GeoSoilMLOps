"""Random Forest regression model definition."""
from sklearn.ensemble import RandomForestRegressor
from .base import ModelSpec


def build_estimator(random_state: int):
    return RandomForestRegressor(random_state=random_state, n_jobs=1)


MODEL_SPEC = ModelSpec(
    name="RF",
    build_estimator=build_estimator,
    explanation=(
        "Random Forest averages many decision trees. n_estimators controls the "
        "forest size; depth and minimum-sample parameters regulate complexity."
    ),
)
