"""K-nearest-neighbors regression model definition."""
from sklearn.neighbors import KNeighborsRegressor
from .base import ModelSpec


def build_estimator(random_state: int):
    return KNeighborsRegressor()


MODEL_SPEC = ModelSpec(
    name="KNN",
    build_estimator=build_estimator,
    explanation=(
        "KNN predicts from nearby training samples. n_neighbors controls local "
        "smoothness, weights controls neighbor influence, and p selects the distance metric."
    ),
)
