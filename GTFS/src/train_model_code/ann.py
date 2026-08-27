"""Artificial neural network regression model definition."""
from sklearn.neural_network import MLPRegressor
from .base import ModelSpec


def build_estimator(random_state: int):
    return MLPRegressor(
        random_state=random_state,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.10,
        n_iter_no_change=20,
        tol=1e-4,
    )


MODEL_SPEC = ModelSpec(
    name="ANN",
    build_estimator=build_estimator,
    explanation=(
        "The ANN uses fully connected hidden layers. hidden_layer_sizes defines "
        "network architecture; alpha adds L2 regularization; learning_rate_init "
        "sets the optimizer step size. Early stopping limits overfitting."
    ),
)
