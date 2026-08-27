"""Support Vector Regression model definition."""
from sklearn.svm import SVR
from .base import ModelSpec


def build_estimator(random_state: int):
    """Create a fresh SVR estimator; SVR itself has no random_state parameter."""
    return SVR()


MODEL_SPEC = ModelSpec(
    name="SVR",
    build_estimator=build_estimator,
    explanation=(
        "SVR fits a regression function inside an epsilon-insensitive margin. "
        "C controls regularization, gamma controls RBF locality, and epsilon "
        "controls the width of the no-penalty tube. Scaling is usually essential."
    ),
)
