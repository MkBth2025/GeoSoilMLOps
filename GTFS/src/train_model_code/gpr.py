"""Gaussian Process regression model definition."""
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from .base import ModelSpec


def build_estimator(random_state: int):
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
    return GaussianProcessRegressor(
        kernel=kernel,
        random_state=random_state,
        normalize_y=True,
        n_restarts_optimizer=2,
    )


MODEL_SPEC = ModelSpec(
    name="GPR",
    build_estimator=build_estimator,
    explanation=(
        "GPR models a distribution over regression functions. The kernel encodes "
        "smoothness assumptions, while alpha represents observation-noise regularization."
    ),
)
