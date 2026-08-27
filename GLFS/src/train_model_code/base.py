"""Shared contracts and utilities for regression model modules."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from sklearn.base import RegressorMixin


@dataclass(frozen=True)
class ModelSpec:
    """Description of one trainable regression model.

    Attributes
    ----------
    name:
        Short name used in ``params.yaml`` and output filenames.
    build_estimator:
        Callable that creates a fresh, unfitted estimator.
    explanation:
        Tutorial-oriented explanation printed or reused by documentation tools.
    optional_dependency:
        Package name required by the model, if any.
    """

    name: str
    build_estimator: Callable[[int], RegressorMixin | None]
    explanation: str
    optional_dependency: str | None = None


def normalize_hidden_layer_sizes(grid: dict[str, Any]) -> dict[str, Any]:
    """Convert YAML ANN layer lists such as ``[32, 16]`` to tuples.

    Scikit-learn expects each candidate architecture to be hashable. YAML loads
    nested sequences as lists, so they must be converted before GridSearchCV.
    """
    cleaned = dict(grid)
    if "hidden_layer_sizes" in cleaned:
        values = cleaned["hidden_layer_sizes"]
        cleaned["hidden_layer_sizes"] = [
            tuple(value) if isinstance(value, list) else value for value in values
        ]
    return cleaned
