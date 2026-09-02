"""Model factory for the five algorithms demonstrated in this repository."""

from __future__ import annotations

from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def make_linear_regression():
    """Linear least-squares regressor with a log-transformed target.

    The estimator is still ordinary LinearRegression; log1p/expm1 simply make
    the heavy-tailed house-price target easier to model and evaluate.
    """

    import numpy as np

    return TransformedTargetRegressor(
        regressor=LinearRegression(),
        func=np.log1p,
        inverse_func=np.expm1,
    )


def make_classifiers() -> dict[str, object]:
    """Return the four classification algorithms with sensible fixed settings.

    Hyperparameters are intentionally explicit rather than hidden behind a large
    search. The project focuses on algorithm behavior and reproducibility.
    """

    return {
        "logistic_regression": LogisticRegression(
            max_iter=2_000,
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=9,
            min_samples_leaf=18,
            class_weight="balanced",
            random_state=42,
        ),
        "support_vector_machine": SVC(
            C=2.0,
            kernel="rbf",
            gamma="scale",
            class_weight="balanced",
            cache_size=1_000,
        ),
        "k_nearest_neighbors": KNeighborsClassifier(
            n_neighbors=17,
            weights="distance",
            metric="minkowski",
            p=2,
            n_jobs=-1,
        ),
    }
