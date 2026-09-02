"""Compare four classic classifiers on phishing website detection."""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ..config import FIGURES_DIR, METRICS_DIR, MODEL_CARDS_DIR, RANDOM_SEED, TEST_SIZE
from ..data import load_phishing
from ..evaluation import classification_metrics, model_score
from ..models import make_classifiers
from ..plots import (
    save_confusion_matrix,
    save_feature_bars,
    save_pr_comparison,
    save_roc_comparison,
)
from ..preprocessing import build_preprocessor

DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "support_vector_machine": "Support Vector Machine",
    "k_nearest_neighbors": "K-Nearest Neighbors",
}


def run() -> dict[str, dict[str, float]]:
    """Train all four classifiers on the same split for a fair comparison."""

    df = load_phishing()
    target = "is_phishing"
    categorical = ["page_category", "tls_issuer"]
    numeric = [column for column in df.columns if column not in categorical + [target]]

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    models = make_classifiers()
    results: dict[str, dict[str, float]] = {}
    curves: dict[str, tuple] = {}

    for key, estimator in models.items():
        # Scaling is kept for every model to make the transformed feature space
        # identical across models. Trees do not need scaling, but it does not
        # change split ordering and simplifies fair pipeline comparison.
        preprocessor = build_preprocessor(numeric, categorical, scale_numeric=True)
        pipeline = Pipeline(
            [
                ("preprocess", preprocessor),
                ("model", estimator),
            ]
        )

        start = time.perf_counter()
        pipeline.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start

        start = time.perf_counter()
        predicted = pipeline.predict(X_test)
        predict_seconds = time.perf_counter() - start
        scores = model_score(pipeline, X_test)

        metrics = classification_metrics(y_test, predicted, scores)
        metrics.update(
            {
                "fit_seconds": float(fit_seconds),
                "predict_seconds": float(predict_seconds),
                "train_rows": int(len(X_train)),
                "test_rows": int(len(X_test)),
            }
        )
        results[key] = metrics
        curves[DISPLAY_NAMES[key]] = (y_test, scores)

        save_confusion_matrix(
            y_test,
            predicted,
            DISPLAY_NAMES[key],
            FIGURES_DIR / f"{key}_confusion_matrix.png",
        )

        feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
        fitted_model = pipeline.named_steps["model"]

        if key == "logistic_regression":
            save_feature_bars(
                feature_names,
                fitted_model.coef_[0],
                "Logistic Regression: Strongest Signed Coefficients",
                "Standardized log-odds coefficient",
                FIGURES_DIR / "logistic_regression_coefficients.png",
            )
        elif key == "decision_tree":
            save_feature_bars(
                feature_names,
                fitted_model.feature_importances_,
                "Decision Tree: Most Important Features",
                "Gini importance",
                FIGURES_DIR / "decision_tree_feature_importance.png",
            )

        extra = ""
        if key == "support_vector_machine":
            extra = (
                f"\n- Support vectors: **{int(np.sum(fitted_model.n_support_)):,}**"
            )
        if key == "k_nearest_neighbors":
            extra = "\n- Neighborhood size: **17** with distance weighting"

        card = f"""# {DISPLAY_NAMES[key]} Model Card

## Problem
Binary phishing-website detection from URL, domain, TLS, redirect, content,
reputation, and page-structure signals.

## Evaluation protocol
- Stratified 80/20 train/test split.
- Identical preprocessing and split across all classifiers.
- Numeric median imputation + scaling.
- Categorical most-frequent imputation + one-hot encoding.
- No test-set information is used during preprocessing.

## Test results
- Accuracy: **{metrics['accuracy']:.3f}**
- Balanced accuracy: **{metrics['balanced_accuracy']:.3f}**
- Precision: **{metrics['precision']:.3f}**
- Recall: **{metrics['recall']:.3f}**
- F1: **{metrics['f1']:.3f}**
- ROC-AUC: **{metrics['roc_auc']:.3f}**
- PR-AUC: **{metrics['pr_auc']:.3f}**
- Fit time: **{metrics['fit_seconds']:.2f}s**
- Prediction time: **{metrics['predict_seconds']:.3f}s**{extra}

## Practical note
The benchmark contains nonlinear feature interactions and a small amount of
label noise. This prevents the task from collapsing into a trivial linearly
separable exercise.
"""
        (MODEL_CARDS_DIR / f"{key}.md").write_text(card, encoding="utf-8")

    save_roc_comparison(curves, FIGURES_DIR / "classification_roc_comparison.png")
    save_pr_comparison(curves, FIGURES_DIR / "classification_pr_comparison.png")

    summary = pd.DataFrame(results).T.sort_values("f1", ascending=False)
    summary.to_csv(METRICS_DIR / "classification_summary.csv", index_label="model")
    with (METRICS_DIR / "classification_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    return results


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
