"""Plotting utilities.

Every function creates its own figure. No global style or hard-coded color
palette is used, so the repository stays compatible with standard Matplotlib
defaults and headless CI environments.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)


def save_actual_vs_predicted(y_true, y_pred, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.35)
    low = float(min(np.min(y_true), np.min(y_pred)))
    high = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([low, high], [low, high], linestyle="--")
    ax.set_xlabel("Actual sale price")
    ax.set_ylabel("Predicted sale price")
    ax.set_title("Linear Regression: Actual vs Predicted House Prices")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_residual_plot(y_pred, residuals, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_pred, residuals, alpha=0.35)
    ax.axhline(0, linestyle="--")
    ax.set_xlabel("Predicted sale price")
    ax.set_ylabel("Residual (actual - predicted)")
    ax.set_title("Linear Regression Residual Diagnostics")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_top_coefficients(names, coefficients, path: Path, top_n: int = 18) -> None:
    order = np.argsort(np.abs(coefficients))[-top_n:]
    selected_names = np.asarray(names)[order]
    selected_values = np.asarray(coefficients)[order]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(selected_names, selected_values)
    ax.set_xlabel("Coefficient magnitude in log-price model")
    ax.set_title("Linear Regression: Largest Absolute Coefficients")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_confusion_matrix(y_true, y_pred, model_name: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["Legitimate", "Phishing"],
        ax=ax,
        colorbar=False,
    )
    ax.set_title(f"{model_name}: Confusion Matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_roc_comparison(curves: dict[str, tuple], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, (y_true, y_score) in curves.items():
        RocCurveDisplay.from_predictions(y_true, y_score, name=name, ax=ax)
    ax.set_title("Phishing Detection: ROC Comparison")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_pr_comparison(curves: dict[str, tuple], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, (y_true, y_score) in curves.items():
        PrecisionRecallDisplay.from_predictions(y_true, y_score, name=name, ax=ax)
    ax.set_title("Phishing Detection: Precision-Recall Comparison")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_feature_bars(names, values, title: str, xlabel: str, path: Path, top_n: int = 18) -> None:
    order = np.argsort(np.abs(values))[-top_n:]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(np.asarray(names)[order], np.asarray(values)[order])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
