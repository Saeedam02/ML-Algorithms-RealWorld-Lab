"""End-to-end Linear Regression experiment on the housing benchmark."""

from __future__ import annotations

import json

from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ..config import FIGURES_DIR, METRICS_DIR, MODEL_CARDS_DIR, RANDOM_SEED, TEST_SIZE
from ..data import load_housing
from ..evaluation import regression_metrics
from ..models import make_linear_regression
from ..plots import save_actual_vs_predicted, save_residual_plot, save_top_coefficients
from ..preprocessing import build_preprocessor


def run() -> dict[str, float]:
    """Train, evaluate, and document the Linear Regression model."""

    df = load_housing()
    target = "sale_price"
    categorical = ["neighborhood", "home_type", "condition", "heating_type"]
    numeric = [column for column in df.columns if column not in categorical + [target]]

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
    )

    preprocessor = build_preprocessor(numeric, categorical, scale_numeric=True)
    model = make_linear_regression()

    pipeline = Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    metrics = regression_metrics(y_test, predictions)
    metrics.update(
        {
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "raw_features": int(X.shape[1]),
        }
    )

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with (METRICS_DIR / "linear_regression.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    save_actual_vs_predicted(
        y_test.to_numpy(),
        predictions,
        FIGURES_DIR / "linear_regression_actual_vs_predicted.png",
    )
    save_residual_plot(
        predictions,
        y_test.to_numpy() - predictions,
        FIGURES_DIR / "linear_regression_residuals.png",
    )

    # Extract coefficients from the fitted transformed-target regressor.
    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    fitted_ttr: TransformedTargetRegressor = pipeline.named_steps["model"]
    coefficients = fitted_ttr.regressor_.coef_
    save_top_coefficients(
        feature_names,
        coefficients,
        FIGURES_DIR / "linear_regression_coefficients.png",
    )

    card = f"""# Linear Regression Model Card

## Problem
Predict residential sale price from mixed property, neighborhood, mobility,
energy, and macroeconomic variables.

## Dataset
- Rows: {len(df):,}
- Raw predictors: {X.shape[1]}
- Missing values: yes
- Numeric + categorical variables: yes
- Rare price outliers: yes

## Pipeline
1. Median imputation for numeric values.
2. Standardization of numeric variables.
3. Most-frequent imputation + one-hot encoding for categorical variables.
4. `log1p` transform of sale price.
5. Ordinary least-squares `LinearRegression`.
6. Inverse transform to dollars for evaluation.

## Test results
- R²: **{metrics['r2']:.3f}**
- MAE: **${metrics['mae']:,.0f}**
- RMSE: **${metrics['rmse']:,.0f}**
- MAPE: **{metrics['mape_pct']:.2f}%**

## Interpretation
Linear regression is highly interpretable and establishes a strong baseline.
Because the benchmark contains interaction effects and heavy-tailed outliers,
residual structure remains; that is intentional and demonstrates where linear
assumptions break down.
"""
    (MODEL_CARDS_DIR / "linear_regression.md").write_text(card, encoding="utf-8")

    return metrics


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
