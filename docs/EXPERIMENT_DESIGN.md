# Experiment Design

## Leakage prevention

Every transformation is placed inside a scikit-learn `Pipeline`:

1. Split raw rows.
2. Fit imputers only on training data.
3. Fit scalers only on training data.
4. Fit category encoders only on training data.
5. Fit model.
6. Evaluate once on the held-out test set.

## Regression protocol

- 80/20 random split.
- Log-transform target during training.
- Report R², MAE, RMSE and MAPE on original dollar scale.
- Inspect actual-vs-predicted, residuals and coefficients.

## Classification protocol

- 80/20 stratified split.
- Same rows for every classifier.
- Same mixed-type preprocessing.
- Report accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, PR-AUC,
  fit time and prediction time.
- Inspect confusion matrices, ROC curve, precision-recall curve and available
  feature-importance diagnostics.

## Why no giant hyperparameter search?

The repository teaches algorithms, not leaderboard chasing. Each model has
reasonable, explicit settings. This makes comparisons reproducible and keeps the
code readable. A future `tuning/` extension can add cross-validated searches
without changing the baseline results.
