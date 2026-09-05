# K-Nearest Neighbors Model Card

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
- Accuracy: **0.742**
- Balanced accuracy: **0.646**
- Precision: **0.770**
- Recall: **0.344**
- F1: **0.476**
- ROC-AUC: **0.781**
- PR-AUC: **0.673**
- Fit time: **0.05s**
- Prediction time: **0.085s**
- Neighborhood size: **17** with distance weighting

## Practical note
The benchmark contains nonlinear feature interactions and a small amount of
label noise. This prevents the task from collapsing into a trivial linearly
separable exercise.
