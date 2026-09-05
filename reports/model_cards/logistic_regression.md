# Logistic Regression Model Card

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
- Accuracy: **0.781**
- Balanced accuracy: **0.775**
- Precision: **0.653**
- Recall: **0.757**
- F1: **0.701**
- ROC-AUC: **0.860**
- PR-AUC: **0.792**
- Fit time: **0.24s**
- Prediction time: **0.008s**

## Practical note
The benchmark contains nonlinear feature interactions and a small amount of
label noise. This prevents the task from collapsing into a trivial linearly
separable exercise.
