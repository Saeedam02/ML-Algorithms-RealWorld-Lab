# Decision Tree Model Card

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
- Accuracy: **0.697**
- Balanced accuracy: **0.674**
- Precision: **0.550**
- Recall: **0.602**
- F1: **0.575**
- ROC-AUC: **0.737**
- PR-AUC: **0.597**
- Fit time: **0.22s**
- Prediction time: **0.007s**

## Practical note
The benchmark contains nonlinear feature interactions and a small amount of
label noise. This prevents the task from collapsing into a trivial linearly
separable exercise.
