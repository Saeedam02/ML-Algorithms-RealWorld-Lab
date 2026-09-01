# ML Interview Cheat Sheet

| Algorithm | Core idea | Scale features? | Handles nonlinearities? | Main risk |
|---|---|---:|---:|---|
| Linear Regression | Least-squares continuous prediction | Usually helpful | No | misspecification / collinearity |
| Logistic Regression | Linear log-odds classifier | Yes | Not by itself | linear boundary |
| Decision Tree | Recursive threshold splits | No | Yes | overfitting |
| SVM (RBF) | Maximum-margin kernel classifier | Yes | Yes | training cost / tuning |
| KNN | Local neighbor voting | Essential | Yes | prediction cost / curse of dimensionality |

## Bias/variance intuition

- Linear and Logistic Regression: relatively high bias, lower variance.
- Deep Decision Tree: low bias, potentially very high variance.
- RBF SVM: flexible; C and gamma control complexity.
- KNN: small K lowers bias and raises variance; large K does the opposite.

## Metrics

For imbalanced security tasks, never rely on accuracy alone.

- Precision: of predicted phishing pages, how many were actually phishing?
- Recall: of all phishing pages, how many did we catch?
- F1: harmonic mean of precision and recall.
- ROC-AUC: ranking quality across thresholds.
- PR-AUC: especially useful when the positive class is relatively rare.
