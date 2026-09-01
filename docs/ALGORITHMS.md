# Algorithm Notes

This document focuses on the five algorithms implemented in the repository.

## 1. Linear Regression

Linear regression models a continuous target as a weighted sum of features:

$$
\hat{y} = \beta_0 + \beta_1x_1 + \cdots + \beta_px_p
$$

Ordinary least squares chooses the coefficients that minimize:

$$
\mathrm{SSE} = \sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

**Strengths:** fast, interpretable, strong baseline, coefficients are useful for
reasoning about direction and magnitude.

**Weaknesses:** linear functional form, sensitivity to collinearity/outliers,
interactions must be represented explicitly.

**Repository problem:** predicting house sale prices.

---

## 2. Logistic Regression

Logistic regression models the log-odds of a binary event:

$$
P(y=1\mid x)=\sigma(z)=\frac{1}{1+e^{-z}}
$$

with

$$
z=\beta_0+\beta^\top x
$$

It is trained by minimizing log loss rather than squared error.

**Strengths:** probabilistic, interpretable, fast, excellent baseline.

**Weaknesses:** linear decision boundary in transformed feature space unless
interactions/nonlinear features are engineered.

**Repository problem:** phishing detection.

---

## 3. Decision Tree

A decision tree recursively partitions feature space. For classification,
candidate splits are commonly evaluated by impurity reduction. Gini impurity is:

$$
G = 1-\sum_{k=1}^{K}p_k^2
$$

**Strengths:** nonlinear interactions, minimal feature assumptions, transparent
if shallow.

**Weaknesses:** unstable and prone to overfitting without depth/min-leaf
regularization.

**Repository problem:** phishing detection.

---

## 4. Support Vector Machine

A linear SVM seeks a separating hyperplane with the largest margin. For
nonlinear boundaries, a kernel replaces explicit dot products.

This repository uses the RBF kernel:

$$
K(x_i,x_j)=\exp(-\gamma\|x_i-x_j\|^2)
$$

**Strengths:** powerful nonlinear decision boundaries, robust high-dimensional
classification.

**Weaknesses:** training cost can become high on very large datasets; feature
scaling is important; direct interpretation is difficult.

**Repository problem:** phishing detection.

---

## 5. K-Nearest Neighbors

KNN is a lazy learner: it stores training samples and predicts using nearby
examples. For classification:

$$
\hat{y}(x)=\mathrm{mode}\{y_i:x_i\in N_k(x)\}
$$

This repository uses distance-weighted voting so closer neighbors contribute
more strongly.

**Strengths:** intuitive, nonlinear, almost no training cost.

**Weaknesses:** prediction cost grows with training data, sensitive to scaling,
and suffers in very high-dimensional spaces.

**Repository problem:** phishing detection.

---

## Why compare four classifiers on the same data?

Using exactly the same split and preprocessing exposes different inductive
biases:

- Logistic Regression tests how far a linear boundary can go.
- Decision Tree captures threshold interactions.
- SVM captures smooth nonlinear boundaries.
- KNN relies on local similarity rather than an explicit global model.

That makes the classification half of the repository a real algorithm
comparison rather than five unrelated toy notebooks.
