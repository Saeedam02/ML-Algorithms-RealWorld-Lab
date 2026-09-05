# Linear Regression Model Card

## Problem
Predict residential sale price from mixed property, neighborhood, mobility,
energy, and macroeconomic variables.

## Dataset
- Rows: 12,000
- Raw predictors: 19
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
- R²: **0.799**
- MAE: **$57,902**
- RMSE: **$83,701**
- MAPE: **7.98%**

## Interpretation
Linear regression is highly interpretable and establishes a strong baseline.
Because the benchmark contains interaction effects and heavy-tailed outliers,
residual structure remains; that is intentional and demonstrates where linear
assumptions break down.
