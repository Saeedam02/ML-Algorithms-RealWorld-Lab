# Dataset Design

## Why the repository uses reproducible synthetic benchmarks

Many popular tutorial datasets are tiny, clean, linearly separable, or so widely
used that model results say little about practical ML engineering.

This repository instead generates two larger **synthetic but realistic**
benchmarks. The advantage is that the entire repo is:

- offline and self-contained,
- license-safe,
- exactly reproducible,
- large enough to expose compute trade-offs,
- intentionally messy enough to require real preprocessing.

The data is synthetic. Do not represent rows as real houses, users, domains, or
security incidents.

## Urban Housing Benchmark

**12,000 rows, 19 predictors + target**

The problem combines property characteristics, neighborhood, infrastructure,
energy, crime, school and macroeconomic variables. It includes:

- numeric + categorical features,
- missing values,
- price outliers,
- heteroscedastic noise,
- nonlinear interaction effects.

Target: `sale_price`.

## Phishing Website Benchmark

**18,000 rows, 30 predictors + target**

Features imitate common URL/domain/TLS/content indicators:

- URL length and composition,
- suspicious top-level domains,
- IP-address URLs,
- HTTPS and certificate signals,
- domain age,
- redirects,
- external-link ratios,
- form actions,
- iframes/popups,
- URL entropy,
- brand mismatch,
- page reputation proxies,
- page type and TLS issuer.

Target: `is_phishing`.

The target includes nonlinear interactions and 2.5% label noise to prevent a
trivial classification exercise.

## Optional real-data extensions

If you want to extend the repository, good public analogues are:

- Ames Housing / OpenML `house_prices` (dataset 42165).
- UCI Phishing Websites (dataset 327, 11,055 instances, 30 features).
- UCI PhiUSIIL Phishing URL (dataset 967, 235,795 instances, 54 features).

Keep real-data adapters separate from the default benchmark so CI remains fully
offline and deterministic.
