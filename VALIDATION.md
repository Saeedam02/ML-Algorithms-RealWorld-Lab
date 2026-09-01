# Validation

Reference build validation for v1.0.0.

## Deterministic checks

- Python compilation: PASS
- Test suite: **7 passed**
- Branch-aware core coverage: **91.30%**
- Configured minimum coverage: **80%**
- Full benchmark execution: PASS
- Generated regression figures: PASS
- Generated classification figures: PASS

## Reference benchmark

The committed outputs were generated from seed `42` using:

```bash
python scripts/generate_data.py
python scripts/run_all.py
pytest --cov=ml_realworld --cov-branch --cov-report=term-missing
```

Timing values in the classifier metrics are machine-dependent. Predictive metrics
should be stable for the pinned random seed and compatible library versions.

The experiment/plot orchestration modules are intentionally excluded from the
coverage threshold; their output is validated by the full benchmark run. Core
data, preprocessing, model factory and metric logic remain coverage-tracked.
