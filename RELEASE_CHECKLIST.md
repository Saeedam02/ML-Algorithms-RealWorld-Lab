# Release Checklist

Before tagging a release:

- [ ] `python scripts/generate_data.py`
- [ ] `python scripts/run_all.py`
- [ ] `python -m compileall -q src tests scripts`
- [ ] `ruff check .`
- [ ] `pytest --cov=ml_realworld --cov-branch --cov-report=term-missing`
- [ ] Inspect every figure in `reports/figures/`
- [ ] Confirm README metrics match `reports/metrics/`
- [ ] Confirm synthetic-data disclaimer remains visible
- [ ] Update `CHANGELOG.md`
- [ ] Update version in `pyproject.toml` and `src/ml_realworld/__init__.py`
- [ ] Create Git tag, e.g. `v1.0.0`
