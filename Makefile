.PHONY: install data run test lint clean

install:
	python -m pip install -e ".[dev]"

data:
	python scripts/generate_data.py

run:
	python scripts/run_all.py

test:
	pytest --cov=ml_realworld --cov-branch --cov-report=term-missing

lint:
	ruff check .

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
