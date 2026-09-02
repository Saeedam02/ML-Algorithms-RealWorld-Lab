"""Central configuration.

Keeping experimental constants in one place makes the repository easier to
reproduce and prevents hidden "magic numbers" from being scattered throughout
the notebooks and scripts.
"""

from __future__ import annotations

from pathlib import Path

RANDOM_SEED = 42
TEST_SIZE = 0.20

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"
MODEL_CARDS_DIR = REPORTS_DIR / "model_cards"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "models"

HOUSING_CSV = DATA_DIR / "urban_housing.csv"
PHISHING_CSV = DATA_DIR / "phishing_websites.csv"

for directory in (
    DATA_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    MODEL_CARDS_DIR,
    ARTIFACTS_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)
