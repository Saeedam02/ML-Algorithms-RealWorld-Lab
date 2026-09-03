"""Run every experiment and refresh all committed output artifacts."""

from __future__ import annotations

import json

from ml_realworld.experiments.classification import run as run_classification
from ml_realworld.experiments.linear_regression import run as run_linear_regression


def main() -> None:
    regression = run_linear_regression()
    classification = run_classification()
    print(
        json.dumps(
            {
                "linear_regression": regression,
                "classification": classification,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
