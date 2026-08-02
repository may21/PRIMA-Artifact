#!/usr/bin/env python3
"""Inspect the serialized PMU predictor model artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        nargs="?",
        default="prima/predictor/rf_model.pkl",
        help="Path to the serialized predictor model.",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    model = joblib.load(model_path)
    estimator = model
    if hasattr(model, "named_steps") and "clf" in model.named_steps:
        estimator = model.named_steps["clf"]

    print(f"path={model_path}")
    print(f"container_type={type(model).__module__}.{type(model).__name__}")
    print(f"estimator_type={type(estimator).__module__}.{type(estimator).__name__}")

    if hasattr(estimator, "get_params"):
        params = estimator.get_params()
        for key in ("n_estimators", "max_depth", "learning_rate", "objective", "random_state", "n_jobs"):
            if key in params:
                print(f"{key}={params[key]}")


if __name__ == "__main__":
    main()
