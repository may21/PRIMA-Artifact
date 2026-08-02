#!/usr/bin/env python3
"""Train the PRIMA PMU predictor from a feature CSV."""

from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_FEATURE_ORDER = [
    "workload",
    "model_size",
    "batch",
    "H",
    "W",
    "precision",
    "num_inputs",
    "num_outputs",
    "total_input_bytes",
    "total_output_bytes",
    "weight_bytes",
    "param_count",
    "op_Add",
    "op_Concat",
    "op_Constant",
    "op_Conv",
    "op_GlobalAveragePool",
    "op_Mul",
    "op_Reshape",
    "op_Sigmoid",
    "op_Split",
]

CATEGORICAL = ["workload", "model_size", "precision"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Training CSV with feature columns and target.")
    parser.add_argument("--target", default="Max_cpu_mem_MB")
    parser.add_argument("--output", default="prima/predictor/rf_model.pkl")
    parser.add_argument("--feature-order-output", default="prima/predictor/feature_order.json")
    parser.add_argument("--metadata-output", default=None, help="Optional path for training metadata JSON.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--cv-splits", type=int, default=5)
    return parser.parse_args()


def build_pipeline(feature_order: list[str], target: str, n_estimators: int, random_state: int) -> Pipeline:
    cat_cols = [c for c in CATEGORICAL if c in feature_order]
    num_cols = [c for c in feature_order if c not in cat_cols and c != target]
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(with_mean=False), num_cols),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )
    return Pipeline(
        [
            ("prep", preprocessor),
            (
                "clf",
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=None,
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    output_path = Path(args.output)
    feature_order_path = Path(args.feature_order_output)
    metadata_path = Path(args.metadata_output) if args.metadata_output else None

    df = pd.read_csv(csv_path)
    missing = [c for c in DEFAULT_FEATURE_ORDER + [args.target] if c not in df.columns]
    if missing:
        raise SystemExit(f"missing columns in {csv_path}: {missing}")

    X = df[DEFAULT_FEATURE_ORDER].copy()
    y = df[args.target].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    model = build_pipeline(DEFAULT_FEATURE_ORDER, args.target, args.n_estimators, args.random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))
    cv = KFold(n_splits=args.cv_splits, shuffle=True, random_state=args.random_state)
    cv_mae = float(-cross_val_score(model, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error").mean())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_order_path.parent.mkdir(parents=True, exist_ok=True)
    if metadata_path:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, output_path)
    feature_order_path.write_text(
        json.dumps({"feature_order": DEFAULT_FEATURE_ORDER, "categorical": CATEGORICAL}, indent=2) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "model_family": "RandomForestRegressor",
        "container_type": "sklearn.pipeline.Pipeline",
        "estimator_type": "sklearn.ensemble.RandomForestRegressor",
        "training_csv": str(csv_path),
        "source_csv": str(csv_path),
        "training_host": socket.gethostname(),
        "target": args.target,
        "rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "n_estimators": args.n_estimators,
        "max_depth": None,
        "random_state": args.random_state,
        "test_size": args.test_size,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "cv_mae": cv_mae,
    }
    if metadata_path:
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
