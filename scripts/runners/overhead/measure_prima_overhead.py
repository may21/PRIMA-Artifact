#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prima.calculator import calculate_budget_plan
from prima.features.extract_features import extract_global_features
from prima.predictor.rf import RFClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--model", default="prima/predictor/rf_model.pkl")
    parser.add_argument("--feature-order", default="prima/predictor/feature_order.json")
    parser.add_argument("--available-mb", type=float, default=7770)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--precision", default="fp32")
    return parser.parse_args()


def read_models(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {"workload", "onnx_path"}
    missing = required - set(rows[0]) if rows else required
    if missing:
        raise SystemExit(f"missing columns in {path}: {sorted(missing)}")
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(raw_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[float]] = {}
    for row in raw_rows:
        groups.setdefault((str(row["workload"]), str(row["stage"])), []).append(
            float(row["elapsed_ms"])
        )
    out = []
    for (workload, stage), values in sorted(groups.items()):
        out.append(
            {
                "workload": workload,
                "stage": stage,
                "samples": len(values),
                "mean_ms": statistics.mean(values),
                "stdev_ms": statistics.stdev(values) if len(values) >= 2 else 0.0,
                "min_ms": min(values),
                "max_ms": max(values),
            }
        )
    return out


def main() -> None:
    args = parse_args()
    models = read_models(Path(args.model_csv))
    rf = RFClient(args.model, args.feature_order)
    raw_rows: list[dict[str, object]] = []

    for repeat in range(1, args.repeats + 1):
        predicted_mb: dict[str, int] = {}
        for row in models:
            workload = row["workload"]
            onnx_path = row["onnx_path"]

            start = time.perf_counter()
            features = extract_global_features(
                onnx_path,
                workload,
                args.batch,
                args.height,
                args.width,
                args.precision,
            )
            feature_ms = (time.perf_counter() - start) * 1000.0

            start = time.perf_counter()
            predicted = rf.predict_max_mem_mb(features)
            predict_ms = (time.perf_counter() - start) * 1000.0
            predicted_mb[workload] = predicted

            raw_rows.append(
                {
                    "repeat": repeat,
                    "workload": workload,
                    "stage": "feature_extraction",
                    "elapsed_ms": feature_ms,
                    "predicted_mb": predicted,
                }
            )
            raw_rows.append(
                {
                    "repeat": repeat,
                    "workload": workload,
                    "stage": "rf_prediction",
                    "elapsed_ms": predict_ms,
                    "predicted_mb": predicted,
                }
            )

        for count in range(1, len(models) + 1):
            selected = dict(list(predicted_mb.items())[:count])
            start = time.perf_counter()
            calculate_budget_plan(selected, args.available_mb)
            calc_ms = (time.perf_counter() - start) * 1000.0
            raw_rows.append(
                {
                    "repeat": repeat,
                    "workload": f"{count}_workloads",
                    "stage": "calculator",
                    "elapsed_ms": calc_ms,
                    "predicted_mb": "",
                }
            )

    output_dir = Path(args.output_dir)
    write_csv(output_dir / "overhead_raw.csv", raw_rows)
    write_csv(output_dir / "overhead_summary.csv", summarize(raw_rows))
    print(output_dir / "overhead_raw.csv")
    print(output_dir / "overhead_summary.csv")


if __name__ == "__main__":
    main()
