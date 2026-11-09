#!/usr/bin/env python3
"""Derive Equal and Oracle memory budgets for the allocation-policy baselines.

Second major revision, reviewer comment C ("baselines that isolate the value of
the predictor and of proportional allocation").  Three limited policies share
the same total budget (AMU) so that only the *split* differs:

  PRIMA  : B_i = PMU_i / sum(PMU) * AMU           (predicted demand; paper)
  Equal  : B_i = AMU / N                           (no demand information)
  Oracle : B_i = MM_i  / sum(MM)  * AMU            (measured isolated demand,
                                                    i.e. a profiling oracle)

MM_i is the measured isolated memory usage of workload i.  By default it is
read from an isolated-baseline result directory produced by
scripts/runners/yolo/run_isolated_baseline.sh (pidstat peak, averaged over
repeats).  It can also be given explicitly with --measured.

The output is an env file that scripts/experiments/run_allocation_baselines.sh
sources and passes to run_concurrent_4mode.sh (EQUAL_BUDGET, ORACLE_BUDGETS,
TOTAL_BUDGET_MIB), plus a CSV documenting every budget for the paper.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

WORKLOADS = ["classify", "detect", "pose", "segment", "obb"]
LABEL_TO_WORKLOAD = {"CLS": "classify", "DET": "detect", "EST": "pose", "SEG": "segment", "OBB": "obb"}
WORKLOAD_TO_LABEL = {v: k for k, v in LABEL_TO_WORKLOAD.items()}

# PRIMA budgets used by the paper runners (MiB).
PRIMA_BUDGETS_MIB = {"classify": 821, "detect": 1071, "pose": 1068, "segment": 1649, "obb": 2233}

MIB_PER_GB = 1024.0  # pidstat column is GB (GiB-based: memory_pct * 8 / 100)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--isolated-root",
        help="Isolated-baseline result root (contains organized/isolated_no_memlimit_pidstat_memory_peak_summary.csv).",
    )
    source.add_argument(
        "--measured",
        help='Explicit measured isolated usage in MiB, e.g. "classify=600 detect=800 pose=800 segment=900 obb=2000".',
    )
    parser.add_argument(
        "--statistic",
        choices=["max", "avg"],
        default="max",
        help="Which pidstat statistic represents MM_i when reading --isolated-root (default: max = peak).",
    )
    parser.add_argument(
        "--total-budget-mib",
        type=int,
        default=sum(PRIMA_BUDGETS_MIB.values()),
        help="AMU distributed by every limited policy (default: sum of PRIMA budgets = %(default)s MiB).",
    )
    parser.add_argument("--env-output", required=True, help="Env file consumed by the experiment wrapper.")
    parser.add_argument("--csv-output", default=None, help="Optional CSV with all budgets per policy.")
    return parser.parse_args()


def read_isolated_peaks(root: Path, statistic: str) -> dict[str, float]:
    summary = root / "organized" / "isolated_no_memlimit_pidstat_memory_peak_summary.csv"
    if not summary.exists():
        raise SystemExit(f"missing {summary}; run run_isolated_baseline.sh first")
    column = {"max": "max_pidstat_memory_gb_8gb", "avg": "avg_pidstat_memory_gb_8gb"}[statistic]
    values: dict[str, list[float]] = defaultdict(list)
    with summary.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            workload = LABEL_TO_WORKLOAD.get(row["workload"], row["workload"])
            values[workload].append(float(row[column]) * MIB_PER_GB)
    missing = [w for w in WORKLOADS if w not in values]
    if missing:
        raise SystemExit(f"isolated results lack workloads: {missing}")
    return {w: sum(v) / len(v) for w, v in values.items()}


def parse_measured(spec: str) -> dict[str, float]:
    measured: dict[str, float] = {}
    for pair in spec.split():
        name, _, value = pair.partition("=")
        name = LABEL_TO_WORKLOAD.get(name.upper(), name.lower())
        measured[name] = float(value.rstrip("mM"))
    missing = [w for w in WORKLOADS if w not in measured]
    if missing:
        raise SystemExit(f"--measured lacks workloads: {missing}")
    return measured


def proportional(measured: dict[str, float], total_mib: int) -> dict[str, int]:
    total_measured = sum(measured[w] for w in WORKLOADS)
    budgets = {w: int(round(measured[w] / total_measured * total_mib)) for w in WORKLOADS}
    # Keep the sum exactly equal to total_mib after rounding.
    drift = total_mib - sum(budgets.values())
    budgets[max(WORKLOADS, key=lambda w: measured[w])] += drift
    return budgets


def main() -> None:
    args = parse_args()
    if args.isolated_root:
        measured = read_isolated_peaks(Path(args.isolated_root), args.statistic)
        source = f"{args.isolated_root} (pidstat {args.statistic})"
    else:
        measured = parse_measured(args.measured)
        source = "explicit --measured"

    total = args.total_budget_mib
    equal = total // len(WORKLOADS)
    oracle = proportional(measured, total)

    under = [w for w in WORKLOADS if oracle[w] < measured[w]]
    if under:
        print(f"[WARN] oracle budget below measured usage for {under}; total budget {total} MiB is smaller than demand", file=sys.stderr)

    env_lines = [
        f"# generated by compute_allocation_baselines.py from {source}",
        f"TOTAL_BUDGET_MIB={total}",
        f"EQUAL_BUDGET={equal}m",
        "ORACLE_BUDGETS=\"" + " ".join(f"{w}={oracle[w]}m" for w in WORKLOADS) + "\"",
    ]
    env_path = Path(args.env_output)
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text("\n".join(env_lines) + "\n", encoding="utf-8")

    rows = []
    for w in WORKLOADS:
        rows.append(
            {
                "workload": WORKLOAD_TO_LABEL[w],
                "workload_name": w,
                "measured_isolated_mib": f"{measured[w]:.1f}",
                "measured_share": f"{measured[w] / sum(measured.values()):.4f}",
                "prima_budget_mib": PRIMA_BUDGETS_MIB[w],
                "prima_share": f"{PRIMA_BUDGETS_MIB[w] / sum(PRIMA_BUDGETS_MIB.values()):.4f}",
                "equal_budget_mib": equal,
                "oracle_budget_mib": oracle[w],
                "total_budget_mib": total,
            }
        )
    if args.csv_output:
        csv_path = Path(args.csv_output)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(json.dumps({"source": source, "total_budget_mib": total, "equal_budget_mib": equal,
                      "oracle_budget_mib": oracle, "prima_budget_mib": PRIMA_BUDGETS_MIB,
                      "measured_isolated_mib": {w: round(v, 1) for w, v in measured.items()},
                      "env_output": str(env_path)}, indent=2))


if __name__ == "__main__":
    main()
