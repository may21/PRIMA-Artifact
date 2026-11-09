#!/usr/bin/env python3
"""Summarize the allocation-policy baseline experiment per mode and workload.

Reads the raw CSVs written by collect_concurrent_results.py and
organize_concurrent_results.py under RESULT_ROOT and writes

  organized/allocation_baselines_summary.csv
  organized/allocation_baselines_summary.md

with, for every (mode, workload): completed/attempted runs, OOM kills,
mean and SD inference time (ms), p50/p99, mean page faults per second, and
per-mode total elapsed time.  Modes are reported in the order
ts_unlimited (None), ts_equal (Equal), ts_oracle (Oracle), ts_memlimit (PRIMA),
then the MPS counterparts.
"""

from __future__ import annotations

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

WORKLOAD_LABELS = {"classify": "CLS", "detect": "DET", "pose": "EST", "segment": "SEG", "obb": "OBB"}
WORKLOAD_ORDER = ["CLS", "DET", "EST", "SEG", "OBB"]
MODE_ORDER = [
    "ts_unlimited", "ts_equal", "ts_oracle", "ts_memlimit",
    "mps20_unlimited", "mps20_equal", "mps20_oracle", "mps20_memlimit",
]
POLICY_LABEL = {
    "unlimited": "None", "equal": "Equal", "oracle": "Oracle", "memlimit": "PRIMA",
}


def policy_of(mode: str) -> str:
    return POLICY_LABEL.get(mode.rsplit("_", 1)[-1], mode)


def label(workload: str) -> str:
    return WORKLOAD_LABELS.get(workload, workload)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    low, high = int(index), min(int(index) + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def fmt(value, digits=1) -> str:
    if value is None or value != value:
        return ""
    return f"{value:.{digits}f}"


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} RESULT_ROOT")
    root = Path(sys.argv[1]).resolve()
    organized = root / "organized"
    organized.mkdir(exist_ok=True)

    inference = read_csv(root / "inference_times_raw.csv")
    runs = read_csv(root / "summary.csv")
    batch_times = read_csv(root / "batch_times_raw.csv")
    perf = read_csv(organized / "perf_by_workload_raw.csv")

    latency = defaultdict(list)
    for row in inference:
        latency[(row["mode"], label(row["workload"]))].append(float(row["inference_ms"]))

    attempts = defaultdict(int)
    completed = defaultdict(int)
    oom = defaultdict(int)
    limits = {}
    for row in runs:
        key = (row["mode"], label(row["workload"]))
        attempts[key] += 1
        completed[key] += int(str(row["exit_status"]) == "0")
        oom[key] += int(str(row["oom_killed"]).lower() == "true")
        limits[key] = row["memory_limit"]

    faults = defaultdict(list)
    for row in perf:
        value = row.get("page_faults", "")
        if value not in ("", None):
            try:
                faults[(row["mode"], row["workload"])].append(float(value))
            except ValueError:
                pass

    elapsed = defaultdict(list)
    for row in batch_times:
        if row["duration_seconds"]:
            elapsed[row["mode"]].append(float(row["duration_seconds"]))

    modes_present = {row["mode"] for row in runs} | {m for m, _ in latency}
    modes = [m for m in MODE_ORDER if m in modes_present] + sorted(modes_present - set(MODE_ORDER))

    rows = []
    for mode in modes:
        for workload in WORKLOAD_ORDER:
            key = (mode, workload)
            values = latency.get(key, [])
            rows.append(
                {
                    "mode": mode,
                    "policy": policy_of(mode),
                    "workload": workload,
                    "memory_limit": limits.get(key, ""),
                    "runs_attempted": attempts.get(key, 0),
                    "runs_completed": completed.get(key, 0),
                    "oom_killed_runs": oom.get(key, 0),
                    "inferences": len(values),
                    "mean_ms": fmt(statistics.fmean(values)) if values else "",
                    "sd_ms": fmt(statistics.pstdev(values)) if len(values) > 1 else "",
                    "p50_ms": fmt(percentile(values, 0.50)) if values else "",
                    "p99_ms": fmt(percentile(values, 0.99)) if values else "",
                    "page_faults_per_sec_mean": fmt(statistics.fmean(faults[key])) if faults.get(key) else "",
                    "batch_elapsed_s_mean": fmt(statistics.fmean(elapsed[mode])) if elapsed.get(mode) else "",
                }
            )

    fieldnames = list(rows[0].keys()) if rows else ["mode"]
    with (organized / "allocation_baselines_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "| Policy | Mode | WL | Limit | Completed | OOM | Mean (ms) | SD (ms) | p50 | p99 | PF/s |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['policy']} | {row['mode']} | {row['workload']} | {row['memory_limit']} | "
            f"{row['runs_completed']}/{row['runs_attempted']} | {row['oom_killed_runs']} | "
            f"{row['mean_ms']} | {row['sd_ms']} | {row['p50_ms']} | {row['p99_ms']} | "
            f"{row['page_faults_per_sec_mean']} |"
        )
    lines.append("")
    lines.append("| Mode | Policy | Total elapsed per batch (s, mean) |")
    lines.append("|---|---|---|")
    for mode in modes:
        lines.append(f"| {mode} | {policy_of(mode)} | {fmt(statistics.fmean(elapsed[mode])) if elapsed.get(mode) else ''} |")
    (organized / "allocation_baselines_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
