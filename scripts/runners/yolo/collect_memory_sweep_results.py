#!/usr/bin/env python3
import csv
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path


WORKLOAD_LABEL = {
    "classify": "CLS",
    "detect": "DET",
    "pose": "EST",
    "segment": "SEG",
    "obb": "OBB",
}
MEMORY_GB = {"0.5GB": 0.5, "1.0GB": 1.0, "1.5GB": 1.5, "2.0GB": 2.0, "2.5GB": 2.5}
IMAGE_RE = re.compile(
    r"^image\s+(?P<index>\d+)/(?P<count>\d+)\s+.*?(?P<ms>\d+(?:\.\d+)?)ms\s*$"
)
NUMBER_RE = re.compile(r"^\d+(?:\.\d+)?$")
TEGRA_RE = re.compile(
    r"RAM\s+(?P<ram>\d+)/(?P<ram_total>\d+)MB.*?"
    r"GR3D_FREQ\s+(?P<gpu>\d+)%.*?"
    r"gpu@(?P<temp>\d+(?:\.\d+)?)C.*?"
    r"VDD_IN\s+(?P<power>\d+)mW/(?P<power_avg>\d+)mW"
)


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def numeric(value):
    value = value.strip()
    if not NUMBER_RE.fullmatch(value):
        return None
    return float(value)


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def stdev(values):
    values = [v for v in values if v is not None]
    return statistics.stdev(values) if len(values) >= 2 else 0 if values else None


def ratio(hits, total):
    return hits / total * 100.0 if total and hits is not None else None


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} RESULT_ROOT")
    root = Path(sys.argv[1]).resolve()
    out = root / "organized"
    out.mkdir(exist_ok=True)

    summary_rows = []
    with (root / "summary.csv").open(newline="", encoding="utf-8") as stream:
        summary_rows = list(csv.DictReader(stream))

    run_rows = []
    inference_rows = []
    perf_interval_rows = []

    for summary in summary_rows:
        workload_name = summary["workload"]
        workload = WORKLOAD_LABEL[workload_name]
        memory_label = summary["memory_label"]
        repeat = int(summary["repeat"])
        run_dir = Path(summary["result_dir"])
        if not run_dir.is_absolute():
            run_dir = root / run_dir

        output_path = run_dir / "output.txt"
        elapsed_sec = None
        image_values = []
        if output_path.exists():
            for line in output_path.read_text(errors="replace").splitlines():
                clean = re.sub(r"\x1b\[[0-9;]*m", "", line).strip()
                match = IMAGE_RE.match(clean)
                if match:
                    row = {
                        "workload": workload,
                        "memory_label": memory_label,
                        "memory_limit_gb": MEMORY_GB[memory_label],
                        "repeat": repeat,
                        "image_index": int(match.group("index")),
                        "image_count": int(match.group("count")),
                        "inference_ms": float(match.group("ms")),
                    }
                    inference_rows.append(row)
                    image_values.append(row["inference_ms"])
                elif NUMBER_RE.fullmatch(clean):
                    elapsed_sec = float(clean)

        cgroup_peak = None
        cgroup_path = run_dir / "cgroup_memory.csv"
        if cgroup_path.exists():
            with cgroup_path.open(newline="", encoding="utf-8") as stream:
                for row in csv.DictReader(stream):
                    for key in ("memory_current_bytes", "memory_peak_bytes"):
                        value = row.get(key, "")
                        if value and value.isdigit():
                            cgroup_peak = max(cgroup_peak or 0, int(value))

        perf_totals = defaultdict(float)
        valid_intervals = 0
        perf_path = run_dir / "perf.csv"
        if perf_path.exists():
            interval_map = defaultdict(dict)
            for raw in perf_path.read_text(errors="replace").splitlines():
                parts = [part.strip() for part in raw.split(",")]
                if len(parts) < 4:
                    continue
                value = numeric(parts[1])
                event = parts[3]
                if value is None:
                    continue
                interval_map[parts[0]][event] = value
                perf_totals[event] += value
            for sample_index, (timestamp, events) in enumerate(
                sorted(interval_map.items(), key=lambda item: float(item[0])), 1
            ):
                valid_intervals += 1
                perf_interval_rows.append(
                    {
                        "workload": workload,
                        "memory_label": memory_label,
                        "memory_limit_gb": MEMORY_GB[memory_label],
                        "repeat": repeat,
                        "sample_index": sample_index,
                        "timestamp_sec": float(timestamp),
                        **{
                            event.replace("-", "_"): events.get(event, "")
                            for event in [
                                "page-faults",
                                "cache-references",
                                "cache-misses",
                                "L1-dcache-loads",
                                "L1-dcache-load-misses",
                                "L1-icache-loads",
                                "L1-icache-load-misses",
                                "l2d_cache",
                                "l2d_cache_refill",
                                "LLC-loads",
                                "LLC-load-misses",
                            ]
                        },
                    }
                )

        tegra_ram = []
        tegra_gpu = []
        tegra_temp = []
        tegra_power = []
        tegra_path = run_dir / "system_tegrastats.txt"
        if tegra_path.exists():
            for line in tegra_path.read_text(errors="replace").splitlines():
                match = TEGRA_RE.search(line)
                if not match:
                    continue
                tegra_ram.append(float(match.group("ram")))
                tegra_gpu.append(float(match.group("gpu")))
                tegra_temp.append(float(match.group("temp")))
                tegra_power.append(float(match.group("power")))

        l1d_loads = perf_totals.get("L1-dcache-loads")
        l1d_misses = perf_totals.get("L1-dcache-load-misses")
        l1i_loads = perf_totals.get("L1-icache-loads")
        l1i_misses = perf_totals.get("L1-icache-load-misses")
        l2_accesses = perf_totals.get("l2d_cache")
        l2_misses = perf_totals.get("l2d_cache_refill")
        llc_loads = perf_totals.get("LLC-loads")
        llc_misses = perf_totals.get("LLC-load-misses")

        run_rows.append(
            {
                "workload": workload,
                "memory_label": memory_label,
                "memory_limit_gb": MEMORY_GB[memory_label],
                "repeat": repeat,
                "exit_status": summary["exit_status"],
                "oom_killed": summary["oom_killed"],
                "success": summary["exit_status"] == "0",
                "raw_inference_count": len(image_values),
                "inference_mean_ms": mean(image_values),
                "inference_stdev_ms": stdev(image_values),
                "end_to_end_elapsed_sec": elapsed_sec,
                "cgroup_peak_bytes": cgroup_peak,
                "cgroup_peak_gb": cgroup_peak / 1024**3 if cgroup_peak else None,
                "tegrastats_ram_peak_mb": max(tegra_ram) if tegra_ram else None,
                "tegrastats_gpu_mean_pct": mean(tegra_gpu),
                "tegrastats_gpu_peak_pct": max(tegra_gpu) if tegra_gpu else None,
                "tegrastats_gpu_temp_peak_c": max(tegra_temp) if tegra_temp else None,
                "tegrastats_power_mean_mw": mean(tegra_power),
                "perf_valid_intervals": valid_intervals,
                "page_faults": perf_totals.get("page-faults"),
                "cache_references": perf_totals.get("cache-references"),
                "cache_misses": perf_totals.get("cache-misses"),
                "cache_miss_rate_pct": ratio(
                    perf_totals.get("cache-misses"), perf_totals.get("cache-references")
                ),
                "l1d_loads": l1d_loads,
                "l1d_misses": l1d_misses,
                "l1d_hits": l1d_loads - l1d_misses if l1d_loads and l1d_misses is not None else None,
                "l1d_hit_rate_pct": ratio(
                    l1d_loads - l1d_misses if l1d_loads and l1d_misses is not None else None,
                    l1d_loads,
                ),
                "l1i_loads": l1i_loads,
                "l1i_misses": l1i_misses,
                "l1i_hits": l1i_loads - l1i_misses if l1i_loads and l1i_misses is not None else None,
                "l1i_hit_rate_pct": ratio(
                    l1i_loads - l1i_misses if l1i_loads and l1i_misses is not None else None,
                    l1i_loads,
                ),
                "l2_accesses": l2_accesses,
                "l2_misses_refills": l2_misses,
                "l2_hits": l2_accesses - l2_misses if l2_accesses and l2_misses is not None else None,
                "l2_hit_rate_pct": ratio(
                    l2_accesses - l2_misses if l2_accesses and l2_misses is not None else None,
                    l2_accesses,
                ),
                "llc_loads": llc_loads,
                "llc_misses": llc_misses,
                "llc_hits": llc_loads - llc_misses if llc_loads and llc_misses is not None else None,
                "llc_hit_rate_pct": ratio(
                    llc_loads - llc_misses if llc_loads and llc_misses is not None else None,
                    llc_loads,
                ),
            }
        )

    condition_groups = defaultdict(list)
    for row in run_rows:
        condition_groups[(row["workload"], row["memory_label"])].append(row)

    condition_rows = []
    metrics = [
        "inference_mean_ms",
        "end_to_end_elapsed_sec",
        "cgroup_peak_gb",
        "page_faults",
        "cache_miss_rate_pct",
        "l1d_hit_rate_pct",
        "l1i_hit_rate_pct",
        "l2_hit_rate_pct",
        "llc_hit_rate_pct",
        "tegrastats_ram_peak_mb",
        "tegrastats_gpu_mean_pct",
        "tegrastats_gpu_peak_pct",
        "tegrastats_power_mean_mw",
    ]
    for (workload, memory_label), rows in condition_groups.items():
        successful = [row for row in rows if row["success"]]
        output = {
            "workload": workload,
            "memory_label": memory_label,
            "memory_limit_gb": MEMORY_GB[memory_label],
            "runs": len(rows),
            "successful_runs": len(successful),
            "oom_runs": sum(str(row["oom_killed"]).lower() == "true" for row in rows),
        }
        for metric in metrics:
            values = [row[metric] for row in successful if row[metric] is not None]
            output[f"{metric}_mean"] = mean(values)
            output[f"{metric}_stdev"] = stdev(values)
        condition_rows.append(output)

    workload_order = {"CLS": 0, "DET": 1, "EST": 2, "SEG": 3, "OBB": 4}
    run_rows.sort(key=lambda r: (workload_order[r["workload"]], r["memory_limit_gb"], r["repeat"]))
    condition_rows.sort(key=lambda r: (workload_order[r["workload"]], r["memory_limit_gb"]))
    inference_rows.sort(
        key=lambda r: (
            workload_order[r["workload"]],
            r["memory_limit_gb"],
            r["repeat"],
            r["image_index"],
        )
    )

    write_csv(out / "run_metrics.csv", list(run_rows[0]), run_rows)
    write_csv(out / "condition_summary.csv", list(condition_rows[0]), condition_rows)
    write_csv(
        out / "inference_raw.csv",
        [
            "workload",
            "memory_label",
            "memory_limit_gb",
            "repeat",
            "image_index",
            "image_count",
            "inference_ms",
        ],
        inference_rows,
    )
    if perf_interval_rows:
        write_csv(out / "cpu_perf_intervals_raw.csv", list(perf_interval_rows[0]), perf_interval_rows)

    minimum_rows = []
    for workload in workload_order:
        candidates = [
            row
            for row in condition_rows
            if row["workload"] == workload and row["successful_runs"] == 3
        ]
        minimum = min(candidates, key=lambda r: r["memory_limit_gb"]) if candidates else None
        minimum_rows.append(
            {
                "workload": workload,
                "minimum_memory_gb_with_3_of_3_success": (
                    minimum["memory_limit_gb"] if minimum else ""
                ),
                "condition": minimum["memory_label"] if minimum else "",
            }
        )
    write_csv(
        out / "minimum_memory_thresholds.csv",
        ["workload", "minimum_memory_gb_with_3_of_3_success", "condition"],
        minimum_rows,
    )

    validation = {
        "runs": len(run_rows),
        "successful_runs": sum(row["success"] for row in run_rows),
        "oom_runs": sum(str(row["oom_killed"]).lower() == "true" for row in run_rows),
        "raw_inference_rows": len(inference_rows),
        "conditions": len(condition_rows),
        "runs_with_expected_100_inferences": sum(row["raw_inference_count"] == 100 for row in run_rows),
    }
    with (out / "validation.txt").open("w") as stream:
        for key, value in validation.items():
            stream.write(f"{key}={value}\n")
            print(f"{key}={value}")


if __name__ == "__main__":
    main()
