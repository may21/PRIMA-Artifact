#!/usr/bin/env python3
import csv
import os
import re
from pathlib import Path


BASE = Path(os.environ.get("CLIP_ROOT", "/home/orin1/prima_artifacts/clip"))
RESULTS = BASE / "results"
MEM_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)(KiB|MiB|GiB)\s*/")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def to_mib(value, unit):
    value = float(value)
    if unit == "KiB":
        return value / 1024.0
    if unit == "GiB":
        return value * 1024.0
    return value


def model_name(path):
    name = path.name.lower()
    if "rn50" in name:
        return "RN50"
    if "vitb32" in name:
        return "ViT-B-32"
    return path.name


def read_key_values(path):
    values = {}
    if not path.exists():
        return values
    for line in path.read_text(errors="replace").splitlines():
        if "=" in line:
            key, value = line.strip().split("=", 1)
            values[key] = value
        elif line.startswith("num_images:"):
            values["num_images"] = line.split(":", 1)[1].strip()
    return values


def read_tegrastats_summary(run_dir):
    summary_path = run_dir / "tegrastats_summary.csv"
    if summary_path.exists():
        with summary_path.open(newline="", encoding="utf-8") as stream:
            return {key: value for key, value in csv.reader(stream)}

    tegra_path = run_dir / "system_tegrastats.txt"
    ram_values = []
    if tegra_path.exists():
        ram_re = re.compile(r"RAM\s+(\d+)/")
        for line in tegra_path.read_text(errors="replace").splitlines():
            match = ram_re.search(line)
            if match:
                ram_values.append(float(match.group(1)))
    return {
        "peak_ram_mb": max(ram_values) if ram_values else "",
        "num_samples": len(ram_values),
    }


def docker_memory_peaks(run_dir):
    peaks = []
    path = run_dir / "docker_stats.txt"
    if not path.exists():
        return peaks
    for line in path.read_text(errors="replace").splitlines():
        match = MEM_RE.search(ANSI_RE.sub("", line))
        if match:
            peaks.append(to_mib(*match.groups()))
    return peaks


def main():
    run_dirs = sorted(
        p
        for p in RESULTS.glob("clip_*_docker_*")
        if p.is_dir()
    )
    if not run_dirs:
        raise SystemExit(f"no CLIP result directories found under {RESULTS}")

    rows = []
    repeat_by_model = {}
    for run_dir in run_dirs:
        model = model_name(run_dir)
        repeat_by_model[model] = repeat_by_model.get(model, 0) + 1
        workload_summary = read_key_values(run_dir / "summary.txt")
        tegra_summary = read_tegrastats_summary(run_dir)
        peaks = docker_memory_peaks(run_dir)
        rows.append(
            {
                "model": model,
                "repeat": repeat_by_model[model],
                "result_dir": str(run_dir.relative_to(BASE)),
                "exit_status": workload_summary.get("exit_status", ""),
                "num_images": workload_summary.get("num_images", ""),
                "elapsed_sec": workload_summary.get("elapsed_sec", ""),
                "peak_system_ram_mb": tegra_summary.get("peak_ram_mb", ""),
                "peak_container_memory_mib": max(peaks) if peaks else "",
                "container_memory_status": (
                    "docker_stats" if peaks else "missing_docker_stats"
                ),
                "tegrastats_samples": tegra_summary.get("num_samples", ""),
            }
        )

    raw_path = BASE / "clip_memory_peak_raw.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    avg_rows = []
    for model in sorted({row["model"] for row in rows}):
        selected = [row for row in rows if row["model"] == model]
        system_peaks = [
            float(row["peak_system_ram_mb"])
            for row in selected
            if row["peak_system_ram_mb"] != ""
        ]
        container_peaks = [
            float(row["peak_container_memory_mib"])
            for row in selected
            if row["peak_container_memory_mib"] != ""
        ]
        avg_rows.append(
            {
                "model": model,
                "repeat_count": len(selected),
                "avg_peak_system_ram_mb": (
                    sum(system_peaks) / len(system_peaks) if system_peaks else ""
                ),
                "avg_peak_container_memory_mib": (
                    sum(container_peaks) / len(container_peaks)
                    if container_peaks
                    else ""
                ),
                "container_memory_valid_repeats": len(container_peaks),
                "min_peak_system_ram_mb": min(system_peaks) if system_peaks else "",
                "max_peak_system_ram_mb": max(system_peaks) if system_peaks else "",
                "min_peak_container_memory_mib": (
                    min(container_peaks) if container_peaks else ""
                ),
                "max_peak_container_memory_mib": (
                    max(container_peaks) if container_peaks else ""
                ),
            }
        )

    avg_path = BASE / "clip_memory_peak_average.csv"
    with avg_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=avg_rows[0].keys())
        writer.writeheader()
        writer.writerows(avg_rows)

    print(raw_path)
    print(avg_path)


if __name__ == "__main__":
    main()
