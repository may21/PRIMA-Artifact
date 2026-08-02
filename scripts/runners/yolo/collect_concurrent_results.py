#!/usr/bin/env python3
import csv
import re
import sys
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
IMAGE_RE = re.compile(
    r"^image\s+(?P<image_index>\d+)/(?P<image_count>\d+)\s+.*?"
    r"(?P<inference_ms>\d+(?:\.\d+)?)ms\s*$"
)
NUMBER_RE = re.compile(r"^\d+(?:\.\d+)?$")
TEGRA_RE = re.compile(
    r"^(?P<date>\d{2}-\d{2}-\d{4})\s+(?P<time>\d{2}:\d{2}:\d{2})\s+"
    r"RAM\s+(?P<ram_used>\d+)/(?P<ram_total>\d+)MB.*?"
    r"SWAP\s+(?P<swap_used>\d+)/(?P<swap_total>\d+)MB.*?"
    r"CPU\s+\[(?P<cpu>[^\]]+)\]\s+"
    r"GR3D_FREQ\s+(?P<gr3d>\d+)%.*?"
    r"gpu@(?P<gpu_temp>\d+(?:\.\d+)?)C.*?"
    r"VDD_IN\s+(?P<power_now>\d+)mW/(?P<power_avg>\d+)mW"
)


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def context_from_run(run_dir):
    return {
        "mode": run_dir.parent.name,
        "repeat": int(run_dir.name.removeprefix("run")),
    }


def parse_output(path):
    inference_rows = []
    elapsed_seconds = ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in lines:
        line = ANSI_RE.sub("", line).strip()
        match = IMAGE_RE.match(line)
        if match:
            inference_rows.append(
                {
                    "image_index": int(match.group("image_index")),
                    "image_count": int(match.group("image_count")),
                    "inference_ms": float(match.group("inference_ms")),
                }
            )
        elif NUMBER_RE.match(line):
            elapsed_seconds = float(line)
    return inference_rows, elapsed_seconds


VALID_WORKLOADS = {"classify", "detect", "pose", "segment", "obb"}


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} RESULT_ROOT")

    root = Path(sys.argv[1]).resolve()
    if not root.is_dir():
        raise SystemExit(f"result root not found: {root}")

    summary_path = root / "summary.csv"
    summary_by_key = {}
    if summary_path.exists():
        with summary_path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                summary_by_key[(row["mode"], int(row["repeat"]), row["workload"])] = row

    inference_rows = []
    elapsed_rows = []
    perf_rows = []
    tegra_rows = []
    vmstat_rows = []
    mpstat_rows = []
    pidstat_rows = []
    batch_time_rows = []

    for run_dir in sorted(root.glob("*/run*")):
        context = context_from_run(run_dir)

        time_path = run_dir / "time.csv"
        if time_path.exists():
            events = {}
            with time_path.open(newline="", encoding="utf-8") as stream:
                for row in csv.DictReader(stream):
                    events[row["event"]] = int(row["epoch_ns"])
            batch_time_rows.append(
                {
                    **context,
                    "start_epoch_ns": events.get("start", ""),
                    "end_epoch_ns": events.get("end", ""),
                    "duration_seconds": (
                        (events["end"] - events["start"]) / 1e9
                        if "start" in events and "end" in events
                        else ""
                    ),
                }
            )

        for workload_dir in sorted(path for path in run_dir.iterdir() if path.is_dir()):
            workload = workload_dir.name
            if workload not in VALID_WORKLOADS:
                continue
            output_path = workload_dir / "output.txt"
            parsed_inference = []
            elapsed_seconds = ""
            if output_path.exists():
                parsed_inference, elapsed_seconds = parse_output(output_path)
                for row in parsed_inference:
                    inference_rows.append(
                        {
                            **context,
                            "workload": workload,
                            **row,
                            "source_file": str(output_path.relative_to(root)),
                        }
                    )

            perf_path = workload_dir / "perf.csv"
            perf_count = 0
            if perf_path.exists():
                for line_number, raw_line in enumerate(
                    perf_path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
                ):
                    if not raw_line.strip() or raw_line.lstrip().startswith("#"):
                        continue
                    parts = [part.strip() for part in raw_line.split(",")]
                    if len(parts) < 6:
                        continue
                    perf_count += 1
                    perf_rows.append(
                        {
                            **context,
                            "workload": workload,
                            "line_number": line_number,
                            "interval_seconds": parts[0],
                            "value": parts[1],
                            "event": parts[3],
                            "runtime_ns": parts[4],
                            "running_percent": parts[5],
                            "raw_line": raw_line,
                        }
                    )

            summary = summary_by_key.get(
                (context["mode"], context["repeat"], workload), {}
            )
            elapsed_rows.append(
                {
                    **context,
                    "workload": workload,
                    "memory_limit": summary.get("memory_limit", ""),
                    "mps_percentage": summary.get("mps_percentage", ""),
                    "exit_status": summary.get("exit_status", ""),
                    "elapsed_seconds": elapsed_seconds,
                    "inference_raw_rows": len(parsed_inference),
                    "perf_raw_rows": perf_count,
                    "perf_attached": (workload_dir / "python_pid.txt").exists(),
                    "output_file": str(output_path.relative_to(root)),
                }
            )

        tegra_path = run_dir / "system_tegrastats.txt"
        if tegra_path.exists():
            for sample_index, raw_line in enumerate(
                tegra_path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
            ):
                match = TEGRA_RE.search(raw_line)
                tegra_rows.append(
                    {
                        **context,
                        "sample_index": sample_index,
                        "date": match.group("date") if match else "",
                        "time": match.group("time") if match else "",
                        "ram_used_mb": match.group("ram_used") if match else "",
                        "ram_total_mb": match.group("ram_total") if match else "",
                        "swap_used_mb": match.group("swap_used") if match else "",
                        "swap_total_mb": match.group("swap_total") if match else "",
                        "cpu": match.group("cpu") if match else "",
                        "gr3d_percent": match.group("gr3d") if match else "",
                        "gpu_temp_c": match.group("gpu_temp") if match else "",
                        "vdd_in_now_mw": match.group("power_now") if match else "",
                        "vdd_in_avg_mw": match.group("power_avg") if match else "",
                        "raw_line": raw_line,
                    }
                )

        vmstat_path = run_dir / "system_vmstat.txt"
        if vmstat_path.exists():
            names = [
                "r", "b", "swpd", "free", "buff", "cache", "si", "so", "bi", "bo",
                "in", "cs", "us", "sy", "id", "wa", "st", "date", "time"
            ]
            for line_number, raw_line in enumerate(
                vmstat_path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
            ):
                parts = raw_line.split()
                if len(parts) != len(names) or not parts[0].isdigit():
                    continue
                vmstat_rows.append(
                    {**context, "line_number": line_number, **dict(zip(names, parts))}
                )

        mpstat_path = run_dir / "system_mpstat.txt"
        if mpstat_path.exists():
            names = [
                "time", "cpu", "usr", "nice", "sys", "iowait", "irq", "soft",
                "steal", "guest", "gnice", "idle"
            ]
            for line_number, raw_line in enumerate(
                mpstat_path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
            ):
                parts = raw_line.split()
                if len(parts) != len(names) or not re.match(r"^\d{2}:\d{2}:\d{2}$", parts[0]):
                    continue
                mpstat_rows.append(
                    {**context, "line_number": line_number, **dict(zip(names, parts))}
                )

        pidstat_path = run_dir / "system_pidstat.txt"
        if pidstat_path.exists():
            for line_number, raw_line in enumerate(
                pidstat_path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
            ):
                if not re.match(r"^\d{2}:\d{2}:\d{2}\s+", raw_line):
                    continue
                pidstat_rows.append(
                    {
                        **context,
                        "line_number": line_number,
                        "raw_line": raw_line,
                    }
                )

    inference_rows.sort(
        key=lambda x: (x["mode"], x["repeat"], x["workload"], x["image_index"])
    )
    elapsed_rows.sort(key=lambda x: (x["mode"], x["repeat"], x["workload"]))
    perf_rows.sort(
        key=lambda x: (x["mode"], x["repeat"], x["workload"], x["line_number"])
    )

    write_csv(
        root / "inference_times_raw.csv",
        [
            "mode", "repeat", "workload", "image_index", "image_count",
            "inference_ms", "source_file"
        ],
        inference_rows,
    )
    write_csv(
        root / "workload_runs_raw.csv",
        [
            "mode", "repeat", "workload", "memory_limit", "mps_percentage",
            "exit_status", "elapsed_seconds", "inference_raw_rows", "perf_raw_rows",
            "perf_attached", "output_file"
        ],
        elapsed_rows,
    )
    write_csv(
        root / "perf_raw.csv",
        [
            "mode", "repeat", "workload", "line_number", "interval_seconds",
            "value", "event", "runtime_ns", "running_percent", "raw_line"
        ],
        perf_rows,
    )
    write_csv(
        root / "tegrastats_raw.csv",
        [
            "mode", "repeat", "sample_index", "date", "time", "ram_used_mb",
            "ram_total_mb", "swap_used_mb", "swap_total_mb", "cpu",
            "gr3d_percent", "gpu_temp_c", "vdd_in_now_mw", "vdd_in_avg_mw",
            "raw_line"
        ],
        tegra_rows,
    )
    write_csv(
        root / "vmstat_raw.csv",
        [
            "mode", "repeat", "line_number", "r", "b", "swpd", "free", "buff",
            "cache", "si", "so", "bi", "bo", "in", "cs", "us", "sy", "id",
            "wa", "st", "date", "time"
        ],
        vmstat_rows,
    )
    write_csv(
        root / "mpstat_raw.csv",
        [
            "mode", "repeat", "line_number", "time", "cpu", "usr", "nice",
            "sys", "iowait", "irq", "soft", "steal", "guest", "gnice", "idle"
        ],
        mpstat_rows,
    )
    write_csv(
        root / "pidstat_raw.csv",
        ["mode", "repeat", "line_number", "raw_line"],
        pidstat_rows,
    )
    write_csv(
        root / "batch_times_raw.csv",
        ["mode", "repeat", "start_epoch_ns", "end_epoch_ns", "duration_seconds"],
        batch_time_rows,
    )

    config = {}
    config_path = root / "experiment_config.txt"
    if config_path.exists():
        for line in config_path.read_text(errors="replace").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                config[key.strip()] = value.strip()
    configured_modes = config.get("modes", "ts_unlimited mps20_unlimited ts_memlimit mps20_memlimit").split()
    try:
        configured_repeats = int(config.get("repeats", "5"))
    except ValueError:
        configured_repeats = 5
    repeat_range = config.get("repeat_range", "")
    if re.fullmatch(r"\d+-\d+", repeat_range):
        repeat_start, repeat_end = map(int, repeat_range.split("-", 1))
        configured_repeats = max(0, repeat_end - repeat_start + 1)
    try:
        expected_images_per_run = int(config.get("image_limit", "100"))
    except ValueError:
        expected_images_per_run = 100
    expected_runs = len(configured_modes) * configured_repeats * 5
    expected_inference_rows = expected_runs * expected_images_per_run
    failures = [row for row in elapsed_rows if str(row["exit_status"]) != "0"]
    missing_inference = [row for row in elapsed_rows if row["inference_raw_rows"] != expected_images_per_run]
    missing_perf = [row for row in elapsed_rows if not row["perf_attached"]]

    validation = [
        ("result_root", str(root)),
        ("workload_runs", len(elapsed_rows)),
        ("expected_workload_runs", expected_runs),
        ("inference_raw_rows", len(inference_rows)),
        ("expected_inference_raw_rows", expected_inference_rows),
        ("perf_raw_rows", len(perf_rows)),
        ("tegrastats_raw_rows", len(tegra_rows)),
        ("vmstat_raw_rows", len(vmstat_rows)),
        ("mpstat_raw_rows", len(mpstat_rows)),
        ("pidstat_raw_rows", len(pidstat_rows)),
        ("failed_runs", len(failures)),
        (
            f"runs_with_inference_count_not_{expected_images_per_run}",
            len(missing_inference),
        ),
        ("runs_without_perf_attachment", len(missing_perf)),
    ]
    with (root / "validation.txt").open("w", encoding="utf-8") as stream:
        for key, value in validation:
            stream.write(f"{key}={value}\n")

    for key, value in validation:
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
