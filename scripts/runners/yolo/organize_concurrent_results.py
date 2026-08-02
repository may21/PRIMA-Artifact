#!/usr/bin/env python3
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path


WORKLOAD_LABELS = {
    "classify": "CLS",
    "detect": "DET",
    "pose": "EST",
    "segment": "SEG",
    "obb": "OBB",
}
WORKLOAD_ORDER = {"CLS": 0, "DET": 1, "EST": 2, "SEG": 3, "OBB": 4}
MODE_ORDER = {
    "ts_unlimited": 0,
    "mps20_unlimited": 1,
    "ts_memlimit": 2,
    "mps20_memlimit": 3,
}
TEGRА_RE = re.compile(
    r"^(?P<date>\d{2}-\d{2}-\d{4})\s+(?P<time>\d{2}:\d{2}:\d{2})\s+"
    r"RAM\s+(?P<ram_used>\d+)/(?P<ram_total>\d+)MB.*?"
    r"(?:SWAP\s+(?P<swap_used>\d+)/(?P<swap_total>\d+)MB.*?)?"
    r"CPU\s+\[(?P<cpu>[^\]]+)\]\s+"
    r"GR3D_FREQ\s+(?P<gpu_usage>\d+)%.*?"
    r"gpu@(?P<gpu_temp>\d+(?:\.\d+)?)C.*?"
    r"VDD_IN\s+(?P<power_now>\d+)mW/(?P<power_avg>\d+)mW"
)


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def number(value):
    value = value.strip()
    if not re.fullmatch(r"\d+(?:\.\d+)?", value):
        return None
    return float(value) if "." in value else int(value)


def run_context(run_dir):
    return run_dir.parent.name, int(run_dir.name.removeprefix("run"))


def sorted_rows(rows, extra=()):
    return sorted(
        rows,
        key=lambda row: (
            MODE_ORDER.get(row["mode"], 99),
            row["repeat"],
            *[row[key] for key in extra],
        ),
    )


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} RESULT_ROOT")

    root = Path(sys.argv[1]).resolve()
    output_dir = root / "organized"
    output_dir.mkdir(exist_ok=True)

    workload_cpu_mem = []
    system_cpu = []
    tegrastats = []
    top_memory = []
    perf_rows = []
    perf_map = []
    expected_pids = {}

    run_dirs = sorted(
        root.glob("*/run*"),
        key=lambda p: (
            MODE_ORDER.get(p.parent.name, 99),
            int(p.name.removeprefix("run")),
        ),
    )

    for run_dir in run_dirs:
        mode, repeat = run_context(run_dir)

        for workload_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
            workload_name = workload_dir.name
            if workload_name not in WORKLOAD_LABELS:
                continue
            workload = WORKLOAD_LABELS[workload_name]
            pid_path = workload_dir / "python_pid.txt"
            pid = int(pid_path.read_text().strip()) if pid_path.exists() else None
            expected_pids[(mode, repeat, workload_name)] = pid

            config = {}
            config_path = workload_dir / "config.txt"
            if config_path.exists():
                for line in config_path.read_text(errors="replace").splitlines():
                    if "=" in line:
                        key, value = line.split("=", 1)
                        config[key] = value

            perf_path = workload_dir / "perf.csv"
            samples = {}
            invalid_rows = 0
            if perf_path.exists():
                for line_number, raw_line in enumerate(
                    perf_path.read_text(errors="replace").splitlines(), 1
                ):
                    parts = [part.strip() for part in raw_line.split(",")]
                    if len(parts) < 4:
                        if raw_line.strip():
                            invalid_rows += 1
                        continue
                    timestamp = parts[0]
                    value = number(parts[1])
                    event = parts[3]
                    runtime_ns = number(parts[4]) if len(parts) > 4 else None
                    running_pct = number(parts[5]) if len(parts) > 5 else None
                    if value is None or event not in {"page-faults", "cache-misses"}:
                        invalid_rows += 1
                        continue
                    sample = samples.setdefault(
                        timestamp,
                        {
                            "mode": mode,
                            "repeat": repeat,
                            "workload": workload,
                            "process_name": workload_name,
                            "pid": pid,
                            "timestamp_sec": float(timestamp),
                            "page_faults": "",
                            "cache_misses": "",
                            "runtime_ns": runtime_ns if runtime_ns is not None else "",
                            "running_pct": running_pct if running_pct is not None else "",
                            "source_file": str(perf_path.relative_to(root)),
                        },
                    )
                    if event == "page-faults":
                        sample["page_faults"] = value
                    else:
                        sample["cache_misses"] = value
                for sample_index, sample in enumerate(
                    sorted(samples.values(), key=lambda row: row["timestamp_sec"]), 1
                ):
                    sample["sample_index"] = sample_index
                    perf_rows.append(sample)

            perf_map.append(
                {
                    "mode": mode,
                    "repeat": repeat,
                    "workload": workload,
                    "process_name": workload_name,
                    "pid": pid if pid is not None else "",
                    "container": config.get("container", ""),
                    "memory_limit": config.get("memory_limit", ""),
                    "mps_percentage": config.get(
                        "mps_active_thread_percentage", ""
                    ),
                    "perf_valid_timestamps": len(samples),
                    "perf_invalid_rows": invalid_rows,
                    "perf_file": str(perf_path.relative_to(root)),
                }
            )

        pidstat_path = run_dir / "system_pidstat.txt"
        if pidstat_path.exists():
            for raw_line in pidstat_path.read_text(errors="replace").splitlines():
                parts = raw_line.split()
                if len(parts) != 16 or parts[-1] not in WORKLOAD_LABELS:
                    continue
                process_name = parts[-1]
                workload = WORKLOAD_LABELS[process_name]
                tgid = int(parts[2])
                expected_pid = expected_pids.get((mode, repeat, process_name))
                workload_cpu_mem.append(
                    {
                        "mode": mode,
                        "repeat": repeat,
                        "timestamp": parts[0],
                        "workload": workload,
                        "process_name": process_name,
                        "pid": tgid,
                        "expected_pid": expected_pid if expected_pid is not None else "",
                        "pid_match": tgid == expected_pid,
                        "cpu_user_pct": float(parts[4]),
                        "cpu_system_pct": float(parts[5]),
                        "cpu_wait_pct": float(parts[7]),
                        "cpu_total_pct": float(parts[8]),
                        "cpu_core": int(parts[9]),
                        "minor_faults_per_sec": float(parts[10]),
                        "major_faults_per_sec": float(parts[11]),
                        "vsz_kb": int(parts[12]),
                        "rss_kb": int(parts[13]),
                        "memory_pct": float(parts[14]),
                    }
                )

        mpstat_path = run_dir / "system_mpstat.txt"
        if mpstat_path.exists():
            for raw_line in mpstat_path.read_text(errors="replace").splitlines():
                parts = raw_line.split()
                if len(parts) != 12 or parts[1] != "all":
                    continue
                system_cpu.append(
                    {
                        "mode": mode,
                        "repeat": repeat,
                        "timestamp": parts[0],
                        "user_pct": float(parts[2]),
                        "nice_pct": float(parts[3]),
                        "system_pct": float(parts[4]),
                        "iowait_pct": float(parts[5]),
                        "irq_pct": float(parts[6]),
                        "softirq_pct": float(parts[7]),
                        "steal_pct": float(parts[8]),
                        "guest_pct": float(parts[9]),
                        "gnice_pct": float(parts[10]),
                        "idle_pct": float(parts[11]),
                    }
                )

        tegra_path = run_dir / "system_tegrastats.txt"
        if tegra_path.exists():
            for sample_index, raw_line in enumerate(
                tegra_path.read_text(errors="replace").splitlines(), 1
            ):
                match = TEGRА_RE.search(raw_line)
                if not match:
                    continue
                tegrastats.append(
                    {
                        "mode": mode,
                        "repeat": repeat,
                        "sample_index": sample_index,
                        "date": match.group("date"),
                        "time": match.group("time"),
                        "ram_used_mb": int(match.group("ram_used")),
                        "ram_total_mb": int(match.group("ram_total")),
                        "swap_used_mb": int(match.group("swap_used") or 0),
                        "swap_total_mb": int(match.group("swap_total") or 0),
                        "gpu_usage_pct": int(match.group("gpu_usage")),
                        "gpu_temp_c": float(match.group("gpu_temp")),
                        "vdd_in_now_mw": int(match.group("power_now")),
                        "vdd_in_avg_mw": int(match.group("power_avg")),
                        "cpu_raw": match.group("cpu"),
                    }
                )

        top_path = run_dir / "system_top.txt"
        if top_path.exists():
            sample_index = 0
            pending = None
            for raw_line in top_path.read_text(errors="replace").splitlines():
                if raw_line.startswith("top - "):
                    sample_index += 1
                    timestamp_match = re.search(r"top -\s+(\d{2}:\d{2}:\d{2})", raw_line)
                    pending = {
                        "mode": mode,
                        "repeat": repeat,
                        "sample_index": sample_index,
                        "timestamp": timestamp_match.group(1) if timestamp_match else "",
                    }
                    continue
                if pending is None:
                    continue
                mem_match = re.match(
                    r"\s*(?:MiB|GiB|KiB) Mem\s*:\s*"
                    r"([\d.]+) total,\s*([\d.]+) free,\s*"
                    r"([\d.]+) used,\s*([\d.]+) buff/cache",
                    raw_line,
                )
                if mem_match:
                    unit = raw_line.lstrip().split()[0]
                    scale = {"KiB": 1 / 1024, "MiB": 1, "GiB": 1024}[unit]
                    pending.update(
                        {
                            "mem_total_mb": float(mem_match.group(1)) * scale,
                            "mem_free_mb": float(mem_match.group(2)) * scale,
                            "mem_used_mb": float(mem_match.group(3)) * scale,
                            "buff_cache_mb": float(mem_match.group(4)) * scale,
                        }
                    )
                    continue
                swap_match = re.match(
                    r"\s*(?:MiB|GiB|KiB) Swap\s*:\s*"
                    r"([\d.]+) total,\s*([\d.]+) free,\s*"
                    r"([\d.]+) used\.\s*([\d.]+) avail Mem",
                    raw_line,
                )
                if swap_match:
                    unit = raw_line.lstrip().split()[0]
                    scale = {"KiB": 1 / 1024, "MiB": 1, "GiB": 1024}[unit]
                    pending.update(
                        {
                            "swap_total_mb": float(swap_match.group(1)) * scale,
                            "swap_free_mb": float(swap_match.group(2)) * scale,
                            "swap_used_mb": float(swap_match.group(3)) * scale,
                            "avail_mem_mb": float(swap_match.group(4)) * scale,
                        }
                    )
                    top_memory.append(pending)
                    pending = None

    workload_cpu_mem = sorted_rows(
        workload_cpu_mem, ("timestamp", "workload")
    )
    system_cpu = sorted_rows(system_cpu, ("timestamp",))
    tegrastats = sorted_rows(tegrastats, ("sample_index",))
    top_memory = sorted_rows(top_memory, ("sample_index",))
    perf_rows = sorted_rows(perf_rows, ("sample_index", "workload"))
    perf_map = sorted_rows(perf_map, ("workload",))

    write_csv(
        output_dir / "workload_cpu_memory_raw.csv",
        [
            "mode", "repeat", "timestamp", "workload", "process_name", "pid",
            "expected_pid", "pid_match", "cpu_user_pct", "cpu_system_pct",
            "cpu_wait_pct", "cpu_total_pct", "cpu_core",
            "minor_faults_per_sec", "major_faults_per_sec", "vsz_kb", "rss_kb",
            "memory_pct",
        ],
        workload_cpu_mem,
    )

    cpu_mem_wide_groups = defaultdict(dict)
    for row in workload_cpu_mem:
        key = (row["mode"], row["repeat"], row["timestamp"])
        cpu_mem_wide_groups[key][row["workload"]] = row
    cpu_mem_wide = []
    for (mode, repeat, timestamp), group in cpu_mem_wide_groups.items():
        row = {"mode": mode, "repeat": repeat, "timestamp": timestamp}
        for workload in ["CLS", "DET", "EST", "SEG", "OBB"]:
            source = group.get(workload, {})
            row[f"{workload}_cpu_pct"] = source.get("cpu_total_pct", "")
            row[f"{workload}_rss_kb"] = source.get("rss_kb", "")
            row[f"{workload}_memory_pct"] = source.get("memory_pct", "")
        cpu_mem_wide.append(row)
    cpu_mem_wide = sorted_rows(cpu_mem_wide, ("timestamp",))
    write_csv(
        output_dir / "workload_cpu_memory_wide.csv",
        ["mode", "repeat", "timestamp"]
        + [
            f"{workload}_{metric}"
            for workload in ["CLS", "DET", "EST", "SEG", "OBB"]
            for metric in ["cpu_pct", "rss_kb", "memory_pct"]
        ],
        cpu_mem_wide,
    )

    write_csv(
        output_dir / "system_cpu_raw.csv",
        [
            "mode", "repeat", "timestamp", "user_pct", "nice_pct",
            "system_pct", "iowait_pct", "irq_pct", "softirq_pct", "steal_pct",
            "guest_pct", "gnice_pct", "idle_pct",
        ],
        system_cpu,
    )
    write_csv(
        output_dir / "tegrastats_ram_gpu_raw.csv",
        [
            "mode", "repeat", "sample_index", "date", "time", "ram_used_mb",
            "ram_total_mb", "swap_used_mb", "swap_total_mb", "gpu_usage_pct",
            "gpu_temp_c", "vdd_in_now_mw", "vdd_in_avg_mw", "cpu_raw",
        ],
        tegrastats,
    )
    write_csv(
        output_dir / "top_memory_raw.csv",
        [
            "mode", "repeat", "sample_index", "timestamp", "mem_total_mb",
            "mem_free_mb", "mem_used_mb", "buff_cache_mb", "swap_total_mb",
            "swap_free_mb", "swap_used_mb", "avail_mem_mb",
        ],
        top_memory,
    )
    write_csv(
        output_dir / "perf_process_workload_map.csv",
        [
            "mode", "repeat", "workload", "process_name", "pid", "container",
            "memory_limit", "mps_percentage", "perf_valid_timestamps",
            "perf_invalid_rows", "perf_file",
        ],
        perf_map,
    )
    write_csv(
        output_dir / "perf_by_workload_raw.csv",
        [
            "mode", "repeat", "workload", "process_name", "pid",
            "sample_index", "timestamp_sec", "page_faults", "cache_misses",
            "runtime_ns", "running_pct", "source_file",
        ],
        perf_rows,
    )

    perf_wide_groups = defaultdict(dict)
    for row in perf_rows:
        key = (row["mode"], row["repeat"], row["sample_index"])
        perf_wide_groups[key][row["workload"]] = row
    perf_wide = []
    for (mode, repeat, sample_index), group in perf_wide_groups.items():
        row = {"mode": mode, "repeat": repeat, "sample_index": sample_index}
        for workload in ["CLS", "DET", "EST", "SEG", "OBB"]:
            source = group.get(workload, {})
            row[f"{workload}_timestamp_sec"] = source.get("timestamp_sec", "")
            row[f"{workload}_page_faults"] = source.get("page_faults", "")
            row[f"{workload}_cache_misses"] = source.get("cache_misses", "")
        perf_wide.append(row)
    perf_wide = sorted_rows(perf_wide, ("sample_index",))
    write_csv(
        output_dir / "perf_by_workload_wide.csv",
        ["mode", "repeat", "sample_index"]
        + [
            f"{workload}_{metric}"
            for workload in ["CLS", "DET", "EST", "SEG", "OBB"]
            for metric in ["timestamp_sec", "page_faults", "cache_misses"]
        ],
        perf_wide,
    )

    pid_mismatches = sum(not row["pid_match"] for row in workload_cpu_mem)
    perf_missing = sum(row["perf_valid_timestamps"] == 0 for row in perf_map)
    validation = {
        "run_directories": len(run_dirs),
        "workload_process_mappings": len(perf_map),
        "workload_cpu_memory_rows": len(workload_cpu_mem),
        "pidstat_pid_mismatches": pid_mismatches,
        "system_cpu_rows": len(system_cpu),
        "tegrastats_rows": len(tegrastats),
        "top_memory_rows": len(top_memory),
        "perf_rows": len(perf_rows),
        "perf_mappings_without_valid_samples": perf_missing,
    }
    with (output_dir / "validation.txt").open("w") as stream:
        for key, value in validation.items():
            stream.write(f"{key}={value}\n")
    for key, value in validation.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
