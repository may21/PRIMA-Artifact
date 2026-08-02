#!/usr/bin/env python3
import csv
import sys
from pathlib import Path

LABELS = {"classify": "CLS", "detect": "DET", "pose": "EST", "segment": "SEG", "obb": "OBB"}
ORDER = ["classify", "detect", "pose", "segment", "obb"]


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: organize_goodput.py RESULT_ROOT")
    root = Path(sys.argv[1]).resolve()
    if not root.exists():
        raise SystemExit(f"result root does not exist: {root}")
    out = root / "organized"
    raw_rows = []
    run_rows = []

    mode_runs = [
        (p.name, 1, p / "run1")
        for p in root.iterdir()
        if p.is_dir() and p.name not in {"organized"} and (p / "run1").is_dir()
    ]
    if not mode_runs and all((root / workload).is_dir() for workload in ORDER):
        mode_runs = [("concurrent", 1, root)]
    if not mode_runs:
        raise SystemExit(f"no goodput result layout found under {root}")

    for mode, repeat, run_dir in sorted(mode_runs):
        for workload in ORDER:
            csv_path = run_dir / workload / "inference_latency.csv"
            log_path = run_dir / workload / "run.log"
            elapsed = ""
            if log_path.exists():
                for line in log_path.read_text(errors="replace").splitlines():
                    if line.startswith("elapsed_sec="):
                        elapsed = line.split("=", 1)[1]
            count = 0
            if csv_path.exists():
                with csv_path.open(newline="", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        count += 1
                        raw_rows.append(
                            {
                                "mode": mode,
                                "repeat": repeat,
                                "workload": LABELS[workload],
                                "request_index": int(row["request_index"]),
                                "source_image_index": int(row["source_image_index"]),
                                "request_started_ns": row.get("request_started_ns", ""),
                                "request_finished_ns": row.get("request_finished_ns", ""),
                                "inference_ms": row["inference_ms"],
                            }
                        )
            run_rows.append(
                {
                    "mode": mode,
                    "repeat": repeat,
                    "workload": LABELS[workload],
                    "completed_requests": count,
                    "elapsed_sec": elapsed,
                    "csv": str(csv_path.relative_to(root)),
                }
            )

    raw_rows.sort(key=lambda r: (r["mode"], r["repeat"], r["request_index"], r["workload"]))
    write_csv(
        out / "inference_times_raw.csv",
        ["mode", "repeat", "workload", "request_index", "source_image_index", "request_started_ns", "request_finished_ns", "inference_ms"],
        raw_rows,
    )
    write_csv(out / "run_summary.csv", list(run_rows[0]), run_rows)

    by_key = {}
    starts = {}
    finishes = {}
    for row in raw_rows:
        key = (row["mode"], row["repeat"], row["request_index"])
        by_key.setdefault(key, {})[row["workload"]] = row["inference_ms"]
        if row["request_started_ns"]:
            starts.setdefault(key, []).append(int(row["request_started_ns"]))
        if row["request_finished_ns"]:
            finishes.setdefault(key, []).append(int(row["request_finished_ns"]))

    wide_rows = []
    for key in sorted(by_key):
        mode, repeat, request_index = key
        vals = by_key[key]
        wide_rows.append(
            {
                "mode": mode,
                "repeat": repeat,
                "request_index": request_index,
                "CLS": vals.get("CLS", ""),
                "DET": vals.get("DET", ""),
                "EST": vals.get("EST", ""),
                "SEG": vals.get("SEG", ""),
                "OBB": vals.get("OBB", ""),
                "first_started_ns": min(starts.get(key, []), default=""),
                "last_finished_ns": max(finishes.get(key, []), default=""),
            }
        )
    write_csv(
        out / "inference_times_wide.csv",
        ["mode", "repeat", "request_index", "CLS", "DET", "EST", "SEG", "OBB", "first_started_ns", "last_finished_ns"],
        wide_rows,
    )

    ok = all(row["completed_requests"] == 5000 for row in run_rows)
    (out / "validation.txt").write_text(
        f"raw_rows={len(raw_rows)}\nwide_rows={len(wide_rows)}\nruns={len(run_rows)}\nall_runs_5000={ok}\n",
        encoding="utf-8",
    )
    print(f"raw_rows={len(raw_rows)}")
    print(f"wide_rows={len(wide_rows)}")
    print(f"runs={len(run_rows)}")


if __name__ == "__main__":
    main()
