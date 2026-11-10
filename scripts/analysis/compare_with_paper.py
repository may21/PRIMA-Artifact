#!/usr/bin/env python3
"""Compare one or more nodes' experiment results with the revised paper.

Whole artifact roots, side by side (result folders are found automatically:
for each experiment the newest numbered folder that contains data, and for
the concurrent experiment the newest folder per mode):

  python3 scripts/analysis/compare_with_paper.py \
    --set Prima=/home/orin/Prima/prima_artifacts \
    --set prima_test=/home/orin/prima_test/prima_artifacts \
    --output-dir paper_comparison

Or explicit result roots for a single set (every flag optional):

  python3 scripts/analysis/compare_with_paper.py \
    --isolated $R/isolated_no_memlimit_100_1 --concurrent $R/concurrent_4mode_1 \
    --sweep $R/isolated_memory_sweep_100_1 --mps50 $R/mps50_partitioning_1 \
    --goodput $PRIMA_ARTIFACT_ROOT/goodput/results/goodput_concurrent_5000x1_15W_1 \
    --overhead results/prima_overhead --clip $PRIMA_ARTIFACT_ROOT/clip

The script reads the raw run files (output.txt, perf.csv, time.csv,
system_tegrastats.txt, inference_latency.csv), not the organized CSVs, so
results organized with older parsers are still usable.

Two kinds of checks are written to <output-dir>/paper_comparison.{md,csv}:
  * value checks: measured vs. the number printed in the paper, within the
    tolerance in configs/paper_reference/revised_manuscript.json;
  * claim checks: the qualitative statements the paper makes (for example
    "PRIMA-TS lowers the mean latency of every workload"). These are the
    ones that matter for the revision; absolute numbers can differ between
    boards, thermal state and workload-script versions.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "runners" / "yolo"))
from log_formats import parse_inference_line  # noqa: E402

WORKLOADS = ["CLS", "DET", "EST", "SEG", "OBB"]
LABEL = {"classify": "CLS", "cls": "CLS", "detect": "DET", "det": "DET", "pose": "EST", "est": "EST",
         "segment": "SEG", "seg": "SEG", "obb": "OBB"}
MODE_NAME = {"ts_unlimited": "TS", "ts_memlimit": "PRIMA-TS", "mps20_unlimited": "MPS", "mps20_memlimit": "PRIMA-MPS",
             "mps50_unlimited": "MPS-4", "mps50_memlimit": "PRIMA-4", "ts_equal": "Equal-TS", "ts_oracle": "Oracle-TS"}
GR3D_RE = re.compile(r"GR3D_FREQ\s+(\d+)%")
RAM_RE = re.compile(r"RAM\s+(\d+)/(\d+)MB")


# ----------------------------------------------------------------- helpers
def workload_of(path: Path) -> str | None:
    for part in reversed(path.parts):
        low = part.lower()
        if low in LABEL:
            return LABEL[low]
        for token in re.split(r"[^a-z0-9]+", low):
            if token in LABEL:
                return LABEL[token]
    return None


def mean(v):
    return statistics.fmean(v) if v else None


def sd(v):
    return statistics.pstdev(v) if len(v) > 1 else None


def pct(values, q):
    if not values:
        return None
    s = sorted(values)
    i = (len(s) - 1) * q
    lo, hi = int(i), min(int(i) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (i - lo)


def inference_ms(output: Path) -> list[float]:
    if not output.exists():
        return []
    values = []
    for line in output.read_text(errors="replace").splitlines():
        parsed = parse_inference_line(line)
        if parsed:
            values.append(parsed[2])
    return values


def page_faults_per_sec(perf: Path) -> list[float]:
    """perf stat -I 1000 -x, : one page-faults value per 1 s interval."""
    if not perf.exists():
        return []
    out = []
    for line in perf.read_text(errors="replace").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4 and parts[3] == "page-faults":
            try:
                out.append(float(parts[1]))
            except ValueError:
                pass
    return out


def tegrastats(path: Path):
    gpu, ram = [], []
    if path.exists():
        for line in path.read_text(errors="replace").splitlines():
            g = GR3D_RE.search(line)
            r = RAM_RE.search(line)
            if g:
                gpu.append(float(g.group(1)))
            if r:
                ram.append(float(r.group(1)))
    return gpu, ram


def batch_seconds(time_csv: Path):
    if not time_csv.exists():
        return None
    events = {}
    for row in csv.DictReader(time_csv.open()):
        events[row["event"]] = int(row["epoch_ns"])
    if "start" in events and "end" in events:
        return (events["end"] - events["start"]) / 1e9
    return None


# ----------------------------------------------------------------- loaders
def load_concurrent(root: Path):
    """{mode: {"lat": {wl: [...]}, "pf": {wl: [...]}, "gpu": [...], "ram": [...], "batch": [...]}}"""
    data = defaultdict(lambda: {"lat": defaultdict(list), "sd_runs": defaultdict(list), "pf": defaultdict(list),
                                "gpu": [], "ram": [], "batch": []})
    for run_dir in sorted(root.glob("*/run*")):
        if not run_dir.is_dir():
            continue
        mode = run_dir.parent.name
        d = data[mode]
        for wl_dir in run_dir.iterdir():
            wl = LABEL.get(wl_dir.name)
            if not wl_dir.is_dir() or wl is None:
                continue
            values = inference_ms(wl_dir / "output.txt")
            d["lat"][wl].extend(values)
            if len(values) > 1:
                d["sd_runs"][wl].append(statistics.pstdev(values))
            d["pf"][wl].extend(page_faults_per_sec(wl_dir / "perf.csv"))
        gpu, ram = tegrastats(run_dir / "system_tegrastats.txt")
        d["gpu"].extend(gpu)
        d["ram"].extend(ram)
        b = batch_seconds(run_dir / "time.csv")
        if b:
            d["batch"].append(b)
    return data


def load_isolated(root: Path):
    lat = defaultdict(list)
    for output in root.rglob("output.txt"):
        wl = workload_of(output.parent.relative_to(root))
        if wl:
            lat[wl].extend(inference_ms(output))
    return lat


def load_sweep(root: Path):
    """{(wl, '1.0'): {"lat": [...], "completed": n, "attempts": n, "pf": [...]}}"""
    image_limit = 100
    cfg = root / "experiment_config.txt"
    if cfg.exists():
        for line in cfg.read_text(errors="replace").splitlines():
            if line.startswith("image_limit="):
                try:
                    image_limit = int(line.split("=", 1)[1])
                except ValueError:
                    pass
    out = defaultdict(lambda: {"lat": [], "completed": 0, "attempts": 0, "pf": []})
    summary = root / "summary.csv"
    rows = list(csv.DictReader(summary.open())) if summary.exists() else []
    if not rows:  # fall back to the directory layout <workload>/<label>/run<n>
        rows = [{"workload": p.parts[-3], "memory_label": p.parts[-2], "result_dir": str(p)}
                for p in root.glob("*/*/run*") if p.is_dir()]
    for row in rows:
        wl = LABEL.get(row["workload"].lower(), row["workload"].upper())
        gb = f"{float(row['memory_label'].upper().rstrip('GB')):.1f}"
        # result_dir is absolute on the node that ran it; results copied elsewhere
        # are found through the <workload>/<label>/run<n> layout instead.
        run_dir = Path(row["result_dir"])
        if not run_dir.exists():
            run_dir = root / row["workload"] / row["memory_label"] / Path(row["result_dir"]).name
        values = inference_ms(run_dir / "output.txt")
        cell = out[(wl, gb)]
        cell["lat"].extend(values)
        cell["completed"] += min(len(values), image_limit)
        cell["attempts"] += image_limit
        cell["pf"].extend(page_faults_per_sec(run_dir / "perf.csv"))
    return out


def goodput_mode(name: str) -> str:
    low = name.lower()
    prima = "mem" in low or "prima" in low or "limit" in low
    if "mps" in low:
        return "PRIMA-MPS" if prima else "MPS"
    return "PRIMA-TS" if prima else "TS"


def load_goodput(root: Path):
    lat = defaultdict(lambda: defaultdict(list))
    organized = root / "organized" / "inference_times_raw.csv"
    if organized.exists():
        for row in csv.DictReader(organized.open()):
            mode = "TS" if row["mode"] == "concurrent" else goodput_mode(row["mode"])
            lat[mode][row["workload"]].append(float(row["inference_ms"]))
        return lat
    for path in root.rglob("inference_latency.csv"):
        rel = path.relative_to(root)
        wl = workload_of(rel.parent)
        mode = "TS" if len(rel.parts) <= 2 else goodput_mode(rel.parts[0])
        for row in csv.DictReader(path.open()):
            try:
                lat[mode][wl].append(float(row["inference_ms"]))
            except (KeyError, ValueError):
                pass
    return lat


# ----------------------------------------------------------------- report
class Report:
    def __init__(self, tol):
        self.tol = tol
        self.rows = []

    def value(self, section, item, paper, measured, tol_key, workload="", mode="", absolute=False, note=""):
        if measured is None:
            status, diff = "MISSING", None
        else:
            if absolute:
                diff = measured - paper
                ok = abs(diff) <= self.tol[tol_key]
            else:
                diff = (measured - paper) / paper * 100 if paper else None
                ok = diff is not None and abs(diff) <= self.tol[tol_key] * 100
            status = "OK" if ok else "DIFF"
        self.rows.append({"kind": "value", "section": section, "item": item, "workload": workload, "mode": mode,
                          "paper": paper, "measured": None if measured is None else round(measured, 2),
                          "diff": None if diff is None else round(diff, 1),
                          "diff_unit": "abs" if absolute else "%", "status": status, "note": note})

    def claim(self, section, statement, holds, evidence):
        status = "MISSING" if holds is None else ("HOLDS" if holds else "FAILS")
        self.rows.append({"kind": "claim", "section": section, "item": statement, "workload": "", "mode": "",
                          "paper": "", "measured": "", "diff": "", "diff_unit": "", "status": status, "note": evidence})

    def write(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "paper_comparison.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.rows[0]) if self.rows else ["kind"])
            w.writeheader()
            w.writerows(self.rows)
        lines = ["# Comparison with the revised paper", ""]
        claims = [r for r in self.rows if r["kind"] == "claim"]
        values = [r for r in self.rows if r["kind"] == "value"]
        counts = defaultdict(int)
        for r in self.rows:
            counts[r["status"]] += 1
        lines.append("Summary: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
        lines += ["", "## Claim checks", "", "| Section | Claim | Status | Evidence |", "|---|---|---|---|"]
        lines += [f"| {r['section']} | {r['item']} | **{r['status']}** | {r['note']} |" for r in claims]
        lines += ["", "## Value checks", "",
                  "| Section | Item | WL | Mode | Paper | Measured | Diff | Status |", "|---|---|---|---|---|---|---|---|"]
        for r in values:
            diff = "" if r["diff"] is None else f"{r['diff']:+}{'%' if r['diff_unit'] == '%' else ''}"
            measured = "" if r["measured"] is None else r["measured"]
            lines.append(f"| {r['section']} | {r['item']} | {r['workload']} | {r['mode']} | {r['paper']} | {measured} | {diff} | {r['status']} |")
        (out_dir / "paper_comparison.md").write_text("\n".join(lines) + "\n")
        return "\n".join(lines)


def fmt(v, d=1):
    return "n/a" if v is None else f"{v:.{d}f}"


def compare_all_lower(report, section, statement, base, test, unit="ms", labels=("baseline", "PRIMA")):
    """Claim holds when test < base for every workload with data."""
    pairs = [(wl, base.get(wl), test.get(wl)) for wl in WORKLOADS if base.get(wl) is not None and test.get(wl) is not None]
    if not pairs:
        report.claim(section, statement, None, "no data")
        return
    holds = all(t < b for _, b, t in pairs)
    evidence = f"{labels[0]} vs {labels[1]} ({unit}): " + "; ".join(f"{wl} {fmt(b)} vs {fmt(t)}" for wl, b, t in pairs)
    report.claim(section, statement, holds, evidence)


# ----------------------------------------------------------------- main
def evaluate(ref, paths) -> "Report":
    """Run every value and claim check for one set of result roots."""
    rep = Report(ref["tolerance"])

    iso = load_isolated(paths.isolated) if paths.isolated else {}
    iso_mean = {wl: mean(v) for wl, v in iso.items()}
    if paths.isolated:
        for wl, paper in ref["fig2_motivation"]["isolated_mean_ms"].items():
            rep.value("Fig 2a", "isolated mean latency (ms)", paper, iso_mean.get(wl), "latency_rel", wl, "Isolated")

    conc, paths.concurrent_sources = load_concurrent_many(paths.concurrent) if paths.concurrent else ({}, {})
    if conc:
        m = {mode: {wl: mean(v) for wl, v in d["lat"].items()} for mode, d in conc.items()}
        s = {mode: {wl: mean(v) for wl, v in d["sd_runs"].items()} for mode, d in conc.items()}
        pf = {mode: {wl: mean(v) for wl, v in d["pf"].items()} for mode, d in conc.items()}
        f2, f51 = ref["fig2_motivation"], ref["sec51_concurrent"]
        for wl, p in f2["ts_mean_ms"].items():
            rep.value("Fig 2a", "TS mean latency (ms)", p, m.get("ts_unlimited", {}).get(wl), "latency_rel", wl, "TS")
        for wl, p in f2["ts_sd_ms"].items():
            rep.value("Fig 2a", "TS latency SD (ms)", p, s.get("ts_unlimited", {}).get(wl), "latency_rel", wl, "TS",
                      note="SD is very sensitive to rare spikes")
        for wl, p in f2["mps20_mean_ms"].items():
            rep.value("Fig 2a", "MPS-2 mean latency (ms)", p, m.get("mps20_unlimited", {}).get(wl), "latency_rel", wl, "MPS")
        for key, mode in [("ts_mean_ms", "ts_unlimited"), ("prima_ts_mean_ms", "ts_memlimit"),
                          ("mps20_mean_ms", "mps20_unlimited"), ("prima_mps20_mean_ms", "mps20_memlimit")]:
            for wl, p in f51[key].items():
                rep.value("Sec 5.1", "mean latency (ms)", p, m.get(mode, {}).get(wl), "latency_rel", wl, MODE_NAME[mode])
        for key, mode in [("ts_sd_ms", "ts_unlimited"), ("prima_ts_sd_ms", "ts_memlimit"),
                          ("mps20_sd_ms", "mps20_unlimited"), ("prima_mps20_sd_ms", "mps20_memlimit")]:
            for wl, p in f51[key].items():
                rep.value("Sec 5.1", "latency SD (ms)", p, s.get(mode, {}).get(wl), "latency_rel", wl, MODE_NAME[mode])
        for mode, (p, _) in ref["table3_total_time_s"].items():
            rep.value("Table 3", "total time for 100 images (s)", p, mean(conc.get(mode, {}).get("batch", [])) if mode in conc else None,
                      "total_time_rel", mode=MODE_NAME[mode])
        for mode, p in f51["gpu_util_median_pct"].items():
            gpu = conc.get(mode, {}).get("gpu", []) if mode in conc else []
            rep.value("Fig 6d", "median GPU utilization (%)", p, statistics.median(gpu) if gpu else None,
                      "gpu_util_abs_pct", mode=MODE_NAME[mode], absolute=True)
        for gpu_mode, base, prima in [("ts", "ts_unlimited", "ts_memlimit"), ("mps20", "mps20_unlimited", "mps20_memlimit")]:
            b, t = pf.get(base, {}).get("OBB"), pf.get(prima, {}).get("OBB")
            red = (b - t) / b * 100 if b and t is not None else None
            rep.value("Sec 5.1", "OBB page-fault reduction (%)", f51["obb_page_fault_reduction_pct"][gpu_mode], red,
                      "page_fault_rel", "OBB", MODE_NAME[prima])
        ram = conc.get("ts_unlimited", {}).get("ram", []) if "ts_unlimited" in conc else []
        rep.value("Fig 2b", "peak total RAM under TS (GB)", f2["total_ram_used_gb_concurrent"],
                  max(ram) / 1024 if ram else None, "memory_rel", mode="TS")

        # claims
        if iso_mean:
            compare_all_lower(rep, "Sec 3", "Co-location slows every workload (TS slower than Isolated)",
                              {wl: m.get("ts_unlimited", {}).get(wl) for wl in WORKLOADS}, iso_mean, labels=("TS", "Isolated"))
        compare_all_lower(rep, "Sec 3", "MPS-2 is slower than TS for every workload",
                          m.get("mps20_unlimited", {}), m.get("ts_unlimited", {}), labels=("MPS-2", "TS"))
        compare_all_lower(rep, "Sec 5.1", "PRIMA-TS lowers mean latency of every workload", m.get("ts_unlimited", {}), m.get("ts_memlimit", {}), labels=("TS", "PRIMA-TS"))
        compare_all_lower(rep, "Sec 5.1", "PRIMA-MPS lowers mean latency of every workload", m.get("mps20_unlimited", {}), m.get("mps20_memlimit", {}), labels=("MPS", "PRIMA-MPS"))
        compare_all_lower(rep, "Sec 5.1", "PRIMA-TS lowers latency SD of every workload", s.get("ts_unlimited", {}), s.get("ts_memlimit", {}), labels=("TS", "PRIMA-TS"))
        compare_all_lower(rep, "Sec 5.1", "PRIMA-MPS lowers latency SD of every workload", s.get("mps20_unlimited", {}), s.get("mps20_memlimit", {}), labels=("MPS", "PRIMA-MPS"))
        compare_all_lower(rep, "Fig 6c", "PRIMA-TS lowers page faults/s of every workload", pf.get("ts_unlimited", {}), pf.get("ts_memlimit", {}), "PF/s", ("TS", "PRIMA-TS"))
        compare_all_lower(rep, "Fig 6c", "PRIMA-MPS lowers page faults/s of every workload", pf.get("mps20_unlimited", {}), pf.get("mps20_memlimit", {}), "PF/s", ("MPS", "PRIMA-MPS"))
        for base, prima in [("ts_unlimited", "ts_memlimit"), ("mps20_unlimited", "mps20_memlimit")]:
            gb, gp = conc.get(base, {}).get("gpu") if base in conc else None, conc.get(prima, {}).get("gpu") if prima in conc else None
            if gb and gp:
                rep.claim("Fig 6d", f"{MODE_NAME[prima]} raises median GPU utilization", statistics.median(gp) > statistics.median(gb),
                          f"{statistics.median(gb):.0f}% → {statistics.median(gp):.0f}%")
            bb, bp = (mean(conc[base]["batch"]) if base in conc else None), (mean(conc[prima]["batch"]) if prima in conc else None)
            if bb and bp:
                rep.claim("Table 3", f"{MODE_NAME[prima]} shortens total time", bp < bb, f"{bb:.1f} s → {bp:.1f} s ({(bb - bp) / bb * 100:.0f}% less)")

    if paths.sweep:
        sw = load_sweep(paths.sweep)
        for key, paper in ref["table2_success_pct"].items():
            wl, gb = key.split("@")
            cell = sw.get((wl, f"{float(gb):.1f}"))
            rate = cell["completed"] / cell["attempts"] * 100 if cell and cell["attempts"] else None
            rep.value("Table 2", "success rate (%)", paper, rate, "success_abs_pct", wl, f"{gb} GB", absolute=True)
        for key, paper in ref["sweep_latency"].items():
            wl, gb = key.split("@")
            cell = sw.get((wl, f"{float(gb):.1f}"))
            rep.value("Sec 3", "isolated latency by limit (ms)", paper["mean_ms"], mean(cell["lat"]) if cell else None, "latency_rel", wl, f"{gb} GB")
        for key, paper in ref["sweep_page_faults_per_sec"].items():
            wl, gb = key.split("@")
            cell = sw.get((wl, f"{float(gb):.1f}"))
            rep.value("Fig 4", "page faults/s by limit", paper, mean(cell["pf"]) if cell else None, "page_fault_rel", wl, f"{gb} GB")
        obb = {gb: (sw[("OBB", gb)]["completed"] / sw[("OBB", gb)]["attempts"] * 100) for gb in ["1.0", "1.5", "2.0", "2.5"]
               if ("OBB", gb) in sw and sw[("OBB", gb)]["attempts"]}
        if obb:
            holds = all(obb.get(g, 100) < 100 for g in ["1.0", "1.5"]) and all(obb.get(g, 0) == 100 for g in ["2.0", "2.5"])
            rep.claim("Table 2", "OBB fails below 2.0 GB and completes at 2.0 GB and above", holds,
                      "; ".join(f"{g} GB {v:.0f}%" for g, v in sorted(obb.items())))
        flat = []
        for wl in WORKLOADS:
            means = [mean(sw[(wl, g)]["lat"]) for g in ["1.0", "1.5", "2.0", "2.5"] if (wl, g) in sw and sw[(wl, g)]["lat"]]
            if len(means) >= 2:
                flat.append((wl, max(means) / min(means)))
        if flat:
            rep.claim("Sec 3", "Once demand is met, isolated latency is flat across 1.0-2.5 GB (max/min < 1.10)",
                      all(r < 1.10 for _, r in flat), "; ".join(f"{wl} ×{r:.2f}" for wl, r in flat))

    if paths.mps50:
        m50 = load_concurrent(paths.mps50)
        if not iso_mean:
            rep.claim("Sec 5.2", "PRIMA gain grows from 2 to 4 SMs", None, "needs --isolated for normalization")
        else:
            def reductions(data, base, prima):
                out = {}
                for wl in WORKLOADS:
                    b, t = mean(data.get(base, {}).get("lat", {}).get(wl, [])), mean(data.get(prima, {}).get("lat", {}).get(wl, []))
                    if b and t:
                        out[wl] = (b - t) / b * 100
                return out
            r2 = reductions(conc, "mps20_unlimited", "mps20_memlimit") if conc else {}
            r4 = reductions(m50, "mps50_unlimited", "mps50_memlimit")
            for label, r, key in [("MPS-2", r2, "mps20"), ("MPS-4", r4, "mps50")]:
                lo, hi = ref["sec52_mps"]["prima_normalized_reduction_pct"][key]
                if r:
                    rep.claim("Sec 5.2", f"PRIMA lowers {label} latency by {lo}-{hi}% per workload",
                              all(lo * 0.5 <= v <= hi * 1.5 for v in r.values()),
                              "; ".join(f"{wl} {v:.0f}%" for wl, v in r.items()) + " (pass band: half the low end to 1.5× the high end)")
            if r2 and r4:
                rep.claim("Sec 5.2", "PRIMA gain is larger at 4 SMs than at 2 SMs", mean(list(r4.values())) > mean(list(r2.values())),
                          f"mean reduction {mean(list(r2.values())):.0f}% → {mean(list(r4.values())):.0f}%")

    if paths.goodput:
        gp = load_goodput(paths.goodput)
        for mode, table in ref["table7_goodput_ms"].items():
            if mode not in gp:
                continue
            for wl, (p50, p90, p99) in table.items():
                values = [v for w in WORKLOADS for v in gp[mode].get(w, [])] if wl == "Overall" else gp[mode].get(wl, [])
                for q, paper in [(0.5, p50), (0.9, p90), (0.99, p99)]:
                    rep.value("Table 7", f"p{int(q * 100)} latency (ms)", paper, pct(values, q), "latency_rel", wl, mode)
        for base, prima in [("TS", "PRIMA-TS"), ("MPS", "PRIMA-MPS")]:
            if base in gp and prima in gp:
                compare_all_lower(rep, "Table 7", f"{prima} lowers p99 of every workload",
                                  {wl: pct(gp[base].get(wl, []), 0.99) for wl in WORKLOADS},
                                  {wl: pct(gp[prima].get(wl, []), 0.99) for wl in WORKLOADS}, labels=(base, prima))
        if set(gp) == {"TS"}:
            rep.claim("Table 7", "Goodput run includes PRIMA modes", False,
                      "only one mode found; run_goodput_concurrent.sh has no memory limits, so Table 7's PRIMA columns cannot be checked")

    if paths.overhead:
        summary = paths.overhead / "overhead_summary.csv"
        rows = list(csv.DictReader(summary.open())) if summary.exists() else []
        fe = {r["workload"]: float(r["mean_ms"]) for r in rows if r["stage"] == "feature_extraction"}
        rep.value("Fig 10a", "feature extraction (ms)", ref["fig10_overhead_ms"]["feature_extraction"]["CLS"], fe.get("CLS"), "overhead_rel", "CLS")
        yolo = [fe[w] for w in WORKLOADS if w in fe]
        rep.value("Sec 5.5", "five-YOLO feature extraction total (ms)", ref["fig10_overhead_ms"]["five_yolo_feature_extraction_total"],
                  sum(yolo) if len(yolo) == 5 else None, "overhead_rel",
                  note="the paper measured on the i9 control node; a Jetson is expected to be slower")

    if paths.clip:
        avg = paths.clip / "clip_memory_peak_average.csv"
        rows = list(csv.DictReader(avg.open())) if avg.exists() else []
        for model, paper in ref["table8_clip_actual_mb"].items():
            key = model.lower().replace("/", "").replace("-", "")
            row = next((r for r in rows if r["model"].lower().replace("/", "").replace("-", "").replace("_", "") in (key, f"clip{key}")
                        or key in r["model"].lower().replace("-", "").replace("_", "")), None)
            value = float(row["avg_peak_container_memory_mib"]) if row and row.get("avg_peak_container_memory_mib") else None
            rep.value("Table 8", "measured memory (MB, container peak)", paper, value, "memory_rel", model)

    return rep


# ----------------------------------------------------------------- discovery
def run_index(path: Path) -> int:
    match = re.search(r"_(\d+)$", path.name)
    return int(match.group(1)) if match else 0


def has_inference(root: Path, limit: int = 40) -> bool:
    for n, output in enumerate(root.rglob("output.txt")):
        if inference_ms(output):
            return True
        if n >= limit:
            break
    return False


def latest_with_data(candidates, check) -> Path | None:
    for path in sorted(candidates, key=run_index, reverse=True):
        if path.is_dir() and check(path):
            return path
    return None


def load_concurrent_many(roots):
    """Merge concurrent roots: for each mode, the newest root with data wins."""
    if isinstance(roots, Path):
        roots = [roots]
    merged, sources = {}, {}
    for root in sorted(roots, key=run_index):
        for mode, data in load_concurrent(root).items():
            if any(data["lat"].values()):
                merged[mode] = data
                sources[mode] = root
    return merged, sources


def discover(artifact_root: Path) -> SimpleNamespace:
    results = artifact_root / "yolo" / "results"
    paths = SimpleNamespace(isolated=None, concurrent=None, sweep=None, mps50=None,
                            goodput=None, overhead=None, clip=None, concurrent_sources={})
    if results.is_dir():
        paths.isolated = latest_with_data(results.glob("isolated_no_memlimit_*"), has_inference)
        conc = [p for p in results.glob("concurrent_4mode_*") if p.is_dir()]
        paths.concurrent = conc or None
        paths.sweep = latest_with_data(results.glob("isolated_memory_sweep_*"), has_inference)
        paths.mps50 = latest_with_data(results.glob("mps50_partitioning_*"), has_inference)
    goodput = artifact_root / "goodput" / "results"
    if goodput.is_dir():
        paths.goodput = latest_with_data(goodput.glob("goodput_concurrent_*"),
                                         lambda p: any(p.rglob("inference_latency.csv")))
    clip = artifact_root / "clip"
    if (clip / "clip_memory_peak_average.csv").exists():
        paths.clip = clip
    overhead = sorted(artifact_root.glob("**/overhead_summary.csv"), key=lambda p: p.stat().st_mtime)
    if overhead:
        paths.overhead = overhead[-1].parent
    return paths


def describe_inputs(name: str, paths) -> list[str]:
    lines = [f"**{name}**", ""]
    for key in ["isolated", "sweep", "mps50", "goodput", "overhead", "clip"]:
        value = getattr(paths, key, None)
        lines.append(f"- {key}: `{value}`" if value else f"- {key}: not found")
    sources = getattr(paths, "concurrent_sources", {}) or {}
    if sources:
        for mode, root in sorted(sources.items()):
            lines.append(f"- concurrent {mode}: `{root}`")
    else:
        lines.append("- concurrent: not found")
    return lines + [""]


# ----------------------------------------------------------------- combined report
def write_combined(reports: dict, inputs: dict, out_dir: Path) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    names = list(reports)
    keyed = {}
    order = []
    for name, rep in reports.items():
        for row in rep.rows:
            key = (row["kind"], row["section"], row["item"], row["workload"], row["mode"])
            if key not in keyed:
                keyed[key] = {"paper": row["paper"], "sets": {}}
                order.append(key)
            keyed[key]["sets"][name] = row

    with (out_dir / "paper_comparison.csv").open("w", newline="") as f:
        fields = ["set", "kind", "section", "item", "workload", "mode", "paper", "measured", "diff", "diff_unit", "status", "note"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for name, rep in reports.items():
            for row in rep.rows:
                w.writerow({"set": name, **row})

    lines = ["# Comparison with the revised paper", "", "## Inputs", ""]
    for name in names:
        lines += inputs[name]
    lines += ["## Summary", "", "| Set | " + " | ".join(["HOLDS", "FAILS", "OK", "DIFF", "MISSING"]) + " |", "|---" * 6 + "|"]
    for name, rep in reports.items():
        counts = defaultdict(int)
        for r in rep.rows:
            counts[r["status"]] += 1
        lines.append(f"| {name} | " + " | ".join(str(counts[k]) for k in ["HOLDS", "FAILS", "OK", "DIFF", "MISSING"]) + " |")

    lines += ["", "## Claim checks", "", "| Section | Claim | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    claim_keys = [k for k in order if k[0] == "claim"]
    for key in claim_keys:
        cells = [f"**{keyed[key]['sets'][n]['status']}**" if n in keyed[key]["sets"] else "" for n in names]
        lines.append(f"| {key[1]} | {key[2]} | " + " | ".join(cells) + " |")
    lines += ["", "### Claim evidence", ""]
    for key in claim_keys:
        lines.append(f"- {key[1]}: {key[2]}")
        for n in names:
            row = keyed[key]["sets"].get(n)
            if row and row["note"]:
                lines.append(f"  - {n}: {row['note']}")

    lines += ["", "## Value checks", "", "Each cell: measured (difference from paper) status.", "",
              "| Section | Item | WL | Mode | Paper | " + " | ".join(names) + " |", "|---|---|---|---|---|" + "---|" * len(names)]
    for key in [k for k in order if k[0] == "value"]:
        cells = []
        for n in names:
            row = keyed[key]["sets"].get(n)
            if not row or row["measured"] is None:
                cells.append("—")
            else:
                diff = "" if row["diff"] is None else f" ({row['diff']:+}{'%' if row['diff_unit'] == '%' else ''})"
                cells.append(f"{row['measured']}{diff} {row['status']}")
        lines.append(f"| {key[1]} | {key[2]} | {key[3]} | {key[4]} | {keyed[key]['paper']} | " + " | ".join(cells) + " |")
    text = "\n".join(lines) + "\n"
    (out_dir / "paper_comparison.md").write_text(text)
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", action="append", default=[], metavar="NAME=ARTIFACT_ROOT",
                    help="compare a whole artifact root (repeatable); result folders are found automatically")
    for name in ["isolated", "concurrent", "sweep", "mps50", "goodput", "overhead", "clip"]:
        ap.add_argument(f"--{name}", type=Path, help=f"explicit {name} result root (single-set mode)")
    ap.add_argument("--reference", type=Path, default=REPO / "configs" / "paper_reference" / "revised_manuscript.json")
    ap.add_argument("--output-dir", type=Path, default=Path("paper_comparison"))
    args = ap.parse_args()
    ref = json.loads(args.reference.read_text())

    sets = {}
    for spec in args.set:
        name, sep, root = spec.partition("=")
        if not sep:
            root, name = name, Path(name).parent.name or name
        root = Path(root).expanduser()
        if not root.is_dir():
            raise SystemExit(f"artifact root not found: {root}")
        sets[name] = discover(root)
    explicit = {k: getattr(args, k) for k in ["isolated", "concurrent", "sweep", "mps50", "goodput", "overhead", "clip"]}
    if any(explicit.values()):
        sets["run"] = SimpleNamespace(**explicit, concurrent_sources={})
    if not sets:
        ap.error("give --set NAME=ARTIFACT_ROOT (repeatable) or explicit result roots")

    reports, inputs = {}, {}
    for name, paths in sets.items():
        reports[name] = evaluate(ref, paths)
        inputs[name] = describe_inputs(name, paths)
    text = write_combined(reports, inputs, args.output_dir)
    print(text)
    print(f"[written] {args.output_dir / 'paper_comparison.md'} and .csv")


if __name__ == "__main__":
    main()
