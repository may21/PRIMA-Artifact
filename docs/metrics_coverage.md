# Metrics coverage

This file maps the paper metrics to the repository runners and generated
artifacts. Raw result directories are intentionally excluded from GitHub.

| Paper item | Metric | Runner | Generated files |
|---|---|---|---|
| Motivating example | Average inference time and standard deviation for Isolated, TS, and MPS | `scripts/experiments/run_isolated_baseline.sh`, `scripts/experiments/run_concurrent_comparison.sh` with `MODE_LIST="ts_unlimited mps20_unlimited"` | `output.txt`, `time.csv`, `summary.csv`, `organized/*inference*` |
| Motivating example | Total system memory usage | Same as above | `system_tegrastats.txt`, `organized/tegrastats_ram_gpu_raw.csv`, isolated `organized/*pidstat_memory*` |
| Motivating example | Per-workload memory usage | Same as above | `cgroup_memory.csv`, `system_pidstat.txt`, `organized/workload_cpu_memory_raw.csv`, `organized/workload_cpu_memory_wide.csv` |
| Memory sweep | Completion threshold, inference time, peak memory, page faults, GPU utilization by memory limit | `scripts/experiments/run_memory_sweep.sh` | `organized/run_metrics.csv`, `organized/condition_summary.csv`, `organized/inference_raw.csv`, `organized/cpu_perf_intervals_raw.csv`, `organized/minimum_memory_thresholds.csv` |
| Predictor comparison | MAE, RMSE, R2, cross-validation MAE | `scripts/setup/train_predictor_rf.py` | stdout JSON and optional `--metadata-output` JSON |
| PRIMA concurrent comparison | Average inference time, peak memory usage, page faults, GPU utilization, total inference time for 100 images | `scripts/experiments/run_concurrent_comparison.sh` | `organized/perf_by_workload_raw.csv`, `organized/perf_by_workload_wide.csv`, `organized/tegrastats_ram_gpu_raw.csv`, `organized/workload_cpu_memory_*`, `summary.csv`, `time.csv` |
| MPS partitioning | Normalized inference time, peak memory usage, page-fault rate for 20% and 50% MPS conditions | `scripts/experiments/run_mps_partitioning.sh` | Same organized concurrent files, with `mps20_*` and `mps50_*` modes |
| Goodput and latency percentiles | Per-request inference latency for 5,000 requests per workload, p50/p90/p99 derivable from raw latency CSVs | `scripts/experiments/run_goodput_latency.sh` | `inference_latency.csv`, `organized/inference_times_raw.csv`, `organized/inference_times_wide.csv`, `organized/run_summary.csv`, `organized/validation.txt` |
| CLIP generalization | Peak system RAM, peak container memory, elapsed time | `scripts/experiments/run_clip_generalization.sh` | `clip_memory_peak_raw.csv`, `clip_memory_peak_average.csv`, per-run `system_tegrastats.txt`, `docker_stats.txt`, `summary.txt` |
| PRIMA overhead | ONNX feature extraction, RF prediction, and Calculator elapsed time | `scripts/experiments/run_prima_overhead.sh` | `overhead_raw.csv`, `overhead_summary.csv` |

Known requirements before a full rerun:

- Run `sudo -v` on Orin1 before runners that use `perf` or cgroup controls.
- Keep swap disabled for the reported memory-isolation and goodput experiments.
- Prepare `/home/orin1/ssd/prima_goodput_5000` before running goodput. OBB defaults to `UNIQUE_IMAGES_OBB=auto`, which uses all DOTA val/test originals while still recording 5,000 measured requests.
- Keep TensorRT engines, image lists, and datasets under `/home/orin1/prima_artifacts`; they are not committed.
- Keep ONNX graph files for Predictor feature extraction available at the paths in `configs/workloads/overhead_models.csv`, or pass a replacement `MODEL_CSV`.
