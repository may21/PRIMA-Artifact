# PRIMA

## 1. Hardware and software

| Item | Configuration |
|---|---|
| CPU | 6-core ARM Cortex-A78AE |
| GPU | 1024-core NVIDIA Ampere GPU |
| Memory | 8 GB LPDDR5 |
| OS | Ubuntu 22.04 |
| JetPack | 6.2 |
| CUDA | 12.6 |
| Inference engine | TensorRT v10.3.0 |
| Container runtime | Docker with NVIDIA runtime |

Notes:

- Set Jetson Orin Nano to 15 W mode before running experiments.
- Turn off swap for memory-isolation experiments.
- Keep large files outside GitHub: datasets, TensorRT engines, Docker images,
  and raw results.

## 2. Repository layout

```text
configs/                 Device, workload, and experiment configuration files
datasets/                Small predictor feature CSVs
prima/                   PRIMA implementation modules
scripts/master/          SSH wrapper for launching experiments on the edge node
scripts/experiments/     Paper experiment wrappers run on the edge node
scripts/runners/         Experiment runners and result organizers
scripts/setup/           Environment checks and helper scripts
requirements.txt         Python package requirements
```

The main design path is:

```text
Manager -> Predictor -> Calculator -> Workload Launcher -> Metrics Collector
```

| Paper component | Repository mapping |
|---|---|
| Manager | `prima/manager/`, `scripts/master/`, `scripts/experiments/` |
| Predictor | `prima/features/`, `prima/predictor/`, `datasets/predictor_memory_features.csv` |
| Calculator | `prima/calculator/`, `prima/allocator/` |
| Workload Launcher | `prima/workload_launcher/`, `prima/runtime/` |
| Metrics Collector | `prima/metrics_collector/` and edge-node system monitors |
| Experiment runners | `scripts/runners/` |

## 3. Setup

Clone the repository on the edge node:

```bash
git clone https://github.com/woosy123/PRIMA.git
cd PRIMA
python3 -m pip install -r requirements.txt
```

Check the edge-node environment:

```bash
./scripts/setup/check_orin_env.sh
nvpmodel -q
cat /proc/swaps
```

`nvpmodel -q` verifies the paper power mode, and `/proc/swaps` verifies that
swap is disabled before memory-isolation experiments.

If experiments are launched from a separate master node, configure SSH access to
the edge node and run:

```bash
ORIN_HOST=$EDGE_HOST ORIN_PORT=$EDGE_PORT \
  ./scripts/master/run_on_orin1.sh check-env
```

## 4. Experiments

Run the wrappers directly on the edge node:

```bash
./scripts/experiments/run_isolated_baseline.sh
./scripts/experiments/run_concurrent_comparison.sh
./scripts/experiments/run_memory_sweep.sh
./scripts/experiments/run_mps_partitioning.sh
./scripts/experiments/run_goodput_latency.sh
./scripts/experiments/run_clip_generalization.sh
./scripts/experiments/run_prima_overhead.sh
```

| Edge-node command | Runs | Main output metrics |
|---|---|---|
| `./scripts/experiments/run_isolated_baseline.sh` | Runs each YOLO workload alone with no Docker memory limit through `scripts/runners/yolo/run_isolated_baseline.sh`. | Isolated inference latency, elapsed time, pidstat memory, tegrastats RAM/GPU, and per-workload result logs. |
| `./scripts/experiments/run_concurrent_comparison.sh` | Runs five YOLO workloads concurrently through `scripts/runners/yolo/run_concurrent_4mode.sh`. By default it executes TS, MPS 20%, PRIMA-TS, and PRIMA-MPS conditions. | Concurrent inference latency, total elapsed time, cgroup memory, page faults, CPU/GPU utilization, and organized comparison CSVs. |
| `./scripts/experiments/run_memory_sweep.sh` | Runs each YOLO workload under isolated Docker memory limits from 0.5 GB to 2.5 GB through `scripts/runners/yolo/run_isolated_memory_sweep.sh`. | Completion threshold, latency, peak memory, page faults, and GPU utilization by memory budget. |
| `./scripts/experiments/run_mps_partitioning.sh` | Runs MPS 20% and MPS 50% partitioning conditions through the YOLO concurrent runners. | Normalized MPS latency, memory usage, and page-fault behavior for partitioning comparison. |
| `./scripts/experiments/run_goodput_latency.sh` | Runs five concurrent YOLO goodput workloads through `scripts/runners/goodput/run_goodput_concurrent.sh`, then organizes latency CSVs. | Per-request latency for 5,000 requests per workload and p50/p90/p99-ready CSV outputs. |
| `./scripts/experiments/run_clip_generalization.sh` | Runs CLIP ViT-B/32 and RN50 Docker profiling through `scripts/runners/clip/`. | Peak system RAM, peak container memory, elapsed time, and CLIP summary CSVs. |
| `./scripts/experiments/run_prima_overhead.sh` | Runs `scripts/runners/overhead/measure_prima_overhead.py` over the ONNX model list. | Feature extraction, RF prediction, and Calculator overhead. |

Or launch the same wrappers from the master node:

```bash
./scripts/master/run_on_orin1.sh ch3-isolated
./scripts/master/run_on_orin1.sh ch3-concurrent
./scripts/master/run_on_orin1.sh ch3-sweep
./scripts/master/run_on_orin1.sh ch5-concurrent
./scripts/master/run_on_orin1.sh ch5-mps
./scripts/master/run_on_orin1.sh ch5-goodput
./scripts/master/run_on_orin1.sh ch5-clip
./scripts/master/run_on_orin1.sh ch5-overhead
```

| Target | Purpose |
|---|---|
| `ch3-isolated` | Calls `./scripts/experiments/run_isolated_baseline.sh` on Orin1. |
| `ch3-concurrent` | Calls `./scripts/experiments/run_concurrent_comparison.sh` with `MODE_LIST="ts_unlimited mps20_unlimited"`. |
| `ch3-sweep` | Calls `./scripts/experiments/run_memory_sweep.sh` on Orin1. |
| `ch5-concurrent` | Calls `./scripts/experiments/run_concurrent_comparison.sh` with all default concurrent modes. |
| `ch5-mps` | Calls `./scripts/experiments/run_mps_partitioning.sh` on Orin1. |
| `ch5-goodput` | Calls `./scripts/experiments/run_goodput_latency.sh` on Orin1. |
| `ch5-clip` | Calls `./scripts/experiments/run_clip_generalization.sh` on Orin1. |
| `ch5-overhead` | Calls `./scripts/experiments/run_prima_overhead.sh` on Orin1. |

The default Chapter 5 concurrent comparison uses the following memory-budget
conditions:

```text
ts_unlimited
mps20_unlimited
ts_memlimit
mps20_memlimit
```

The paper budget vector used by the YOLO concurrent runners is:

```text
classify 821m
detect   1071m
pose     1068m
segment  1649m
obb      2233m
```

## 5. External assets

Prepare these assets on the edge node before running the full artifact:

| Asset | Notes |
|---|---|
| YOLO workload assets and results | `/home/orin1/prima_artifacts/yolo` |
| YOLO runtime images and TensorRT engines | `/home/orin1/prima_artifacts/yolo_runtime` |
| Goodput datasets | `/mnt/prima_usb/prima_goodput_5000` |
| Goodput state and results | `/home/orin1/prima_artifacts/goodput` |
| CLIP model/data cache and results | `/home/orin1/prima_artifacts/clip` |
| ONNX files for overhead measurement | `/home/orin1/prima_artifacts/onnx_cache` |

These assets are excluded from GitHub because they are large or machine-specific.
Experiment runners and result organizers are included under `scripts/runners/`.
The CLIP runtime was built from `ultralytics/ultralytics:8.3.102-jetson-jetpack6`
by installing [openai/CLIP](https://github.com/openai/CLIP) with pip, excluding
the conda-based PyTorch installation command from the CLIP README. Only the
base Ultralytics image is kept on the edge node by default.

The repository includes only the small predictor feature table:

```text
datasets/predictor_memory_features.csv
```

Optional USB storage for goodput datasets:

```bash
sudo mkdir -p /mnt/prima_usb
sudo mount /dev/sda1 /mnt/prima_usb
sudo chown -R orin1:orin1 /mnt/prima_usb
```

## 6. File locations

GitHub repository:

| Path | Contents |
|---|---|
| `configs/devices/orin_nano_15w.yaml` | Jetson Orin Nano paper environment |
| `configs/experiments/paper_concurrent_4mode_100.yaml` | Concurrent TS/MPS/PRIMA experiment configuration |
| `configs/workloads/yolo_100.yaml` | YOLO workload configuration |
| `configs/workloads/goodput_5000.yaml` | Goodput workload configuration |
| `configs/workloads/overhead_models.csv` | ONNX paths used by the overhead runner |
| `datasets/predictor_memory_features.csv` | Small predictor feature table |
| `prima/` | PRIMA implementation modules |
| `scripts/experiments/` | Edge-node experiment wrappers |
| `scripts/master/run_on_orin1.sh` | Master-node SSH launcher |
| `scripts/runners/yolo/` | YOLO concurrent, memory-sweep, and result-organization runners |
| `scripts/runners/goodput/` | Goodput latency runner and organizer |
| `scripts/runners/clip/` | CLIP Docker runners, inference scripts, and summarizer |
| `scripts/runners/overhead/` | Predictor and Calculator overhead runner |
| `scripts/setup/` | Environment and helper scripts |
| `requirements.txt` | Python dependencies |

Orin1 and USB storage:

| Path | Contents |
|---|---|
| `/home/orin1/woosy/PRIMA` | GitHub clone used by `scripts/master/run_on_orin1.sh` |
| `/home/orin1/prima_artifacts/yolo` | YOLO workload assets, image lists, and result root |
| `/home/orin1/prima_artifacts/yolo/workload/image_lists` | `coco_100.txt`, `dota_100.txt`, `imagenet_100.txt` |
| `/home/orin1/prima_artifacts/yolo/results` | Chapter 3 and concurrent YOLO result directories |
| `/home/orin1/prima_artifacts/yolo_runtime` | YOLO runtime images and TensorRT engines |
| `/home/orin1/prima_artifacts/goodput` | Goodput state and result root |
| `/home/orin1/prima_artifacts/goodput/results` | Goodput result directories |
| `/home/orin1/prima_artifacts/clip` | CLIP model/data cache and result files |
| `/home/orin1/prima_artifacts/onnx_cache` | ONNX files used by the overhead runner |
| `/mnt/prima_usb/prima_goodput_5000` | Goodput image datasets prepared outside GitHub |
| `/home/orin1/ssd/prima_goodput_5000` | Current SSD-backed ImageNet and DOTA goodput images |
| `/mnt/prima_usb/prima_dataset_expansion` | Existing model/profile backup artifacts, not required for goodput dataset preparation |

On the current Orin1 machine, the `prima_artifacts` entries are symbolic links
to existing storage locations so that large data does not need to be moved.
The current ImageNet and DOTA goodput folders are stored on the Orin1 SSD and
bind-mounted into `/mnt/prima_usb/prima_goodput_5000` because the attached USB
device showed very slow write throughput during dataset preparation. The runner
paths remain unchanged.

For the goodput dataset, CLS uses ImageNet images and DET/EST/SEG share COCO
images. OBB uses DOTA val/test original images without crop, tile, or source
image modification. By default, `UNIQUE_IMAGES_OBB=auto` uses all DOTA images
present in `expanded_dota_images`; the OBB runner still records 5,000 measured
requests by cycling through those originals.
