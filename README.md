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

### 3.1 Directory layout

Code, artifacts and large datasets live side by side under one work root.
Every runner derives its paths from these variables, so the same commands
work on any Jetson:

```text
$PRIMA_WORK_ROOT/
├── PRIMA-Artifact/          this repository
├── prima_artifacts/         $PRIMA_ARTIFACT_ROOT
│   ├── yolo/workload/       workload scripts, .pt/.onnx/.engine, image_lists/, 100-image inputs
│   ├── yolo/results/        YOLO experiment results
│   ├── paper_images/        optional: the paper's expanded_* image folders
│   ├── yolo_runtime/        optional: 100-image inputs, if kept apart from workload/
│   ├── goodput/results/     goodput results
│   ├── clip/                CLIP weights, CIFAR-100, results
│   └── onnx_cache/          ONNX graphs for the overhead experiment
└── ssd/                     $PRIMA_SSD_ROOT
    └── prima_goodput_5000/  5,000-image ImageNet/COCO/DOTA sets for goodput
```

```bash
export PRIMA_WORK_ROOT=$HOME/prima_test        # any writable location
export PRIMA_ARTIFACT_ROOT=$PRIMA_WORK_ROOT/prima_artifacts
export PRIMA_SSD_ROOT=$PRIMA_WORK_ROOT/ssd
```

`PRIMA_WORK_ROOT` alone is enough: the other two default to the layout above.
Without any of them, the original orin1 paths are used.
The work root must be on a filesystem that supports Unix ownership: exFAT/FAT
mounts are owned by root and cannot be written by the experiment user.

Path overrides, when a machine deviates from the layout:

| Variable | Default | Used by |
|---|---|---|
| `DATASET_ROOT` | the first of `prima_artifacts/paper_images`, `prima_artifacts/yolo_runtime`, `prima_artifacts/yolo/workload` that holds the images named in the installed image lists; runners stop if a listed image is missing | YOLO runners |
| `YOLO_ROOT` | the first of `yolo_runtime`, `yolo/workload` that holds `yolov8n.engine` | goodput |
| `GOODPUT_DATASET_ROOT` | `$PRIMA_SSD_ROOT/prima_goodput_5000` | goodput |
| `CLIP_ROOT` | `$PRIMA_ARTIFACT_ROOT/clip` | CLIP |
| `PERF_BIN` | the first `perf` that runs on this kernel | perf-based runners |

### 3.2 One-time preparation

```bash
git clone https://github.com/may21/PRIMA-Artifact.git "$PRIMA_WORK_ROOT/PRIMA-Artifact"
cd "$PRIMA_WORK_ROOT/PRIMA-Artifact"
python3 -m pip install -r requirements.txt        # a virtualenv is recommended

sudo nvpmodel -m 0                                # 15 W mode on Orin Nano (check with nvpmodel -q)
sudo swapoff -a
sudo apt install time linux-nvidia-tegra-tools-5.15.0-1030   # /usr/bin/time and a working perf; match your kernel ABI

docker pull ultralytics/ultralytics:8.3.102-jetson-jetpack6
./scripts/setup/prepare_goodput_datasets.sh       # 5,000-image sets in $PRIMA_SSD_ROOT
./scripts/setup/export_yolo_engines.sh            # .onnx/.engine in yolo/workload, ONNX copy in onnx_cache
./scripts/setup/build_clip_image.sh               # clip-full:jetson, CIFAR-100, CLIP weights
```

The paper's five workload scripts and image lists are in `workloads/yolo/`.
Install them, and copy the paper's image folders (`expanded_imagenet_images`,
`expanded_coco_images`, `expanded_dota_images` from orin1's `yolo_new/`; the
DOTA images must be the full-size originals) into the dataset root before
preparing the inputs:

```bash
./scripts/setup/install_workloads.sh
./scripts/setup/prepare_yolo_inputs.sh            # checks every listed image is present
```

[docs/workload_interface.md](docs/workload_interface.md) explains why these
scripts and images matter: the memory pressure the paper measures comes from
them.

**Clock limits.** `nvpmodel -q` can report a power mode whose limits are not in effect:
on some JetPack 6.2 boards the nvpmodel boot service fails with "Reboot required for
changing to this power mode" at every boot, and the CPU, GPU and memory clocks stay
at the MAXN SUPER limits. Apply the mode's limits directly after each boot; every YOLO
and goodput runner checks them, records them in `experiment_config.txt`, and stops if
they are above the mode (override with `PRIMA_ALLOW_UNCAPPED=1`):

```bash
sudo -v && ./scripts/setup/apply_power_mode_limits.sh
```

Then verify the node; it lists every missing item:

```bash
./scripts/setup/check_orin_env.sh
```

### 3.3 Notes

- Runners force `LC_ALL=C`, so sysstat logs are parsed correctly on machines
  with a non-English locale. Logs collected earlier under a Korean locale are
  also accepted by the organizers.
- Run `sudo -v` first and start the experiment scripts as the normal user; they
  call `sudo -n` only for `perf_event_paranoid`, swap and cgroup control.
- If experiments are launched from a separate master node, configure SSH access
  to the edge node and run:

```bash
ORIN_HOST=$EDGE_HOST ORIN_PORT=$EDGE_PORT ORIN_REPO=$PRIMA_WORK_ROOT/PRIMA-Artifact \
  ./scripts/master/run_on_orin1.sh check-env
```

## 4. Experiments

Run these wrappers directly on the edge node.

| Experiment | Command | Summary |
|---|---|---|
| Isolated baseline | `./scripts/experiments/run_isolated_baseline.sh` | Runs each YOLO workload alone and collects latency, elapsed time, pidstat memory, and tegrastats logs. |
| Concurrent comparison | `./scripts/experiments/run_concurrent_comparison.sh` | Runs five YOLO workloads under TS, MPS, PRIMA-TS, and PRIMA-MPS; collects latency, memory, page faults, CPU/GPU utilization, and comparison CSVs. |
| Memory sweep | `./scripts/experiments/run_memory_sweep.sh` | Sweeps isolated Docker memory limits from 0.5 GB to 2.5 GB; collects completion threshold, latency, memory, page faults, and GPU utilization. |
| MPS partitioning | `./scripts/experiments/run_mps_partitioning.sh` | Compares MPS 20% and MPS 50%; collects normalized latency, memory usage, and page-fault behavior. |
| Goodput latency | `./scripts/experiments/run_goodput_latency.sh` | Runs five concurrent YOLO goodput workloads for 5,000 requests per workload and exports latency summary CSVs. |
| CLIP generalization | `./scripts/experiments/run_clip_generalization.sh` | Profiles CLIP ViT-B/32 and RN50; collects peak RAM, peak container memory, elapsed time, and CLIP summaries. |
| PRIMA overhead | `./scripts/experiments/run_prima_overhead.sh` | Measures feature extraction, RF prediction, and Calculator overhead over the ONNX model list. |

From a master node, use the same launcher with a target name:

```bash
./scripts/master/run_on_orin1.sh <target>
```

| Target | Runs on Orin1 |
|---|---|
| `ch3-isolated` | Isolated baseline |
| `ch3-concurrent` | Concurrent TS and MPS 20% only |
| `ch3-sweep` | Isolated memory sweep |
| `ch5-concurrent` | Full concurrent comparison |
| `ch5-mps` | MPS 20% vs. MPS 50% partitioning |
| `ch5-goodput` | Goodput latency |
| `ch5-clip` | CLIP generalization |
| `ch5-overhead` | PRIMA overhead |

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

Large or device-specific assets are excluded from GitHub and prepared with the
scripts in Section 3.2:

| Asset | Location | Prepared by |
|---|---|---|
| Workload scripts, `.pt`, `.onnx`, `.engine` | `$PRIMA_ARTIFACT_ROOT/yolo/workload` | `install_workloads.sh`; `export_yolo_engines.sh` |
| Paper image set (orin1 `yolo_new/expanded_*`) and lists | `$PRIMA_ARTIFACT_ROOT/yolo/workload` (or `yolo_runtime`) | copied; `prepare_yolo_inputs.sh` checks it |
| Goodput datasets | `$PRIMA_SSD_ROOT/prima_goodput_5000` | `prepare_goodput_datasets.sh` |
| ONNX files for overhead measurement | `$PRIMA_ARTIFACT_ROOT/onnx_cache` | `export_yolo_engines.sh` |
| CLIP image, weights and CIFAR-100 | `clip-full:jetson`, `$PRIMA_ARTIFACT_ROOT/clip` | `build_clip_image.sh` |
