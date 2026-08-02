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

Prepare these assets on the edge node before running the full artifact:

| Asset | Notes |
|---|---|
| YOLO workload assets and results | `/home/orin1/prima_artifacts/yolo` |
| YOLO runtime images and TensorRT engines | `/home/orin1/prima_artifacts/yolo_runtime` |
| Goodput datasets | `/home/orin1/ssd/prima_goodput_5000` |
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

## 6. Orin1 layout

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

Prepared Orin1:

Use the prepared GitHub clone as the working directory:

```bash
cd /home/orin1/woosy/PRIMA
```

The prepared Orin1 layout keeps code, external assets, and goodput datasets in
separate locations:

```text
/home/orin1/woosy/PRIMA              GitHub clone and experiment scripts
/home/orin1/prima_artifacts          External assets used by the scripts
/home/orin1/ssd/prima_goodput_5000   Goodput image datasets
```

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
| `/home/orin1/ssd/prima_goodput_5000` | Goodput image datasets prepared outside GitHub |

On Orin1, external assets are already placed under `prima_artifacts`, and
goodput images are stored under `/home/orin1/ssd/prima_goodput_5000`.

For goodput, CLS uses 5,000 ImageNet images. DET/EST/SEG use the same 5,000
COCO images. OBB uses the available DOTA val/test images as-is; if fewer than
5,000 DOTA images are present, the runner reuses the image list until it records
5,000 requests.

## 7. ETRI workload note

The ETRI workload was prepared on a separate Jetson Orin Nano edge node
(`orin-test`).

Workload directory on `orin-test`:

```text
/home/orin-test/etri2
```

Recorded directory contents:

```text
172.17.11.121.mp4
172.17.11.122.mp4
test_deepDet_lib_console_lprDet
test_deepDet_lib_console_speedDet
```

The workload used the JetPack-matched NVIDIA L4T image:

```text
nvcr.io/nvidia/l4t-jetpack:r36.4.0
image id: 51f1e16a5dd9
size: 9.83 GB
```

Container creation used the NVIDIA runtime option:

```bash
docker run -it --runtime=nvidia -v ~/etri2:/home 51f /bin/bash
```

Recorded container:

```text
container id: f7e3e981d06c
image: 51f
command: /bin/bash
name: kind_faraday
```

Restart and attach:

```bash
docker start f7e
docker attach f7e
```

After attaching to the container, the mounted `/home` directory contains the
ETRI workload files.

Run LPR detection:

```bash
cd /home
cd test_deepDet_lib_console_lprDet
./etriDeepDet
```

Run speed detection:

```bash
cd /home
cd test_deepDet_lib_console_speedDet
./etriDeepDet
```

During execution, the program prints recognition results for each frame and
creates date/time-stamped result directories under `_Result`.

Boot-time command sequence on the Jetson board:

```bash
mount /dev/sda1 ~/etri
docker start c6c
docker attach c6c
cd /home
cd test_deepDet_lib_console_lprDet
./etriDeepDet
```

Use the container id shown by `docker ps -a` if it differs from the recorded
`f7e` or `c6c` prefix.
