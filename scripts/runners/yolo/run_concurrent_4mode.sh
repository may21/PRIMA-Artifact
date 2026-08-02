#!/bin/bash
set -euo pipefail

# Four concurrent-execution conditions:
#   1) ts_unlimited       : Time-slicing, no per-container memory limit
#   2) mps20_unlimited    : MPS 20% per context (= 2 SM on this Orin), no memory limit
#   3) ts_memlimit        : Time-slicing, workload-specific memory limits
#   4) mps20_memlimit     : MPS 20% per context + workload-specific memory limits
#
# All conditions:
#   - start five workloads concurrently through a shared start gate
#   - set cgroup v2 memory.swap.max=0 before opening the gate
#   - wait 7 seconds before simultaneous workload release
#   - retry up to 10 seconds to attach perf to each workload
#   - collect pidstat/mpstat/vmstat/top/tegrastats and cgroup memory
#
# Usage:
#   ./run_concurrent_4mode.sh       # default: 3 repeats per mode
#   ./run_concurrent_4mode.sh 3

REPEATS="${1:-3}"
IMAGE_LIMIT="${IMAGE_LIMIT:-100}"
IMAGE="${IMAGE:-ultralytics/ultralytics:8.3.102-jetson-jetpack6}"
PRIMA_ARTIFACT_ROOT="${PRIMA_ARTIFACT_ROOT:-/home/orin1/prima_artifacts}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${PRIMA_ARTIFACT_ROOT}/yolo}"
HOST_YOLO_DIR="${HOST_YOLO_DIR:-${EXPERIMENT_ROOT}/workload}"
DATASET_ROOT="${DATASET_ROOT:-${EXPERIMENT_ROOT}/workload}"
RESULT_BASE="${RESULT_BASE:-${EXPERIMENT_ROOT}/results}"
CONTAINER_YOLO_DIR="/usr/src/ultralytics/yolo_new"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MPS_PIPE_DIR="/tmp/nvidia-mps"
MPS_LOG_DIR="/tmp/nvidia-log"
START_DELAY_SECONDS=7
PERF_RETRY_SECONDS=10

read -r -a WORKLOADS <<< "${WORKLOAD_LIST:-classify detect pose segment obb}"
read -r -a MODES <<< "${MODE_LIST:-ts_unlimited mps20_unlimited ts_memlimit mps20_memlimit}"
REPEAT_START="${REPEAT_START:-1}"
REPEAT_END="${REPEAT_END:-$REPEATS}"

declare -A SCRIPTS=(
  [classify]="classify.py"
  [detect]="detect.py"
  [pose]="pose.py"
  [segment]="segment.py"
  [obb]="obb.py"
)

# Original per-workload Docker memory limits.
declare -A MEM_LIMITS=(
  [classify]="821m"
  [detect]="1071m"
  [pose]="1068m"
  [segment]="1649m"
  [obb]="2233m"
)

PERF_EVENTS="page-faults,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,L1-icache-loads,L1-icache-load-misses,l2d_cache,l2d_cache_refill,LLC-loads,LLC-load-misses"

if [[ -n "${RESULT_ROOT_OVERRIDE:-}" ]]; then
  RESULT_ROOT="$RESULT_ROOT_OVERRIDE"
else
  RUN_INDEX=1
  while [[ -e "${RESULT_BASE}/concurrent_4mode_${RUN_INDEX}" ]]; do
    RUN_INDEX=$((RUN_INDEX + 1))
  done
  RESULT_ROOT="${RESULT_BASE}/concurrent_4mode_${RUN_INDEX}"
fi
SUMMARY="${RESULT_ROOT}/summary.csv"

ACTIVE_CONTAINERS=()
MONITOR_PIDS=()
PERF_PIDS=()
CGROUP_MONITOR_PIDS=()
ORIGINAL_PERF_PARANOID=""

is_mps_mode() {
  [[ "$1" == "mps20_unlimited" || "$1" == "mps20_memlimit" ]]
}

is_memlimit_mode() {
  [[ "$1" == "ts_memlimit" || "$1" == "mps20_memlimit" ]]
}

stop_pid_list() {
  local pid
  for pid in "$@"; do
    [[ -z "$pid" ]] && continue
    kill -INT "$pid" >/dev/null 2>&1 || true
  done
  sleep 0.5
  for pid in "$@"; do
    [[ -z "$pid" ]] && continue
    kill -TERM "$pid" >/dev/null 2>&1 || true
  done
}

stop_monitors() {
  stop_pid_list "${PERF_PIDS[@]:-}"
  stop_pid_list "${CGROUP_MONITOR_PIDS[@]:-}"
  stop_pid_list "${MONITOR_PIDS[@]:-}"
  PERF_PIDS=()
  CGROUP_MONITOR_PIDS=()
  MONITOR_PIDS=()
}

remove_containers() {
  local cname
  for cname in "${ACTIVE_CONTAINERS[@]:-}"; do
    [[ -z "$cname" ]] && continue
    docker rm -f "$cname" >/dev/null 2>&1 || true
  done
  ACTIVE_CONTAINERS=()
}

stop_mps() {
  CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE_DIR" \
    nvidia-cuda-mps-control <<< quit >/dev/null 2>&1 || true
  pkill -f '^nvidia-cuda-mps-server' >/dev/null 2>&1 || true
  pkill -f '^nvidia-cuda-mps-control -d' >/dev/null 2>&1 || true
}

start_mps() {
  stop_mps
  rm -rf "$MPS_PIPE_DIR" "$MPS_LOG_DIR"
  mkdir -p "$MPS_PIPE_DIR" "$MPS_LOG_DIR"
  chown -R "$(id -u):$(id -g)" "$MPS_PIPE_DIR" "$MPS_LOG_DIR"

  export CUDA_VISIBLE_DEVICES=0
  export CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE_DIR"
  export CUDA_MPS_LOG_DIRECTORY="$MPS_LOG_DIR"
  nvidia-cuda-mps-control -d
  sleep 1

  pgrep -f '^nvidia-cuda-mps-control -d' >/dev/null || {
    echo "[ERROR] MPS control daemon did not start" >&2
    exit 1
  }
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  stop_monitors
  remove_containers
  stop_mps
  if [[ -n "$ORIGINAL_PERF_PARANOID" ]]; then
    sudo -n sysctl -w "kernel.perf_event_paranoid=${ORIGINAL_PERF_PARANOID}" \
      >/dev/null 2>&1 || true
  fi
  exit "$rc"
}
trap cleanup EXIT INT TERM

if ! [[ "$REPEATS" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] repeats must be a positive integer" >&2
  exit 2
fi
if ! [[ "$IMAGE_LIMIT" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] IMAGE_LIMIT must be a positive integer" >&2
  exit 2
fi

for command_name in docker perf pidstat mpstat vmstat top tegrastats nvidia-cuda-mps-control; do
  command -v "$command_name" >/dev/null 2>&1 || {
    echo "[ERROR] missing command: $command_name" >&2
    exit 1
  }
done

if [[ "$(stat -fc %T /sys/fs/cgroup)" != "cgroup2fs" ]]; then
  echo "[ERROR] cgroup v2 is required to enforce memory.swap.max=0" >&2
  exit 1
fi

docker image inspect "$IMAGE" >/dev/null 2>&1 || {
  echo "[ERROR] Docker image not found: $IMAGE" >&2
  exit 1
}

for workload in "${WORKLOADS[@]}"; do
  [[ -f "${HOST_YOLO_DIR}/${SCRIPTS[$workload]}" ]] || {
    echo "[ERROR] missing script: ${SCRIPTS[$workload]}" >&2
    exit 1
  }
done
for list_name in imagenet coco dota; do
  [[ -f "${HOST_YOLO_DIR}/image_lists/${list_name}_${IMAGE_LIMIT}.txt" ]] || {
    echo "[ERROR] missing image list: image_lists/${list_name}_${IMAGE_LIMIT}.txt" >&2
    exit 1
  }
done

echo "[INFO] Authenticate once for perf and cgroup swap control."
sudo -v
if [[ -s /proc/swaps ]] && [[ "$(wc -l < /proc/swaps)" -gt 1 ]]; then
  timeout 20s sudo -n swapoff -a || true
fi
if [[ -s /proc/swaps ]] && [[ "$(wc -l < /proc/swaps)" -gt 1 ]]; then
  echo "[ERROR] host swap is still active after swapoff -a" >&2
  cat /proc/swaps >&2
  exit 1
fi
ORIGINAL_PERF_PARANOID="$(cat /proc/sys/kernel/perf_event_paranoid)"
sudo -n sysctl -w kernel.perf_event_paranoid=-1 >/dev/null

# Restore perf_event_paranoid even if the parent is killed.
sudo -n bash -c '
  parent="$1"; original="$2"
  while kill -0 "$parent" 2>/dev/null; do sleep 10; done
  sysctl -w "kernel.perf_event_paranoid=${original}" >/dev/null
' _ "$$" "$ORIGINAL_PERF_PARANOID" >/dev/null 2>&1 &

mkdir -p "$RESULT_ROOT"
echo "mode,repeat,workload,memory_limit,swap_max,container,exit_status,oom_killed,result_dir" > "$SUMMARY"

{
  echo "experiment=concurrent_4mode"
  echo "device=$(tr -d '\0' </proc/device-tree/model)"
  echo "jetson_release=$(head -1 /etc/nv_tegra_release)"
  echo "image=$IMAGE"
  echo "repeats=$REPEATS"
  echo "image_limit=$IMAGE_LIMIT"
  echo "requested_power_mode=${POWER_MODE_NAME:-unchanged}"
  echo "modes=${MODES[*]}"
  echo "mps_active_thread_percentage=20"
  echo "effective_mps_sm=2 (empirically verified on this device)"
  echo "swap_policy=memory.swap.max=0 for every container before workload release"
  echo "start_delay_seconds=$START_DELAY_SECONDS"
  echo "perf_events=$PERF_EVENTS"
  echo "nsight_compute=disabled"
  nvpmodel -q 2>/dev/null | tr '\n' ' '
  echo
} > "${RESULT_ROOT}/experiment_config.txt"

echo "[INFO] result root: $RESULT_ROOT"
echo "[INFO] repeat range: ${REPEAT_START}-${REPEAT_END}"
echo "[INFO] batches: $((${#MODES[@]} * (REPEAT_END - REPEAT_START + 1)))"
echo "[INFO] workload runs: $((${#MODES[@]} * (REPEAT_END - REPEAT_START + 1) * ${#WORKLOADS[@]}))"

for mode in "${MODES[@]}"; do
  for repeat in $(seq "$REPEAT_START" "$REPEAT_END"); do
    RUN_DIR="${RESULT_ROOT}/${mode}/run${repeat}"
    GATE_DIR="${RUN_DIR}/start_gate"
    mkdir -p "$GATE_DIR"
    rm -f "${GATE_DIR}/go"

    echo "============================================================"
    echo "[BATCH] mode=$mode repeat=$repeat"
    echo "[BATCH] result_dir=$RUN_DIR"
    echo "============================================================"

    stop_monitors
    remove_containers

    if is_mps_mode "$mode"; then
      start_mps
    else
      stop_mps
    fi

    ACTIVE_CONTAINERS=()
    for workload in "${WORKLOADS[@]}"; do
      workload_dir="${RUN_DIR}/${workload}"
      mkdir -p "${workload_dir}/runs3"
      cname="fourmode-${mode//_/-}-${workload}-r${repeat}"
      ACTIVE_CONTAINERS+=("$cname")
      docker rm -f "$cname" >/dev/null 2>&1 || true

      create_args=(
        docker create
        --name "$cname"
        --runtime=nvidia
        --ipc=host
        --user "$(id -u):$(id -g)"
        --pull=never
        -e CUDA_VISIBLE_DEVICES=0
        -e HOME=/tmp
        -e MPLCONFIGDIR=/tmp/matplotlib
        -e YOLO_CONFIG_DIR=/tmp/ultralytics
        -e IMAGE_LIMIT="$IMAGE_LIMIT"
        -v "${HOST_YOLO_DIR}:${CONTAINER_YOLO_DIR}"
        -v "${DATASET_ROOT}/expanded_imagenet_images:${CONTAINER_YOLO_DIR}/expanded_imagenet_images:ro"
        -v "${DATASET_ROOT}/expanded_coco_images:${CONTAINER_YOLO_DIR}/expanded_coco_images:ro"
        -v "${DATASET_ROOT}/expanded_dota_images:${CONTAINER_YOLO_DIR}/expanded_dota_images:ro"
        -v "${workload_dir}/runs3:${CONTAINER_YOLO_DIR}/runs3"
        -v "${GATE_DIR}:/start_gate"
      )

      memory_limit="unlimited"
      if is_memlimit_mode "$mode"; then
        memory_limit="${MEM_LIMITS[$workload]}"
        create_args+=(
          --memory="$memory_limit"
          --memory-swap="$memory_limit"
        )
      fi

      if is_mps_mode "$mode"; then
        create_args+=(
          -v "${MPS_PIPE_DIR}:${MPS_PIPE_DIR}"
          -v "${MPS_LOG_DIR}:${MPS_LOG_DIR}"
          -e CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE_DIR"
          -e CUDA_MPS_LOG_DIRECTORY="$MPS_LOG_DIR"
          -e CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING=1
          -e CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=20
        )
      fi

      create_args+=(
        "$IMAGE"
        bash -lc
        "while [[ ! -f /start_gate/go ]]; do sleep 0.02; done; cd '${CONTAINER_YOLO_DIR}'; exec python3 '${SCRIPTS[$workload]}'"
      )
      "${create_args[@]}" >/dev/null

      cat > "${workload_dir}/config.txt" <<EOF
mode=${mode}
repeat=${repeat}
workload=${workload}
script=${SCRIPTS[$workload]}
container=${cname}
memory_limit=${memory_limit}
swap_max=0
mps_active_thread_percentage=$([[ "$mode" == mps20_* ]] && echo 20 || echo unset)
effective_sm=$([[ "$mode" == mps20_* ]] && echo 2 || echo unrestricted)
image_limit=${IMAGE_LIMIT}
EOF
    done

    # Start all containers. Their workload commands remain blocked on the gate.
    START_PIDS=()
    for cname in "${ACTIVE_CONTAINERS[@]}"; do
      docker start "$cname" >/dev/null 2>&1 &
      START_PIDS+=("$!")
    done
    for start_pid in "${START_PIDS[@]}"; do wait "$start_pid"; done

    # Enforce zero swap in each container cgroup before any Python workload starts.
    for workload in "${WORKLOADS[@]}"; do
      cname="fourmode-${mode//_/-}-${workload}-r${repeat}"
      workload_dir="${RUN_DIR}/${workload}"
      init_pid="$(docker inspect -f '{{.State.Pid}}' "$cname")"
      cgroup_rel="$(awk -F: '$1=="0"{print $3}' "/proc/${init_pid}/cgroup")"
      cgroup_dir="/sys/fs/cgroup${cgroup_rel}"

      [[ -d "$cgroup_dir" ]] || {
        echo "[ERROR] cgroup not found for $cname: $cgroup_dir" >&2
        exit 1
      }
      [[ -f "${cgroup_dir}/memory.swap.max" ]] || {
        echo "[ERROR] memory.swap.max unavailable for $cname" >&2
        exit 1
      }

      echo 0 | sudo -n tee "${cgroup_dir}/memory.swap.max" >/dev/null
      swap_max="$(cat "${cgroup_dir}/memory.swap.max")"
      swap_current="$(cat "${cgroup_dir}/memory.swap.current")"
      [[ "$swap_max" == "0" && "$swap_current" == "0" ]] || {
        echo "[ERROR] swap control failed: $cname max=$swap_max current=$swap_current" >&2
        exit 1
      }

      echo "$init_pid" > "${workload_dir}/container_init_pid.txt"
      echo "$cgroup_dir" > "${workload_dir}/cgroup_path.txt"
      cat > "${workload_dir}/swap_control.txt" <<EOF
memory.swap.max=${swap_max}
memory.swap.current_before_start=${swap_current}
EOF

      echo "epoch_ns,memory_current_bytes,memory_peak_bytes,swap_current_bytes" \
        > "${workload_dir}/cgroup_memory.csv"
      (
        while [[ -d "$cgroup_dir" ]]; do
          current="$(cat "${cgroup_dir}/memory.current" 2>/dev/null || break)"
          peak="$(cat "${cgroup_dir}/memory.peak" 2>/dev/null || echo '')"
          swap="$(cat "${cgroup_dir}/memory.swap.current" 2>/dev/null || echo '')"
          echo "$(date +%s%N),${current},${peak},${swap}"
          sleep 0.5
        done
      ) >> "${workload_dir}/cgroup_memory.csv" &
      CGROUP_MONITOR_PIDS+=("$!")
    done

    echo "[WAIT] ${START_DELAY_SECONDS}s before simultaneous gate release"
    sleep "$START_DELAY_SECONDS"

    echo "event,epoch_ns" > "${RUN_DIR}/time.csv"
    echo "start,$(date +%s%N)" >> "${RUN_DIR}/time.csv"

    pidstat -h -r -u -I -t 1 > "${RUN_DIR}/system_pidstat.txt" 2>&1 &
    MONITOR_PIDS+=("$!")
    mpstat -P ALL 1 > "${RUN_DIR}/system_mpstat.txt" 2>&1 &
    MONITOR_PIDS+=("$!")
    vmstat 1 -t > "${RUN_DIR}/system_vmstat.txt" 2>&1 &
    MONITOR_PIDS+=("$!")
    top -b -d 1 > "${RUN_DIR}/system_top.txt" 2>&1 &
    MONITOR_PIDS+=("$!")
    tegrastats --interval 1000 --logfile "${RUN_DIR}/system_tegrastats.txt" &
    MONITOR_PIDS+=("$!")

    echo "[START] release all five workloads"
    touch "${GATE_DIR}/go"

    declare -A PERF_ATTACHED=()
    for retry in $(seq 1 "$PERF_RETRY_SECONDS"); do
      pending=0
      for workload in "${WORKLOADS[@]}"; do
        cname="fourmode-${mode//_/-}-${workload}-r${repeat}"
        workload_dir="${RUN_DIR}/${workload}"
        [[ -n "${PERF_ATTACHED[$cname]:-}" ]] && continue

        python_pid="$(
          docker top "$cname" -eo pid,comm,args 2>/dev/null |
            awk '$2 ~ /^python(3)?$/ {print $1; exit}'
        )"
        if [[ -n "$python_pid" ]]; then
          echo "$python_pid" > "${workload_dir}/python_pid.txt"
          perf stat -I 1000 -x, -e "$PERF_EVENTS" -p "$python_pid" \
            > "${workload_dir}/perf.csv" 2>&1 &
          PERF_PIDS+=("$!")
          PERF_ATTACHED["$cname"]="$python_pid"
          echo "[PERF] workload=$workload pid=$python_pid"
        else
          pending=$((pending + 1))
        fi
      done
      [[ "$pending" -eq 0 ]] && break
      echo "[PERF] retry=${retry}/${PERF_RETRY_SECONDS} pending=$pending"
      sleep 1
    done

    for workload in "${WORKLOADS[@]}"; do
      cname="fourmode-${mode//_/-}-${workload}-r${repeat}"
      workload_dir="${RUN_DIR}/${workload}"
      [[ -n "${PERF_ATTACHED[$cname]:-}" ]] ||
        echo "[WARN] perf not attached" > "${workload_dir}/perf_not_attached.txt"
    done

    for workload in "${WORKLOADS[@]}"; do
      cname="fourmode-${mode//_/-}-${workload}-r${repeat}"
      workload_dir="${RUN_DIR}/${workload}"
      exit_status="$(docker wait "$cname" 2>/dev/null || echo docker_wait_error)"
      docker logs "$cname" > "${workload_dir}/output.txt" 2>&1 || true
      docker inspect "$cname" > "${workload_dir}/docker_inspect_after.json"
      oom_killed="$(docker inspect -f '{{.State.OOMKilled}}' "$cname" 2>/dev/null || echo unknown)"
      swap_peak="$(
        awk -F, 'NR>1 && $4 ~ /^[0-9]+$/ && $4>m {m=$4} END{print m+0}' \
          "${workload_dir}/cgroup_memory.csv"
      )"
      memory_limit="unlimited"
      is_memlimit_mode "$mode" && memory_limit="${MEM_LIMITS[$workload]}"

      echo "${mode},${repeat},${workload},${memory_limit},0,${cname},${exit_status},${oom_killed},${workload_dir}" \
        >> "$SUMMARY"
      echo "memory.swap.peak_observed=${swap_peak}" >> "${workload_dir}/swap_control.txt"

      if [[ "$exit_status" == "0" && "$swap_peak" == "0" ]]; then
        echo "[OK] mode=$mode repeat=$repeat workload=$workload swap=0"
      else
        echo "[WARN] mode=$mode repeat=$repeat workload=$workload exit=$exit_status oom=$oom_killed swap_peak=$swap_peak"
      fi
    done

    echo "end,$(date +%s%N)" >> "${RUN_DIR}/time.csv"
    stop_monitors
    remove_containers
    stop_mps
    unset PERF_ATTACHED
    sleep 2
  done
done

echo "[ORGANIZE] building CSV files"
python3 "${SCRIPT_DIR}/collect_concurrent_results.py" "$RESULT_ROOT"
python3 "${SCRIPT_DIR}/organize_concurrent_results.py" "$RESULT_ROOT"

echo "[DONE] result root: $RESULT_ROOT"
echo "[DONE] summary: $SUMMARY"
