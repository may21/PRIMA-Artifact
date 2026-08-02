#!/bin/bash
set -euo pipefail

REPEATS="${1:-1}"
IMAGE="${IMAGE:-ultralytics/ultralytics:8.3.102-jetson-jetpack6}"
PRIMA_ARTIFACT_ROOT="${PRIMA_ARTIFACT_ROOT:-/home/orin1/prima_artifacts}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${PRIMA_ARTIFACT_ROOT}/yolo}"
HOST_YOLO_DIR="${HOST_YOLO_DIR:-${EXPERIMENT_ROOT}/workload}"
DATASET_ROOT="${DATASET_ROOT:-${PRIMA_ARTIFACT_ROOT}/yolo_runtime}"
RESULT_BASE="${RESULT_BASE:-${EXPERIMENT_ROOT}/results}"
CONTAINER_YOLO_DIR="/usr/src/ultralytics/yolo_new"
START_DELAY_SECONDS=7
PERF_RETRY_SECONDS=10
IMAGE_LIMIT="${IMAGE_LIMIT:-100}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

read -r -a WORKLOADS <<< "${WORKLOAD_LIST:-classify detect pose segment obb}"
MEMORY_LABELS=(0.5GB 1.0GB 1.5GB 2.0GB 2.5GB)
MEMORY_LIMITS=(512m 1g 1536m 2g 2560m)
read -r -a MEMORY_INDEXES <<< "${MEMORY_INDEX_LIST:-0 1 2 3 4}"
REPEAT_START="${REPEAT_START:-1}"
REPEAT_END="${REPEAT_END:-$REPEATS}"

declare -A SCRIPTS=(
  [classify]="classify.py"
  [detect]="detect.py"
  [pose]="pose.py"
  [segment]="segment.py"
  [obb]="obb.py"
)

if [[ -n "${RESULT_ROOT_OVERRIDE:-}" ]]; then
  RESULT_ROOT="$RESULT_ROOT_OVERRIDE"
else
  RUN_INDEX=1
  while [[ -e "${RESULT_BASE}/isolated_memory_sweep_100_${RUN_INDEX}" ]]; do
    RUN_INDEX=$((RUN_INDEX + 1))
  done
  RESULT_ROOT="${RESULT_BASE}/isolated_memory_sweep_100_${RUN_INDEX}"
fi
SUMMARY="${RESULT_ROOT}/summary.csv"

MONITOR_PIDS=()
PERF_PID=""
STATS_PID=""
CGROUP_PID=""
ACTIVE_CONTAINER=""
ORIGINAL_PERF_PARANOID=""

stop_pid() {
  local pid="${1:-}"
  [[ -z "$pid" ]] && return
  kill -INT "$pid" >/dev/null 2>&1 || true
  sleep 0.2
  kill -TERM "$pid" >/dev/null 2>&1 || true
}

stop_monitors() {
  local pid
  stop_pid "$PERF_PID"
  stop_pid "$STATS_PID"
  stop_pid "$CGROUP_PID"
  for pid in "${MONITOR_PIDS[@]:-}"; do stop_pid "$pid"; done
  MONITOR_PIDS=()
  PERF_PID=""
  STATS_PID=""
  CGROUP_PID=""
}

cleanup_container() {
  if [[ -n "$ACTIVE_CONTAINER" ]]; then
    docker rm -f "$ACTIVE_CONTAINER" >/dev/null 2>&1 || true
  fi
  ACTIVE_CONTAINER=""
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  stop_monitors
  cleanup_container
  if [[ -n "$ORIGINAL_PERF_PARANOID" ]]; then
    sudo -n sysctl -w "kernel.perf_event_paranoid=${ORIGINAL_PERF_PARANOID}" \
      >/dev/null 2>&1 || true
  fi
  exit "$rc"
}
trap cleanup EXIT INT TERM

for command_name in docker perf pidstat mpstat vmstat top tegrastats; do
  command -v "$command_name" >/dev/null 2>&1 || {
    echo "[ERROR] missing command: $command_name" >&2
    exit 1
  }
done
docker image inspect "$IMAGE" >/dev/null 2>&1 || {
  echo "[ERROR] missing image: $IMAGE" >&2
  exit 1
}

echo "[INFO] Verify swap/perf prerequisites."
if [[ -s /proc/swaps ]] && [[ "$(wc -l < /proc/swaps)" -gt 1 ]]; then
  timeout 20s sudo -n swapoff -a || true
fi
if [[ -s /proc/swaps ]] && [[ "$(wc -l < /proc/swaps)" -gt 1 ]]; then
  echo "[ERROR] host swap is still active" >&2
  cat /proc/swaps >&2
  exit 1
fi
sudo -n true || {
  echo "[ERROR] sudo credential is not cached; run 'sudo -v' before launching this script" >&2
  exit 1
}
ORIGINAL_PERF_PARANOID="$(cat /proc/sys/kernel/perf_event_paranoid)"
sudo -n sysctl -w kernel.perf_event_paranoid=-1 >/dev/null
sudo -n bash -c '
  parent="$1"; original="$2"
  while kill -0 "$parent" 2>/dev/null; do sleep 10; done
  sysctl -w "kernel.perf_event_paranoid=${original}" >/dev/null
' _ "$$" "$ORIGINAL_PERF_PARANOID" >/dev/null 2>&1 &

mkdir -p "$RESULT_ROOT"
if [[ ! -f "$SUMMARY" ]]; then
  echo "workload,memory_label,memory_limit,repeat,container,exit_status,oom_killed,memory_peak_bytes,result_dir" > "$SUMMARY"
fi

{
  echo "experiment=isolated_memory_sweep"
  echo "device=$(tr -d '\0' </proc/device-tree/model)"
  echo "jetson_release=$(head -1 /etc/nv_tegra_release)"
  echo "image=$IMAGE"
  echo "repeats=$REPEATS"
  echo "repeat_range=${REPEAT_START}-${REPEAT_END}"
  echo "image_limit=${IMAGE_LIMIT}"
  echo "host_yolo_dir=${HOST_YOLO_DIR}"
  echo "dataset_root=${DATASET_ROOT}"
  echo "memory_limits=${MEMORY_LIMITS[*]}"
  echo "swap_policy=host swap off; docker memory-swap equals memory limit; cgroup swap.current sampled"
  echo "perf_events=page-faults,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,L1-icache-loads,L1-icache-load-misses,l2d_cache,l2d_cache_refill,LLC-loads,LLC-load-misses"
  nvpmodel -q 2>/dev/null | tr '\n' ' '
  echo
} > "${RESULT_ROOT}/experiment_config.txt"

echo "[INFO] result root: $RESULT_ROOT"
echo "[INFO] total runs: $((${#WORKLOADS[@]} * ${#MEMORY_INDEXES[@]} * (REPEAT_END - REPEAT_START + 1)))"

for workload in "${WORKLOADS[@]}"; do
  for memory_index in "${MEMORY_INDEXES[@]}"; do
    memory_label="${MEMORY_LABELS[$memory_index]}"
    memory_limit="${MEMORY_LIMITS[$memory_index]}"

    for repeat in $(seq "$REPEAT_START" "$REPEAT_END"); do
      RUN_DIR="${RESULT_ROOT}/${workload}/${memory_label}/run${repeat}"
      mkdir -p "$RUN_DIR/runs3"
      cname="memsweep-${workload}-${memory_label//./p}-r${repeat}"
      ACTIVE_CONTAINER="$cname"

      echo "============================================================"
      echo "[RUN] workload=$workload memory=$memory_label repeat=$repeat"
      echo "[RUN] result_dir=$RUN_DIR"
      echo "============================================================"

      stop_monitors
      cleanup_container
      ACTIVE_CONTAINER="$cname"
      docker rm -f "$cname" >/dev/null 2>&1 || true

      docker create \
        --name "$cname" \
        --runtime=nvidia \
        --ipc=host \
        --user "$(id -u):$(id -g)" \
        --memory="$memory_limit" \
        --memory-swap="$memory_limit" \
        --memory-swappiness=0 \
        --pull=never \
        -e CUDA_VISIBLE_DEVICES=0 \
        -e HOME=/tmp \
        -e MPLCONFIGDIR=/tmp/matplotlib \
        -e YOLO_CONFIG_DIR=/tmp/ultralytics \
        -e IMAGE_LIMIT="${IMAGE_LIMIT}" \
        -v "${HOST_YOLO_DIR}:${CONTAINER_YOLO_DIR}" \
        -v "${DATASET_ROOT}/expanded_imagenet_images:${CONTAINER_YOLO_DIR}/expanded_imagenet_images:ro" \
        -v "${DATASET_ROOT}/expanded_coco_images:${CONTAINER_YOLO_DIR}/expanded_coco_images:ro" \
        -v "${DATASET_ROOT}/expanded_dota_images:${CONTAINER_YOLO_DIR}/expanded_dota_images:ro" \
        -v "${RUN_DIR}/runs3:${CONTAINER_YOLO_DIR}/runs3" \
        "$IMAGE" \
        bash -lc "cd '${CONTAINER_YOLO_DIR}' && exec python3 '${SCRIPTS[$workload]}'" \
        >/dev/null

      docker inspect "$cname" > "${RUN_DIR}/docker_inspect_before.json"
      cat > "${RUN_DIR}/config.txt" <<EOF
workload=${workload}
script=${SCRIPTS[$workload]}
memory_label=${memory_label}
memory_limit=${memory_limit}
memory_swap=${memory_limit}
repeat=${repeat}
container=${cname}
image=${IMAGE}
image_limit=${IMAGE_LIMIT}
EOF

      echo "[WAIT] ${START_DELAY_SECONDS}s"
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

      docker start "$cname" >/dev/null

      host_pid=""
      for retry in $(seq 1 "$PERF_RETRY_SECONDS"); do
        host_pid="$(
          docker top "$cname" -eo pid,comm,args 2>/dev/null |
            awk '$2 ~ /^python(3)?$/ {print $1; exit}'
        )"
        [[ -n "$host_pid" ]] && break
        sleep 1
      done

      if [[ -n "$host_pid" ]]; then
        echo "$host_pid" > "${RUN_DIR}/python_pid.txt"
        perf stat -I 1000 -x, \
          -e page-faults,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,L1-icache-loads,L1-icache-load-misses,l2d_cache,l2d_cache_refill,LLC-loads,LLC-load-misses \
          -p "$host_pid" > "${RUN_DIR}/perf.csv" 2>&1 &
        PERF_PID=$!

        cgroup_rel="$(awk -F: '$1=="0"{print $3}' "/proc/${host_pid}/cgroup" 2>/dev/null || true)"
        if [[ -n "$cgroup_rel" && -d "/sys/fs/cgroup${cgroup_rel}" ]]; then
          CGROUP_DIR="/sys/fs/cgroup${cgroup_rel}"
          echo "epoch_ns,memory_current_bytes,memory_peak_bytes,memory_swap_current_bytes" > "${RUN_DIR}/cgroup_memory.csv"
          (
            while kill -0 "$host_pid" 2>/dev/null; do
              current="$(cat "${CGROUP_DIR}/memory.current" 2>/dev/null || echo '')"
              peak="$(cat "${CGROUP_DIR}/memory.peak" 2>/dev/null || echo '')"
              swap_current="$(cat "${CGROUP_DIR}/memory.swap.current" 2>/dev/null || echo '')"
              echo "$(date +%s%N),${current},${peak},${swap_current}"
              sleep 0.5
            done
          ) >> "${RUN_DIR}/cgroup_memory.csv" &
          CGROUP_PID=$!
          echo "$CGROUP_DIR" > "${RUN_DIR}/cgroup_path.txt"
        fi
      else
        echo "[WARN] Python PID not found" | tee "${RUN_DIR}/perf_not_attached.txt"
      fi

      echo "epoch_ns,memory_usage,memory_percent" > "${RUN_DIR}/docker_memory.csv"
      (
        while [[ "$(docker inspect -f '{{.State.Running}}' "$cname" 2>/dev/null || echo false)" == "true" ]]; do
          stats="$(docker stats --no-stream --format '{{.MemUsage}},{{.MemPerc}}' "$cname" 2>/dev/null || true)"
          [[ -n "$stats" ]] && echo "$(date +%s%N),$stats"
          sleep 0.5
        done
      ) >> "${RUN_DIR}/docker_memory.csv" &
      STATS_PID=$!

      exit_status="$(docker wait "$cname" 2>/dev/null || echo docker_wait_error)"
      echo "end,$(date +%s%N)" >> "${RUN_DIR}/time.csv"
      stop_monitors

      docker logs "$cname" > "${RUN_DIR}/output.txt" 2>&1 || true
      docker inspect "$cname" > "${RUN_DIR}/docker_inspect_after.json"
      oom_killed="$(docker inspect -f '{{.State.OOMKilled}}' "$cname" 2>/dev/null || echo unknown)"

      memory_peak=""
      if [[ -f "${RUN_DIR}/cgroup_memory.csv" ]]; then
        memory_peak="$(
          awk -F, '
            NR>1 {
              if($2 ~ /^[0-9]+$/ && $2>current_max) current_max=$2
              if($3 ~ /^[0-9]+$/ && $3>peak_max) peak_max=$3
            }
            END {
              if(peak_max) print peak_max
              else if(current_max) print current_max
            }
          ' "${RUN_DIR}/cgroup_memory.csv"
        )"
      fi

      echo "${workload},${memory_label},${memory_limit},${repeat},${cname},${exit_status},${oom_killed},${memory_peak},${RUN_DIR}" >> "$SUMMARY"

      if [[ "$exit_status" == "0" ]]; then
        echo "[OK] workload=$workload memory=$memory_label repeat=$repeat"
      else
        echo "[WARN] workload=$workload memory=$memory_label repeat=$repeat exit=$exit_status oom=$oom_killed"
      fi

      cleanup_container
      sleep 2
    done
  done
done

echo "[ORGANIZE] building isolated-memory CSV files"
python3 "${SCRIPT_DIR}/collect_memory_sweep_results.py" "$RESULT_ROOT" || true
python3 "${SCRIPT_DIR}/collect_memory_sweep_raw_metrics.py" "$RESULT_ROOT" || true

echo "[DONE] result root: $RESULT_ROOT"
echo "[DONE] summary: $SUMMARY"
