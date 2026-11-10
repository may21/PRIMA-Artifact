#!/usr/bin/env bash
# Shared helpers for PRIMA experiment runners. Source it near the top of a
# runner:  source "$(dirname "${BASH_SOURCE[0]}")/../../lib/prima_common.sh"
#
# What it provides
#   * LC_ALL=C for the whole runner, so pidstat/mpstat/vmstat/top print the
#     English 24-hour format the organizers parse (a ko_KR locale prints
#     "오후 03시 12분 01초" and breaks every parser).
#   * Path defaults that follow one layout on any machine:
#       PRIMA_WORK_ROOT/            (optional umbrella directory)
#         prima_artifacts/          -> PRIMA_ARTIFACT_ROOT
#         ssd/                      -> PRIMA_SSD_ROOT (goodput datasets)
#     Without PRIMA_WORK_ROOT the original orin1 defaults are kept.
#   * prima_resolve_dataset_root / prima_resolve_yolo_root: pick the
#     directory that actually holds the datasets / TensorRT engines.
#   * prima_resolve_perf: a perf binary that really runs on this kernel
#     (Jetson's /usr/bin/perf wrapper often cannot find a matching binary).
#   * prima_tty_prepare / prima_tty_restore: keep the terminal usable.

export LC_ALL=C
export LANG=C

if [[ -n "${PRIMA_WORK_ROOT:-}" ]]; then
  : "${PRIMA_ARTIFACT_ROOT:=${PRIMA_WORK_ROOT}/prima_artifacts}"
  : "${PRIMA_SSD_ROOT:=${PRIMA_WORK_ROOT}/ssd}"
fi
: "${PRIMA_ARTIFACT_ROOT:=/home/orin1/prima_artifacts}"
: "${PRIMA_SSD_ROOT:=/home/orin1/ssd}"
export PRIMA_ARTIFACT_ROOT PRIMA_SSD_ROOT

# Directory that contains expanded_{imagenet,coco,dota}_images.
# Order: $DATASET_ROOT; otherwise the first of $PRIMA_ARTIFACT_ROOT/paper_images,
# $PRIMA_ARTIFACT_ROOT/yolo_runtime, <experiment>/workload that holds the
# images named in the installed image lists (the paper set); if no list is
# installed, the first of those that holds expanded_coco_images.
prima_resolve_dataset_root() {
  local experiment_root="${1:-${PRIMA_ARTIFACT_ROOT}/yolo}"
  if [[ -n "${DATASET_ROOT:-}" ]]; then
    echo "$DATASET_ROOT"
    return
  fi
  local -a candidates=("${PRIMA_ARTIFACT_ROOT}/paper_images" "${PRIMA_ARTIFACT_ROOT}/yolo_runtime" "${experiment_root}/workload")
  local candidate list probe=""
  for list in "${experiment_root}"/workload/image_lists/dota_*.txt; do
    [[ -f "$list" ]] && probe="$(basename "$(head -1 "$list")")" && break
  done
  if [[ -n "$probe" ]]; then
    for candidate in "${candidates[@]}"; do
      if [[ -f "${candidate}/expanded_dota_images/${probe}" ]]; then
        echo "$candidate"
        return
      fi
    done
  fi
  for candidate in "${candidates[@]}"; do
    if [[ -d "${candidate}/expanded_coco_images" ]]; then
      echo "$candidate"
      return
    fi
  done
  echo "${experiment_root}/workload"
}

# Stop before a run when an image named in the lists is not in the dataset
# root. Usage: prima_check_image_set DATASET_ROOT LIST_DIR IMAGE_LIMIT
prima_check_image_set() {
  local dataset_root="$1" list_dir="$2" limit="$3" ds list entry missing checked=0
  for ds in imagenet coco dota; do
    list="${list_dir}/${ds}_${limit}.txt"
    [[ -f "$list" ]] || continue
    checked=$((checked + 1))
    missing=0
    while IFS= read -r entry; do
      [[ -z "$entry" ]] && continue
      [[ -f "${dataset_root}/expanded_${ds}_images/$(basename "$entry")" ]] || missing=$((missing + 1))
    done < "$list"
    if (( missing > 0 )); then
      echo "[ERROR] ${missing} image(s) listed in ${list} are missing from ${dataset_root}/expanded_${ds}_images." >&2
      echo "        Set DATASET_ROOT to the folder holding the paper images (expanded_*), or see README Section 3.2." >&2
      exit 1
    fi
  done
  if (( checked == 0 )); then
    echo "[WARN] no image lists for IMAGE_LIMIT=${limit} in ${list_dir}; the image set is not checked." >&2
  else
    echo "[INFO] image set OK: ${dataset_root} (lists ${list_dir}/*_${limit}.txt)"
  fi
}

# Directory that contains the yolov8n*.engine files (goodput runner).
# Order: $YOLO_ROOT, $PRIMA_ARTIFACT_ROOT/yolo_runtime, $PRIMA_ARTIFACT_ROOT/yolo/workload.
prima_resolve_yolo_root() {
  if [[ -n "${YOLO_ROOT:-}" ]]; then
    echo "$YOLO_ROOT"
    return
  fi
  local candidate
  for candidate in "${PRIMA_ARTIFACT_ROOT}/yolo_runtime" "${PRIMA_ARTIFACT_ROOT}/yolo/workload"; do
    if [[ -f "${candidate}/yolov8n.engine" ]]; then
      echo "$candidate"
      return
    fi
  done
  echo "${PRIMA_ARTIFACT_ROOT}/yolo_runtime"
}

# Echo a perf binary that works on the running kernel, or return 1.
# Order: $PERF_BIN, perf on PATH, Jetson tegra tools, generic linux-tools.
prima_resolve_perf() {
  local candidate
  local -a candidates=()
  [[ -n "${PERF_BIN:-}" ]] && candidates+=("$PERF_BIN")
  candidate="$(command -v perf 2>/dev/null || true)"
  [[ -n "$candidate" ]] && candidates+=("$candidate")
  for candidate in /usr/lib/linux-nvidia-tegra-tools-*/perf /usr/lib/linux-tools-*/perf; do
    [[ -x "$candidate" ]] && candidates+=("$candidate")
  done
  for candidate in "${candidates[@]}"; do
    if [[ -x "$candidate" ]] && "$candidate" --version >/dev/null 2>&1; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

# Sets PERF_BIN or exits with an installation hint.
prima_require_perf() {
  if ! PERF_BIN="$(prima_resolve_perf)"; then
    cat >&2 <<'MSG'
[ERROR] no working perf binary for this kernel.
        On Jetson: sudo apt install linux-nvidia-tegra-tools-$(uname -r | sed 's/-tegra$//' | cut -d- -f1)-<abi>
        (e.g. linux-nvidia-tegra-tools-5.15.0-1030), or set PERF_BIN=/path/to/perf.
MSG
    exit 1
  fi
  export PERF_BIN
  echo "[INFO] perf: ${PERF_BIN}"
}

# Fail early when a required command is missing.
prima_require_commands() {
  local name
  for name in "$@"; do
    command -v "$name" >/dev/null 2>&1 || {
      echo "[ERROR] missing command: $name" >&2
      exit 1
    }
  done
}

# True while PID is running (zombies, i.e. exited but not yet reaped, count as gone).
prima_pid_alive() {
  local state
  kill -0 "$1" 2>/dev/null || return 1
  state="$(ps -o stat= -p "$1" 2>/dev/null || true)"
  [[ -n "$state" && "$state" != Z* ]]
}

# Stop background PIDs: SIGINT, then SIGTERM, then SIGKILL after ~3 s, and reap them.
prima_stop_pids() {
  local pid
  local -a alive=()
  for pid in "$@"; do
    [[ -n "$pid" ]] && prima_pid_alive "$pid" && alive+=("$pid")
  done
  if [[ ${#alive[@]} -gt 0 ]]; then
    for pid in "${alive[@]}"; do kill -INT "$pid" >/dev/null 2>&1 || true; done
    sleep 0.5
    for pid in "${alive[@]}"; do kill -TERM "$pid" >/dev/null 2>&1 || true; done
    local waited=0 any
    while (( waited < 30 )); do
      any=0
      for pid in "${alive[@]}"; do prima_pid_alive "$pid" && any=1; done
      (( any == 0 )) && break
      sleep 0.1
      waited=$((waited + 1))
    done
    for pid in "${alive[@]}"; do
      prima_pid_alive "$pid" && kill -KILL "$pid" >/dev/null 2>&1 || true
    done
  fi
  for pid in "$@"; do
    [[ -n "$pid" ]] && wait "$pid" 2>/dev/null || true
  done
  return 0
}

prima_tty_prepare() {
  if [[ -t 0 ]]; then
    stty opost onlcr 2>/dev/null || true
  fi
}

prima_tty_restore() {
  if [[ -t 0 ]]; then
    stty sane 2>/dev/null || true
  fi
}
