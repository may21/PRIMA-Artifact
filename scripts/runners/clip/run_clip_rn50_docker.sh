#!/bin/bash

# Keep cleanup active even when the workload fails.

WORKLOAD_NAME="clip_rn50_docker"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRIMA_ARTIFACT_ROOT="${PRIMA_ARTIFACT_ROOT:-/home/orin1/prima_artifacts}"
CLIP_ROOT="${CLIP_ROOT:-${PRIMA_ARTIFACT_ROOT}/clip}"
SCRIPT_PATH="${SCRIPT_DIR}/infer_clip_rn50.py"
SCRIPT="infer_clip_rn50.py"
DOCKER_IMAGE="${DOCKER_IMAGE:-clip-full:jetson}"
CONTAINER_NAME="${WORKLOAD_NAME}_$(date +%Y%m%d_%H%M%S)"

cd "${CLIP_ROOT}"
RESULT_DIR="results/${WORKLOAD_NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RESULT_DIR}"

OUTPUT_LOG="${RESULT_DIR}/output.txt"
TIME_LOG="${RESULT_DIR}/time.txt"
SUMMARY_LOG="${RESULT_DIR}/summary.txt"
DOCKER_STATS_LOG="${RESULT_DIR}/docker_stats.txt"

MPSTAT_PID=""
VMSTAT_PID=""
PIDSTAT_PID=""
TEGRA_PID=""
DOCKER_STATS_PID=""

cleanup() {
    echo "[INFO] Cleaning up monitors..."

    if [ -n "$MPSTAT_PID" ]; then kill "$MPSTAT_PID" 2>/dev/null || true; fi
    if [ -n "$VMSTAT_PID" ]; then kill "$VMSTAT_PID" 2>/dev/null || true; fi
    if [ -n "$PIDSTAT_PID" ]; then kill "$PIDSTAT_PID" 2>/dev/null || true; fi
    if [ -n "$TEGRA_PID" ]; then kill "$TEGRA_PID" 2>/dev/null || true; fi
    if [ -n "$DOCKER_STATS_PID" ]; then kill "$DOCKER_STATS_PID" 2>/dev/null || true; fi

    wait "$MPSTAT_PID" 2>/dev/null || true
    wait "$VMSTAT_PID" 2>/dev/null || true
    wait "$PIDSTAT_PID" 2>/dev/null || true
    wait "$TEGRA_PID" 2>/dev/null || true
    wait "$DOCKER_STATS_PID" 2>/dev/null || true

    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
}

trap cleanup EXIT INT TERM

echo "[INFO] RESULT_DIR=${RESULT_DIR}"
echo "[INFO] SCRIPT=${SCRIPT}"
echo "[INFO] DOCKER_IMAGE=${DOCKER_IMAGE}"
echo "[INFO] CONTAINER_NAME=${CONTAINER_NAME}"

{
    echo "===== DATE ====="
    date

    echo
    echo "===== HOST ====="
    hostname

    echo
    echo "===== WORKDIR ====="
    pwd

    echo
    echo "===== DOCKER ====="
    docker --version
    docker images | grep -E "clip-full|ultralytics" || true

    echo
    echo "===== DOCKER IMAGE INSPECT ====="
    docker image inspect "${DOCKER_IMAGE}" --format 'Id={{.Id}} Size={{.Size}} Created={{.Created}}' || true

    echo
    echo "===== SCRIPT ====="
    echo "${SCRIPT}"

    echo
    echo "===== FILES ====="
    ls -lh

    echo
    echo "===== MONITOR TOOLS ====="
    which mpstat || true
    which vmstat || true
    which pidstat || true
    which tegrastats || true
    which docker || true
} > "${SUMMARY_LOG}"

echo "[INFO] Starting monitors..."

mpstat -P ALL 1 > "${RESULT_DIR}/system_mpstat.txt" 2>&1 &
MPSTAT_PID=$!

vmstat 1 -t > "${RESULT_DIR}/system_vmstat.txt" 2>&1 &
VMSTAT_PID=$!

pidstat -h -r -u -I 1 > "${RESULT_DIR}/system_pidstat.txt" 2>&1 &
PIDSTAT_PID=$!

tegrastats --interval 1000 --logfile "${RESULT_DIR}/system_tegrastats.txt" &
TEGRA_PID=$!

sleep 2

echo "[INFO] Starting Docker container..."

START_NS=$(date +%s%N)

/usr/bin/time -v docker run \
    --name "${CONTAINER_NAME}" \
    --runtime=nvidia \
    --ipc=host \
    -v "$(pwd)":/workspace \
    -v "${SCRIPT_PATH}":/workspace/"${SCRIPT}":ro \
    -w /workspace \
    "${DOCKER_IMAGE}" \
    python3 "${SCRIPT}" > "${OUTPUT_LOG}" 2> "${TIME_LOG}" &

APP_PID=$!

sleep 1

echo "[INFO] Starting docker stats..."

docker stats "${CONTAINER_NAME}" \
    --format "table {{.Container}}\t{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.NetIO}}\t{{.BlockIO}}\t{{.PIDs}}" \
    > "${DOCKER_STATS_LOG}" 2>&1 &
DOCKER_STATS_PID=$!

wait "${APP_PID}"
APP_STATUS=$?

END_NS=$(date +%s%N)
ELAPSED_MS=$(( (END_NS - START_NS) / 1000000 ))

cleanup
trap - EXIT INT TERM

{
    echo
    echo "===== RUN SUMMARY ====="
    echo "workload=${WORKLOAD_NAME}"
    echo "script=${SCRIPT}"
    echo "docker_image=${DOCKER_IMAGE}"
    echo "container_name=${CONTAINER_NAME}"
    echo "exit_status=${APP_STATUS}"
    echo "elapsed_ms=${ELAPSED_MS}"
    echo "elapsed_sec=$(awk "BEGIN {printf \"%.3f\", ${ELAPSED_MS}/1000}")"

    echo
    echo "===== WORKLOAD OUTPUT ====="
    cat "${OUTPUT_LOG}"

    echo
    echo
    echo "===== /usr/bin/time -v ====="
    cat "${TIME_LOG}"

    echo
    echo
    echo "===== DOCKER STATS ====="
    cat "${DOCKER_STATS_LOG}"

    echo
    echo
    echo "===== LOG FILES ====="
    echo "mpstat=${RESULT_DIR}/system_mpstat.txt"
    echo "vmstat=${RESULT_DIR}/system_vmstat.txt"
    echo "pidstat=${RESULT_DIR}/system_pidstat.txt"
    echo "tegrastats=${RESULT_DIR}/system_tegrastats.txt"
    echo "docker_stats=${DOCKER_STATS_LOG}"
    echo "output=${OUTPUT_LOG}"
    echo "time=${TIME_LOG}"
} >> "${SUMMARY_LOG}"

echo "[DONE] Profiling finished."
echo "[DONE] RESULT_DIR=${RESULT_DIR}"
echo "[DONE] exit_status=${APP_STATUS}"
echo "[DONE] elapsed_ms=${ELAPSED_MS}"
echo
echo "===== WORKLOAD OUTPUT ====="
cat "${OUTPUT_LOG}"
echo
echo "===== SUMMARY ====="
cat "${SUMMARY_LOG}"

exit "${APP_STATUS}"
