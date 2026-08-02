#!/usr/bin/env bash
set -euo pipefail
PRIMA_ARTIFACT_ROOT="${PRIMA_ARTIFACT_ROOT:-/home/orin1/prima_artifacts}"
BASE="${EXPERIMENT_ROOT:-${PRIMA_ARTIFACT_ROOT}/yolo}"
cd "$BASE"
IMAGE="${IMAGE:-ultralytics/ultralytics:8.3.102-jetson-jetpack6}"
HOST_YOLO_DIR="${HOST_YOLO_DIR:-$BASE/workload}"
DATASET_ROOT="${DATASET_ROOT:-${PRIMA_ARTIFACT_ROOT}/yolo_runtime}"
CONTAINER_YOLO_DIR="/usr/src/ultralytics/yolo_new"
IMAGE_LIMIT="${IMAGE_LIMIT:-100}"
REPEAT_START="${REPEAT_START:-1}"
REPEAT_END="${REPEAT_END:-3}"
RESULT_ROOT="${RESULT_ROOT_OVERRIDE:-$BASE/results/isolated_no_memlimit_100_1}"
if [[ -z "${RESULT_ROOT_OVERRIDE:-}" ]]; then
  idx=1
  while [[ -e "$BASE/results/isolated_no_memlimit_100_${idx}" ]]; do idx=$((idx+1)); done
  RESULT_ROOT="$BASE/results/isolated_no_memlimit_100_${idx}"
fi
mkdir -p "$RESULT_ROOT"
SUMMARY="$RESULT_ROOT/summary.csv"
echo "mode,workload,repeat,container,exit_status,oom_killed,result_dir" > "$SUMMARY"
{
  echo "experiment=isolated_no_memlimit"
  echo "image_limit=$IMAGE_LIMIT"
  echo "repeat_range=${REPEAT_START}-${REPEAT_END}"
  echo "memory_limit=unlimited"
  echo "image=$IMAGE"
  nvpmodel -q 2>/dev/null | tr '\n' ' '; echo
} > "$RESULT_ROOT/experiment_config.txt"
cat /proc/swaps > "$RESULT_ROOT/swaps_start.txt"
nvpmodel -q > "$RESULT_ROOT/nvpmodel_start.txt" 2>&1 || true
if [[ -s /proc/swaps ]] && [[ "$(wc -l < /proc/swaps)" -gt 1 ]]; then
  echo "[ERROR] host swap is active" >&2
  cat /proc/swaps >&2
  exit 1
fi

declare -A SCRIPTS=([classify]=classify.py [detect]=detect.py [pose]=pose.py [segment]=segment.py [obb]=obb.py)
WORKLOADS=(classify detect pose segment obb)
MONITOR_PIDS=()
stop_monitors(){
  for p in "${MONITOR_PIDS[@]:-}"; do kill -INT "$p" >/dev/null 2>&1 || true; done
  sleep 0.2
  for p in "${MONITOR_PIDS[@]:-}"; do kill -TERM "$p" >/dev/null 2>&1 || true; done
  MONITOR_PIDS=()
}
cleanup_container(){ docker rm -f "$1" >/dev/null 2>&1 || true; }
trap 'stop_monitors; exit 130' INT TERM
for repeat in $(seq "$REPEAT_START" "$REPEAT_END"); do
  for workload in "${WORKLOADS[@]}"; do
    RUN_DIR="$RESULT_ROOT/$workload/unlimited/run${repeat}"
    mkdir -p "$RUN_DIR/runs3"
    cname="isolated-nomem-${workload}-r${repeat}"
    cleanup_container "$cname"
    echo "============================================================"
    echo "[RUN] workload=$workload repeat=$repeat result_dir=$RUN_DIR"
    echo "============================================================"
    cat > "$RUN_DIR/config.txt" <<CFG
mode=isolated_no_memlimit
workload=$workload
repeat=$repeat
memory_limit=unlimited
image_limit=$IMAGE_LIMIT
container=$cname
CFG
    docker create \
      --name "$cname" \
      --runtime=nvidia \
      --ipc=host \
      --user "$(id -u):$(id -g)" \
      --pull=never \
      -e CUDA_VISIBLE_DEVICES=0 \
      -e HOME=/tmp \
      -e MPLCONFIGDIR=/tmp/matplotlib \
      -e YOLO_CONFIG_DIR=/tmp/ultralytics \
      -e IMAGE_LIMIT="$IMAGE_LIMIT" \
      -v "$HOST_YOLO_DIR:$CONTAINER_YOLO_DIR" \
      -v "$DATASET_ROOT/expanded_imagenet_images:$CONTAINER_YOLO_DIR/expanded_imagenet_images:ro" \
      -v "$DATASET_ROOT/expanded_coco_images:$CONTAINER_YOLO_DIR/expanded_coco_images:ro" \
      -v "$DATASET_ROOT/expanded_dota_images:$CONTAINER_YOLO_DIR/expanded_dota_images:ro" \
      -v "$RUN_DIR/runs3:$CONTAINER_YOLO_DIR/runs3" \
      "$IMAGE" bash -lc "cd '$CONTAINER_YOLO_DIR' && exec python3 '${SCRIPTS[$workload]}'" >/dev/null
    docker inspect "$cname" > "$RUN_DIR/docker_inspect_before.json" 2>/dev/null || true
    sleep 5
    echo "event,epoch_ns" > "$RUN_DIR/time.csv"
    echo "start,$(date +%s%N)" >> "$RUN_DIR/time.csv"
    pidstat -h -r -u -I -t 1 > "$RUN_DIR/system_pidstat.txt" 2>&1 & MONITOR_PIDS+=("$!")
    mpstat -P ALL 1 > "$RUN_DIR/system_mpstat.txt" 2>&1 & MONITOR_PIDS+=("$!")
    vmstat 1 -t > "$RUN_DIR/system_vmstat.txt" 2>&1 & MONITOR_PIDS+=("$!")
    top -b -d 1 > "$RUN_DIR/system_top.txt" 2>&1 & MONITOR_PIDS+=("$!")
    tegrastats --interval 1000 --logfile "$RUN_DIR/system_tegrastats.txt" & MONITOR_PIDS+=("$!")
    docker start "$cname" >/dev/null
    host_pid=""
    for i in $(seq 1 15); do
      host_pid="$(docker top "$cname" -eo pid,comm,args 2>/dev/null | awk '$2 ~ /^python(3)?$/ {print $1; exit}')"
      [[ -n "$host_pid" ]] && break
      sleep 1
    done
    if [[ -n "$host_pid" ]]; then
      echo "$host_pid" > "$RUN_DIR/python_pid.txt"
      cgroup_rel="$(awk -F: '$1=="0"{print $3}' "/proc/$host_pid/cgroup" 2>/dev/null || true)"
      if [[ -n "$cgroup_rel" && -d "/sys/fs/cgroup${cgroup_rel}" ]]; then
        CGROUP_DIR="/sys/fs/cgroup${cgroup_rel}"
        echo "$CGROUP_DIR" > "$RUN_DIR/cgroup_path.txt"
        echo "epoch_ns,memory_current_bytes,memory_peak_bytes,memory_swap_current_bytes" > "$RUN_DIR/cgroup_memory.csv"
        (
          while kill -0 "$host_pid" 2>/dev/null; do
            echo "$(date +%s%N),$(cat "$CGROUP_DIR/memory.current" 2>/dev/null || echo),$(cat "$CGROUP_DIR/memory.peak" 2>/dev/null || echo),$(cat "$CGROUP_DIR/memory.swap.current" 2>/dev/null || echo)"
            sleep 0.5
          done
        ) >> "$RUN_DIR/cgroup_memory.csv" & MONITOR_PIDS+=("$!")
      fi
    else
      echo "[WARN] python pid not found" | tee "$RUN_DIR/python_pid_missing.txt"
    fi
    exit_status=0
    docker wait "$cname" > "$RUN_DIR/docker_wait.txt" || exit_status=$?
    if [[ -f "$RUN_DIR/docker_wait.txt" ]]; then exit_status="$(cat "$RUN_DIR/docker_wait.txt" | tail -1)"; fi
    echo "end,$(date +%s%N)" >> "$RUN_DIR/time.csv"
    stop_monitors
    docker inspect "$cname" > "$RUN_DIR/docker_inspect_after.json" 2>/dev/null || true
    oom="$(docker inspect -f '{{.State.OOMKilled}}' "$cname" 2>/dev/null || echo false)"
    docker logs "$cname" > "$RUN_DIR/output.txt" 2>&1 || true
    echo "isolated_no_memlimit,$workload,$repeat,$cname,$exit_status,$oom,$RUN_DIR" >> "$SUMMARY"
    cleanup_container "$cname"
    echo "[DONE] workload=$workload repeat=$repeat exit=$exit_status oom=$oom"
  done
done
RESULT_ROOT_PATH="$RESULT_ROOT" python3 - <<'REMOTEPY'
import csv, pathlib, re, collections, os
root=pathlib.Path(os.environ['RESULT_ROOT_PATH'])
out=root/'organized'; out.mkdir(exist_ok=True)
workload_map={'classify':'CLS','detect':'DET','pose':'EST','segment':'SEG','obb':'OBB'}
rows=[]; parse=[]
pat=re.compile(r'^\d{2}:\d{2}:\d{2}$')
def parse_line(line):
    p=line.strip().split()
    if len(p)<16 or not pat.match(p[0]): return None
    return {'timestamp':p[0],'uid':p[1],'tgid':p[2],'tid':p[3],'cpu_user_pct':p[4],'cpu_system_pct':p[5],'cpu_guest_pct':p[6],'cpu_wait_pct':p[7],'cpu_total_pct':p[8],'cpu_core':p[9],'minor_faults_per_sec':p[10],'major_faults_per_sec':p[11],'vsz_kb':p[12],'rss_kb':p[13],'memory_pct':p[14],'command':' '.join(p[15:])}
for pidfile in sorted(root.glob('*/*/run*/python_pid.txt')):
    run_dir=pidfile.parent
    workload_name=run_dir.parts[-3]
    repeat=re.sub(r'\D','',run_dir.name) or run_dir.name
    wl=workload_map.get(workload_name, workload_name.upper())
    pid=pidfile.read_text().strip().split()[0]
    samples=0
    pidstat=run_dir/'system_pidstat.txt'
    for line in pidstat.open(errors='ignore'):
        d=parse_line(line)
        if not d or not (d['tgid']==pid and d['tid']=='-'): continue
        samples+=1
        pct=float(d['memory_pct'] or 0); rss=float(d['rss_kb'] or 0)
        rows.append({'prima_setting':'ISOLATED','mode':'isolated_no_memlimit','repeat':repeat,'sample_index':samples,'timestamp':d['timestamp'],'workload':wl,'workload_name':workload_name,'python_pid':pid,'tgid':d['tgid'],'tid':d['tid'],'command':d['command'],'memory_pct':d['memory_pct'],'pidstat_memory_gb_8gb':f'{pct*8/100:.9f}','rss_kb':d['rss_kb'],'rss_gb':f'{rss/1024/1024:.9f}','vsz_kb':d['vsz_kb'],'minor_faults_per_sec':d['minor_faults_per_sec'],'major_faults_per_sec':d['major_faults_per_sec'],'cpu_total_pct':d['cpu_total_pct'],'source_file':str(pidstat)})
    parse.append({'workload':wl,'workload_name':workload_name,'repeat':repeat,'python_pid':pid,'pidstat_process_rows':samples,'source_file':str(pidstat)})
fields=['prima_setting','mode','repeat','sample_index','timestamp','workload','pidstat_memory_gb_8gb','memory_pct','rss_gb','rss_kb','workload_name','python_pid','tgid','tid','command','vsz_kb','minor_faults_per_sec','major_faults_per_sec','cpu_total_pct','source_file']
with (out/'isolated_no_memlimit_pidstat_memory_gb_long.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fields,extrasaction='ignore'); w.writeheader(); w.writerows(rows)
g=collections.defaultdict(dict)
for r in rows:
    k=(r['prima_setting'],r['mode'],r['repeat'],r['sample_index'])
    rec=g[k]; rec.update({'prima_setting':r['prima_setting'],'mode':r['mode'],'repeat':r['repeat'],'sample_index':r['sample_index']})
    wl=r['workload']; rec[f'{wl}_timestamp']=r['timestamp']; rec[f'{wl}_pidstat_memory_gb']=r['pidstat_memory_gb_8gb']
wide=list(g.values()); wide.sort(key=lambda r:(int(r['repeat']), int(r['sample_index'])))
wfields=['prima_setting','mode','repeat','sample_index']
for wl in ['CLS','DET','EST','SEG','OBB']: wfields += [f'{wl}_timestamp',f'{wl}_pidstat_memory_gb']
with (out/'isolated_no_memlimit_pidstat_memory_gb_wide.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,wfields,extrasaction='ignore'); w.writeheader(); w.writerows(wide)
by=collections.defaultdict(list)
for r in rows: by[(r['workload'],r['repeat'])].append(r)
peaks=[]
for (wl,rep),rs in sorted(by.items(), key=lambda kv:(int(kv[0][1]), kv[0][0])):
    vals=[float(r['pidstat_memory_gb_8gb']) for r in rs]
    peaks.append({'prima_setting':'ISOLATED','mode':'isolated_no_memlimit','repeat':rep,'workload':wl,'samples':len(rs),'max_pidstat_memory_gb_8gb':f'{max(vals):.9f}','avg_pidstat_memory_gb_8gb':f'{sum(vals)/len(vals):.9f}'})
with (out/'isolated_no_memlimit_pidstat_memory_peak_summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,list(peaks[0].keys()) if peaks else ['mode']); w.writeheader(); w.writerows(peaks)
with (out/'isolated_no_memlimit_pidstat_parse_summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,list(parse[0].keys()) if parse else ['mode']); w.writeheader(); w.writerows(parse)
(out/'validation.txt').write_text('\n'.join([f'root={root}',f'runs={len(parse)}',f'long_rows={len(rows)}',f'wide_rows={len(wide)}',f'peak_rows={len(peaks)}','conversion: pidstat_memory_gb_8gb = memory_pct * 8 / 100'])+'\n')
REMOTEPY
