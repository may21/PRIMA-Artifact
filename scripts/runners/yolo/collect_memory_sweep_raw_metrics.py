#!/usr/bin/env python3
import csv, re, sys
from pathlib import Path

WL={"classify":"CLS","detect":"DET","pose":"EST","segment":"SEG","obb":"OBB"}
MEMGB={"0.5GB":0.5,"1.0GB":1.0,"1.5GB":1.5,"2.0GB":2.0,"2.5GB":2.5}
TEGRA=re.compile(r"^(?:(?P<date>\d{2}-\d{2}-\d{4})\s+(?P<time>\d{2}:\d{2}:\d{2})\s+)?RAM\s+(?P<ram>\d+)/(?P<total>\d+)MB.*?GR3D_FREQ\s+(?P<gpu>\d+)%.*?gpu@(?P<temp>\d+(?:\.\d+)?)C.*?VDD_IN\s+(?P<power>\d+)mW/(?P<power_avg>\d+)mW")
TOP_MEM=re.compile(r"^MiB Mem\s*:\s*(?P<total>[0-9.]+)\s+total,\s*(?P<free>[0-9.]+)\s+free,\s*(?P<used>[0-9.]+)\s+used,\s*(?P<buff>[0-9.]+)\s+buff/cache")
TOP_SWAP=re.compile(r"^MiB Swap\s*:\s*(?P<total>[0-9.]+)\s+total,\s*(?P<free>[0-9.]+)\s+free,\s*(?P<used>[0-9.]+)\s+used\.\s*(?P<avail>[0-9.]+)\s+avail Mem")

def write(path, fields, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

def num(x):
    x=str(x).strip()
    try: return float(x) if '.' in x else int(x)
    except Exception: return ''

def main():
    if len(sys.argv)!=2: raise SystemExit(f'usage: {Path(sys.argv[0]).name} RESULT_ROOT')
    root=Path(sys.argv[1]).resolve(); out=root/'organized'; out.mkdir(exist_ok=True)
    summaries=list(csv.DictReader((root/'summary.csv').open()))
    tegra=[]; top=[]; cgroup=[]; perf=[]
    for s in summaries:
        w=s['workload']; label=s['memory_label']; rep=int(s['repeat']); run=Path(s['result_dir'])
        if not run.is_absolute(): run=root/run
        common={'workload':WL.get(w,w),'process_name':w,'memory_label':label,'memory_limit_gb':MEMGB.get(label,''),'repeat':rep}
        p=run/'system_tegrastats.txt'
        if p.exists():
            for i,line in enumerate(p.read_text(errors='replace').splitlines(),1):
                m=TEGRA.search(line)
                if m:
                    tegra.append({**common,'sample_index':i,'date':m.group('date') or '', 'time':m.group('time') or '', 'ram_used_mb':int(m.group('ram')), 'ram_total_mb':int(m.group('total')), 'gpu_usage_pct':int(m.group('gpu')), 'gpu_temp_c':float(m.group('temp')), 'vdd_in_now_mw':int(m.group('power')), 'vdd_in_avg_mw':int(m.group('power_avg')), 'raw_line':line})
        p=run/'system_top.txt'
        if p.exists():
            sample=0; pending=None
            for line in p.read_text(errors='replace').splitlines():
                m=TOP_MEM.search(line.strip())
                if m:
                    sample+=1; pending={**common,'sample_index':sample,'mem_total_mib':float(m.group('total')),'mem_free_mib':float(m.group('free')),'mem_used_mib':float(m.group('used')),'mem_buff_cache_mib':float(m.group('buff')),'swap_total_mib':'','swap_free_mib':'','swap_used_mib':'','mem_available_mib':''}
                    top.append(pending); continue
                m=TOP_SWAP.search(line.strip())
                if m and pending is not None:
                    pending.update({'swap_total_mib':float(m.group('total')),'swap_free_mib':float(m.group('free')),'swap_used_mib':float(m.group('used')),'mem_available_mib':float(m.group('avail'))})
        p=run/'cgroup_memory.csv'
        if p.exists():
            for i,row in enumerate(csv.DictReader(p.open()),1):
                cgroup.append({**common,'sample_index':i,'epoch_ns':row.get('epoch_ns',''),'memory_current_bytes':row.get('memory_current_bytes',''),'memory_peak_bytes':row.get('memory_peak_bytes',''),'memory_swap_current_bytes':row.get('memory_swap_current_bytes','')})
        p=run/'perf.csv'
        if p.exists():
            for ln,line in enumerate(p.read_text(errors='replace').splitlines(),1):
                parts=[x.strip() for x in line.split(',')]
                if len(parts)>=4 and parts[1].replace('.','',1).isdigit():
                    perf.append({**common,'line_number':ln,'interval_seconds':parts[0],'value':parts[1],'event':parts[3],'runtime_ns':parts[4] if len(parts)>4 else '', 'running_pct':parts[5] if len(parts)>5 else '', 'raw_line':line})
    if tegra: write(out/'tegrastats_ram_gpu_raw.csv', list(tegra[0]), tegra)
    if top: write(out/'top_memory_raw.csv', list(top[0]), top)
    if cgroup: write(out/'cgroup_memory_raw.csv', list(cgroup[0]), cgroup)
    if perf: write(out/'perf_events_raw.csv', list(perf[0]), perf)
    with (out/'raw_validation.txt').open('w') as f:
        for k,v in {'summary_runs':len(summaries),'tegrastats_rows':len(tegra),'top_memory_rows':len(top),'cgroup_rows':len(cgroup),'perf_rows':len(perf)}.items():
            print(f'{k}={v}'); f.write(f'{k}={v}\n')
if __name__=='__main__': main()
