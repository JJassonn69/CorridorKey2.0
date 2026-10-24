"""Produce a media-free summary from private Mac reports and server samples."""
import argparse
import json
from pathlib import Path
import statistics


def load(path):return json.loads(path.read_text())


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    samples=[json.loads(line) for line in (args.root/'timing-monitor.jsonl').read_text().splitlines()]
    events=[json.loads(line) for line in (args.root/'events.jsonl').read_text().splitlines()]
    rows=[]
    for name in ['timing-one','timing-two']:
        summary=load(args.root/name/'summary.json')
        times=[j['server_timings'] for j in summary['jobs_detail']]
        begin=min(t['service_start_monotonic'] for t in times)
        end=max(t['service_end_monotonic'] for t in times)
        startup=max(e['monotonic'] for e in events if e['event']=='startup' and e['monotonic']<begin)
        selected=[s for s in samples if startup<=s['monotonic']<=end]
        active=[s for s in selected if begin<=s['monotonic']<=end]
        cpu=[]
        for a,b in zip(active,active[1:]):
            ta,tb=a['cpu_ticks'][:8],b['cpu_ticks'][:8]
            total=sum(tb)-sum(ta)
            idle=(tb[3]+tb[4])-(ta[3]+ta[4])
            if total>0:cpu.append(100*(1-idle/total))
        exact=[load(p) for p in (args.root/name).glob('*/exactness.json')]
        result=dict(workers=summary['workers'],jobs=summary['jobs'],frames=summary['frames'],
            wall_seconds=summary['wall_seconds'],end_to_end_fps=summary['end_to_end_aggregate_fps'],
            job_median_seconds=summary['job_median_seconds'],job_p95_seconds=None,
            service_median_seconds=statistics.median(t['service_seconds'] for t in times),
            queue_median_seconds=statistics.median(t['queue_seconds'] for t in times),
            worker_inference_median_seconds=statistics.median(t['worker_inference_seconds'] for t in times),
            server_service_span_fps=summary['frames']/(end-begin),
            inference_span_fps=summary['frames']/(max(t['inference_end_monotonic'] for t in times)-min(t['inference_start_monotonic'] for t in times)),
            peak_device_memory_mib_including_load=max(int(s['gpu']['memory_mib']) for s in selected),
            peak_device_memory_mib_active=max(int(s['gpu']['memory_mib']) for s in active),
            peak_worker_allocated_gib=max(t['worker_peak_allocated_bytes'] for t in times)/2**30,
            peak_worker_reserved_gib=max(t['worker_peak_reserved_bytes'] for t in times)/2**30,
            temperature_range_c=[min(int(s['gpu']['temperature_c']) for s in active),max(int(s['gpu']['temperature_c']) for s in active)],
            load_average_range=[min(s['load_average'][0] for s in active),max(s['load_average'][0] for s in active)],
            host_cpu_busy_median_percent=statistics.median(cpu),
            thermal_slowdown_samples=sum(s['gpu']['sw_thermal_slowdown']=='Active' or s['gpu']['hw_thermal_slowdown']=='Active' for s in active),
            active_samples=len(active),sampling_interval_seconds=1,
            exact_exr_files=sum(e['files'] for e in exact),exact_passed=len(exact)==summary['jobs'] and all(e['pass_exact'] for e in exact),
            busy_probe=load(args.root/name/'busy-probe.json')['summary'],request_summary=summary['request_summary'])
        stages=[load(p)['stage_seconds'] for p in (args.root/name).glob('*/report.json')]
        result['median_stage_seconds']={k:statistics.median(s[k] for s in stages) for k in stages[0]}
        rows.append(result)
    qualification=[load(p) for pattern in ['exact-one-*.json','exact-two-*.json'] for p in args.root.glob(pattern)]
    if len(qualification)!=6 or not all(q['pass_exact'] for q in qualification):
        raise RuntimeError('Incomplete or failed raw qualification matrix')
    accepted=load(args.root.parent/'speed3/acceptance.json')['source_files']
    reference=load(args.root/'references/escape-s001.json')['source_files']
    out=dict(date='2026-10-24',route='Mac over authenticated SSH/ProxyJump on local network',
        controlled_capacity_benchmark=False,conditions='CPU oversubscription and observed GPU thermal slowdown',
        runtime_sources_match_accepted_suite=accepted==reference,
        qualification_raw_frames=sum(q['raw_frames'] for q in qualification),
        qualification_exr_files=sum(q['exr_files'] for q in qualification),
        rows=rows,throughput_change_percent=100*(rows[1]['end_to_end_fps']/rows[0]['end_to_end_fps']-1),
        median_latency_change_percent=100*(rows[1]['job_median_seconds']/rows[0]['job_median_seconds']-1))
    args.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
