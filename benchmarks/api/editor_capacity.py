"""Summarize device samples inside completed API service intervals only."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from service.common import save


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--monitor',type=Path,required=True)
    ap.add_argument('--jobs',type=Path,nargs='+',required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();samples=[json.loads(s) for s in args.monitor.read_text().splitlines()]
    cases=[]
    for path in args.jobs:
        rows=json.loads(path.read_text());timings=[r['server_timings'] for r in rows]
        selected=[s for s in samples if any(t['service_start_monotonic']<=s['monotonic']<=t['service_end_monotonic'] for t in timings)]
        assert selected,'No device samples cover the service interval'
        gpu=[s['gpu'] for s in selected]
        cases.append(dict(name=path.parent.name,samples=len(selected),
            sampled_peak_device_mib=max(int(s['memory_mib']) for s in gpu),
            max_temperature_c=max(int(s['temperature_c']) for s in gpu),
            thermal_slowdown_samples=sum(s['sw_thermal_slowdown']=='Active' or s['hw_thermal_slowdown']=='Active' for s in gpu),
            load_average_1m_range=[min(s['load_average'][0] for s in selected),max(s['load_average'][0] for s in selected)],
            worker_peak_allocated_bytes=max(t['worker_peak_allocated_bytes'] for t in timings),
            worker_peak_reserved_bytes=max(t['worker_peak_reserved_bytes'] for t in timings)))
    result=dict(sample_interval_seconds=1,scope='API service only; includes idle resident worker and desktop GPU use; excludes independent direct runs',cases=cases)
    save(args.output,result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
