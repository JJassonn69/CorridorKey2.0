"""Mac end-to-end worker comparison; each lane represents a separate user."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import statistics
import time

spec=importlib.util.spec_from_file_location('client',Path(__file__).with_name('client.py'))
client=importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)


def verify_cache(run, references):
    local=json.loads((run/'cache/manifest.json').read_text())
    actual={(f['source_frame'],f['output']):f['sha256'] for f in local['result']['files']}
    expected={}
    for f in references['rendered']:
        p=Path(f['file'])
        name=p.stem
        output,index=name.rsplit('_',1)
        expected[(int(index),output)]=f['sha256']
    selected=local['result']['recipe']['output_range']
    expected={k:v for k,v in expected.items() if selected['start']<=k[0]<selected['end']}
    intact=all(client.digest(f['path'])==f['sha256'] for f in local['local_files'])
    if actual!=expected or not intact:
        raise RuntimeError('Downloaded result differs from direct inference')
    return dict(pass_exact=True,files=len(actual),cached_files_rehashed=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base-url',default='http://127.0.0.1:18080')
    ap.add_argument('--tokens',type=Path,nargs=2,required=True)
    ap.add_argument('--fixture-root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--references',type=Path)
    ap.add_argument('--repetitions',type=int,default=3)
    ap.add_argument('--workers',type=int,choices=(1,2),required=True)
    ap.add_argument('--clips',nargs=2,default=['escape-s001','saber-s009'])
    args=ap.parse_args()
    args.output.mkdir(mode=0o700,parents=True,exist_ok=False)
    tokens=[p.read_text().strip() for p in args.tokens]
    probe=client.Client(args.base_url,tokens[0])
    try:
        caps,_=probe.json('capabilities','GET','/v1/capabilities')
        if caps['limits']['max_active_gpu_jobs']!=args.workers:
            raise RuntimeError('Server worker count differs from the requested experiment')
        deadline=time.monotonic()+180
        while caps.get('worker_pool',{}).get('ready',0)!=args.workers:
            if time.monotonic()>deadline:raise TimeoutError('Workers did not become idle/ready')
            time.sleep(.5)
            caps,_=probe.json('capabilities','GET','/v1/capabilities')
    finally:
        probe.close()
    barrier=__import__('threading').Barrier(2)
    def lane(index):
        c=client.Client(args.base_url,tokens[index])
        rows=[]
        try:
            barrier.wait()
            for repetition in range(args.repetitions):
                name=args.clips[index]
                out=args.output/f'lane-{index}-run-{repetition}'
                report=client.run_job(c,args.fixture_root/name/'fixture.json','native-720p-24x12',out,
                    route_label='Mac through loopback SSH/ProxyJump; shared GPU server')
                if args.references:
                    reference=json.loads((args.references/(name+'.json')).read_text())
                    client.save(out/'exactness.json',verify_cache(out,reference))
                rows.append(dict(fixture=name,end_to_end_seconds=report['end_to_end_seconds'],
                                 server_timings=report['server_timings'],state=json.loads((out/'state.json').read_text())))
                # Preserve the first full local result; other repetitions retain their
                # verified manifest/checksums. Release remote quota only after the gate.
                if args.references:
                    state=rows[-1]['state']
                    c.json('asset_delete','DELETE','/v1/assets/'+state['asset_id'],expect=(202,))
                    if repetition:
                        for f in (out/'cache').glob('*.exr'):
                            f.unlink()
            return rows,c.records
        finally:
            c.close()
    started=time.perf_counter()
    def busy_probe():
        c=client.Client(args.base_url,tokens[0])
        try:
            deadline=time.monotonic()+180
            while True:
                caps,_=c.json('readiness','GET','/v1/capabilities')
                if caps['worker_pool']['busy']>0:break
                if time.monotonic()>deadline:raise TimeoutError('No running job for busy probe')
                time.sleep(.5)
            c.records.clear()
            for _ in range(50):c.json('busy_capabilities','GET','/v1/capabilities')
            client.save(args.output/'busy-probe.json',dict(requests=c.records,summary=client.summaries(c.records)))
        finally:c.close()
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(lane,i) for i in range(2)]
        busy=pool.submit(busy_probe)
        results=[f.result() for f in futures]
        busy.result()
    seconds=time.perf_counter()-started
    rows=[j for lane_rows,_ in results for j in lane_rows]
    request_rows=[r for _,r in results for r in r]
    summary=dict(workers=args.workers,client_lanes=2,jobs=len(rows),frames=72*len(rows),
        capture_raw=caps.get('worker_pool',{}).get('capture_raw'),
        execution_fingerprint=caps['profiles'][0]['execution_fingerprint'],
        wall_seconds=seconds,end_to_end_aggregate_fps=72*len(rows)/seconds,
        job_median_seconds=statistics.median(j['end_to_end_seconds'] for j in rows),
        job_p95_seconds=None,request_summary=client.summaries(request_rows),jobs_detail=rows,
        scope='Full Mac workflow with fresh uploads and verified local outputs; shared server load; no WAN claim')
    client.save(args.output/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('request_summary','jobs_detail')},indent=2))


if __name__=='__main__':main()
