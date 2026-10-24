"""Run selected editor-input workflows from the Mac and retain resumable evidence."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import time

spec=importlib.util.spec_from_file_location('client',Path(__file__).with_name('client.py'))
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--matrix',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--base-url',default='http://127.0.0.1:18081')
    ap.add_argument('--tokens',type=Path,nargs='+',required=True);ap.add_argument('--names',nargs='+')
    ap.add_argument('--parallel',type=int,default=1);ap.add_argument('--resume',action='store_true')
    ap.add_argument('--allow-testing-profile',action='store_true');args=ap.parse_args()
    matrix=json.loads(args.matrix.read_text());matrix=[m for m in matrix if not args.names or m['name'] in args.names]
    args.output.mkdir(mode=0o700,parents=True,exist_ok=True)
    tokens=[p.read_text().strip() for p in args.tokens]
    def run(pair):
        i,case=pair;client=c.Client(args.base_url,tokens[i%len(tokens)])
        out=args.output/case['name']
        try:
            report=c.run_job(client,case['fixture'],case['profile'],out,args.resume,timeout=1800,
                purpose='preview' if case.get('output_range') else 'full',frame_range=case.get('output_range'),
                route_label='Mac via authenticated SSH/ProxyJump; editor-input correctness qualification',
                guidance_path=case['guidance'],allow_testing_profile=args.allow_testing_profile)
            result=dict(name=case['name'],job_id=json.loads((out/'state.json').read_text())['job_id'],
                        received_files=report['result_files'],server_timings=report['server_timings'])
            c.save(out/'case.json',result)
            print(json.dumps(dict(name=case['name'],passed_transport=True,received_files=report['result_files'])),flush=True)
            return result
        finally:client.close()
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:rows=list(pool.map(run,enumerate(matrix)))
    c.save(args.output/'jobs.json',rows)


if __name__=='__main__':main()
