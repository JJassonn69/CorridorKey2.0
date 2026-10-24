"""Compare private worker captures with the unchanged direct-inference harness."""
import argparse
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from service.common import digest, save


def compare(reference, worker):
    a=json.loads((reference/'manifest.json').read_text())
    b=json.loads(worker.read_text())
    raw_a={f['index']:f['sha256'] for f in a['frames']}
    raw_b={f['index']:f.get('sha256') for f in b['raw']}
    exr_a={Path(f['file']).name:f['sha256'] for f in a['rendered']}
    exr_b={Path(f['path']).name:f['sha256'] for f in b['files']}
    # Verify reference files still have the recorded digests before trusting them.
    integrity=all(digest(reference/f['file'])==f['sha256'] for f in a['frames']+a['rendered'])
    return dict(pass_exact=integrity and raw_a==raw_b and exr_a==exr_b,
                reference_integrity=integrity,raw_frames=len(raw_a),exr_files=len(exr_a),
                changed_raw=[i for i in raw_a if raw_a[i]!=raw_b.get(i)],
                changed_exr=[i for i in exr_a if exr_a[i]!=exr_b.get(i)],
                direct_seconds=a['seconds'],worker_inference_seconds=b['inference_seconds'],
                reference_sources=a['source_files'],worker_pid=b['pid'])


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--bench',type=Path,required=True)
    ap.add_argument('--package',type=Path,required=True)
    ap.add_argument('--job',required=True)
    ap.add_argument('--name',required=True)
    args=ap.parse_args()
    db=sqlite3.connect(args.data/'state.sqlite3')
    job=json.loads(db.execute('SELECT doc FROM records WHERE id=? AND kind="job"',(args.job,)).fetchone()[0])
    source=args.data/'assets'/job['request']['asset_id']/'frames'
    case=dict(name=args.name,input=str(source),config=dict(hann_chunk=24,hann_stride=12,num_frames=72,stage2_batch=1,
                                                        carry_hint=True,auto_vram=False,model_scale=1,low_vram=False))
    case_path=args.data/'direct-cases'/(args.name+'.json');save(case_path,case)
    reference=args.data/'references'/args.name
    if not reference.exists():
        subprocess.run([sys.executable,str(args.bench/'bench/run_case.py'),'--package',str(args.package),
            '--case',str(case_path),'--output',str(reference),'--mode','quality','--warmup-frames','8'],check=True,cwd=args.bench)
    report=compare(reference,args.data/'jobs'/args.job/'published/worker.json')
    execution=json.loads((args.data/'execution.json').read_text())
    report['runtime_sources_match']=report.pop('reference_sources')==execution['runtime']
    report['pass_exact']=report['pass_exact'] and report['runtime_sources_match']
    save(args.data/'qualification'/(args.name+'.json'),report)
    print(json.dumps(report,indent=2))
    if not report['pass_exact']:raise SystemExit(1)


if __name__=='__main__':main()
