"""Compare reused workers against independently qualified direct references."""
import argparse
import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from service.common import digest,save


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--jobs',type=Path,required=True)
    ap.add_argument('--references',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();db=sqlite3.connect(args.data/'state.sqlite3');reports=[]
    for item in json.loads(args.jobs.read_text()):
        job=json.loads(db.execute('SELECT doc FROM records WHERE kind=? AND id=?',('job',item['job_id'])).fetchone()[0])
        assert job['state']=='succeeded'
        reference=json.loads((args.references/item['name']/'reference.json').read_text())
        root=args.data/'jobs'/job['id']/'published';worker=json.loads((root/'worker.json').read_text())
        assert worker['capture_raw'] and worker['raw']==reference['raw'],'Raw predictions differ'
        assert job['recipe']['processing_fingerprint']==reference['processing_fingerprint'],'Recipe differs'
        rendered={Path(f['path']).name:f['sha256'] for f in worker['files']}
        assert rendered==reference['rendered'],'EXR hashes differ'
        assert all(digest(root/f['path'])==f['sha256'] for f in worker['files']),'Published bytes differ'
        reports.append(dict(name=item['name'],job_id=job['id'],passed=True,raw_frames=len(worker['raw']),
            exr_files=len(worker['files']),peak_allocated_bytes=worker['peak_allocated_bytes'],
            peak_reserved_bytes=worker['peak_reserved_bytes'],server_timings=item['server_timings']))
    save(args.output,dict(passed=True,raw_frames=sum(r['raw_frames'] for r in reports),
        exr_files=sum(r['exr_files'] for r in reports),cases=reports))
    print(json.dumps(reports),flush=True)


if __name__=='__main__':main()
