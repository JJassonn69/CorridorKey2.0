"""Recheck every frozen native/low-memory/hint fixture without changing validators."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from service.common import digest, save

CASES=('short','tail','720p','1080p','hair','motion','godiva','low-vram','low-vram-tail','linear-exr','sequence-hint')
CONTROL_PACKAGES={'attrs','jsonschema','jsonschema-specifications','referencing','rpds-py','typing-extensions'}


def gate(reference,candidate):
    a=json.loads((reference/'manifest.json').read_text());b=json.loads((candidate/'manifest.json').read_text())
    assert a['status']==b['status']=='complete' and a['mode']==b['mode']=='quality'
    assert a['warmup_frames']==b['warmup_frames']
    ca,cb=a['contract'].copy(),b['contract'].copy()
    pa,pb=ca.pop('packages'),cb.pop('packages')
    delta={n:[pa.get(n),pb.get(n)] for n in pa.keys()|pb.keys() if pa.get(n)!=pb.get(n)}
    assert set(delta)<=CONTROL_PACKAGES,delta
    assert ca==cb,'Input, configuration, assets, hardware or precision contract changed'
    assert len(a['frames'])==len(b['frames'])==ca['config']['num_frames']
    assert len(a['rendered'])==len(b['rendered'])==2*len(a['frames'])
    for category in ('frames','rendered'):
        for x,y in zip(a[category],b[category]):
            assert x==y,(category,x.get('file'),'metadata or digest differs')
            assert digest(reference/x['file'])==x['sha256']==digest(candidate/y['file'])
    return dict(passed=True,frames=len(a['frames']),exact_files=len(a['frames'])+len(a['rendered']),
                service_only_package_changes=delta)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bench',type=Path,required=True)
    ap.add_argument('--package',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();args.output.mkdir(mode=0o700,parents=True,exist_ok=True)
    reports=[]
    for name in CASES:
        case=args.bench/('bench/optimization' if name in ('low-vram','low-vram-tail','linear-exr','sequence-hint') else 'bench')/('case-'+name+'.json')
        output=args.output/name
        with (args.output/(name+'.log')).open('a') as log:
            if not output.exists():
                subprocess.run([sys.executable,str(args.bench/'bench/run_case.py'),'--package',str(args.package),
                    '--case',str(case),'--output',str(output),'--mode','quality','--warmup-frames','8'],
                    cwd=args.bench,stdout=log,stderr=subprocess.STDOUT,check=True)
        current={str(p.relative_to(args.package)):digest(p) for p in sorted(args.package.rglob('*.py')) if '.git' not in p.parts}
        assert json.loads((output/'manifest.json').read_text())['source_files']==current
        report=dict(case=name,**gate(args.bench/('results/speed3-nn-depthwise-'+name+'-quality'),output))
        reports.append(report);save(args.output/'report.json',dict(passed=len(reports)==len(CASES),
            frames=sum(r['frames'] for r in reports),exact_files=sum(r['exact_files'] for r in reports),cases=reports))
        print(json.dumps(report),flush=True)


if __name__=='__main__':main()
