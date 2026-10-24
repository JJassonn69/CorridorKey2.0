"""Independent direct calls and a separate mask oracle for each API job."""
import argparse
import dataclasses
import hashlib
import io
import json
import os
from pathlib import Path
import random
import sqlite3
import subprocess
import sys
import time

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from service.common import digest,save


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--package',type=Path,required=True);ap.add_argument('--baseline',type=Path,required=True)
    ap.add_argument('--jobs',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--index',type=int);args=ap.parse_args();jobs=json.loads(args.jobs.read_text())
    args.output.mkdir(mode=0o700,parents=True,exist_ok=True)
    if args.index is None:
        reports=[]
        for i,item in enumerate(jobs):
            with (args.output/(item['name']+'.log')).open('w') as log:
                subprocess.run([sys.executable,__file__,'--data',str(args.data),'--package',str(args.package),
                    '--baseline',str(args.baseline),'--jobs',str(args.jobs),'--output',str(args.output),'--index',str(i)],
                    stdout=log,stderr=subprocess.STDOUT,check=True)
            result=json.loads((args.output/item['name']/'report.json').read_text());reports.append(result)
            save(args.output/'summary.json',dict(passed=len(reports)==len(jobs),raw_frames=sum(r['raw_frames'] for r in reports),
                exr_files=sum(r['exr_files'] for r in reports),cases=reports))
            print(json.dumps(result),flush=True)
        return
    item=jobs[args.index];db=sqlite3.connect(args.data/'state.sqlite3')
    def record(kind,id):return json.loads(db.execute('SELECT doc FROM records WHERE kind=? AND id=?',(kind,id)).fetchone()[0])
    job=record('job',item['job_id']);assert job['state']=='succeeded'
    request=job['request'];carry=dict(enabled=True,confidence=.95,precedence='provided');carry.update(request.get('carry',{}))
    hint=request.get('hint');mode=hint['mode'] if hint else None
    legacy=mode!='keyframes' and not (mode=='sequence' and carry['precedence']=='provided')
    package=args.baseline if legacy else args.package
    sys.path.insert(0,str(package))
    import cv2
    import numpy as np
    import torch
    from corridorkey.runtime import InferenceConfig,RuntimePaths,WriterConfig,load_model,run_inference
    torch.set_num_threads(2);cv2.setNumThreads(1);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=False
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    assets={id:record('asset',id) for id in job['_assets']}
    def paths(id):
        result={}
        for f in assets[id]['manifest']['files']:
            p=args.data/'assets'/id/'frames'/f'{f["frame_index"]:06d}{".png" if f["media_type"]=="image/png" else ".exr"}'
            assert p.stat().st_size==f['bytes'] and digest(p)==f['sha256']
            result[f['frame_index']]=p
        return result
    for id in assets:paths(id)
    source=assets[request['asset_id']]['manifest'];count=source['frame_count']
    out=args.output/item['name'];out.mkdir(mode=0o700,exist_ok=False)
    cfg=InferenceConfig(hann_chunk=24,hann_stride=12,num_frames=count,stage2_batch=1,auto_vram=False,
        model_scale=1,low_vram=False,outputs={'alpha','fg'},output_format='exr',exr_dtype='float',temp_dir=out/'scratch',
        frame_dir_linear=source['colour']['transfer']=='linear',carry_hint=carry['enabled'],carry_hint_quality=carry['confidence'],
        hint_quality=hint.get('confidence',.95) if hint else .95)
    def arguments(limit):
        result=dict(initial_hint_path=None,hint_video_path=None)
        if not legacy:result['hint_precedence']=carry['precedence']
        if not hint:return result
        frames=paths(hint['asset_id'])
        if mode=='first_frame':result['initial_hint_path']=str(frames[0])
        elif mode=='sequence':result['hint_video_path']=str(frames[0].parent)
        else:
            confidence={f['frame_index']:f['confidence'] for f in hint.get('frame_confidence',[])}
            mapped={i:dict(path=str(p),quality=confidence.get(i,hint.get('confidence',.95))) for i,p in frames.items() if i<limit}
            if mapped:result['hint_frames']=mapped
        return result
    model=load_model(RuntimePaths.from_package(package),device='cuda')
    input_dir=str(args.data/'assets'/request['asset_id']/'frames')
    run_inference(model,input_dir,str(out/'warmup'),dataclasses.replace(cfg,num_frames=min(8,count)),WriterConfig(),**arguments(min(8,count)))
    raw=[]
    def capture(_module,_input,result):
        logits=result[0] if isinstance(result,tuple) else result
        for frame in logits[0]:
            array=frame.detach().float().cpu().numpy();assert np.isfinite(array).all()
            buffer=io.BytesIO();np.save(buffer,array,allow_pickle=False)
            raw.append(dict(index=len(raw),shape=list(frame.shape),model_dtype=str(frame.dtype),sha256=hashlib.sha256(buffer.getbuffer()).hexdigest()))
    hook=model.stage2.register_forward_hook(capture)
    try:run_inference(model,input_dir,str(out/'exr'),cfg,WriterConfig(),**arguments(count))
    finally:hook.remove()
    # Independent piecewise selection expresses the contract without calling the
    # service constraint implementation. Foreground files are never rewritten.
    for p in (out/'exr').glob('*_alpha_exr/alpha_*.exr'):
        i=int(p.stem.rsplit('_',1)[1]);alpha=cv2.imread(str(p),cv2.IMREAD_UNCHANGED)
        keep=np.zeros_like(alpha);background=np.zeros_like(alpha)
        for binding in request.get('masks',[]):
            path=paths(binding['asset_id'])[0 if binding['mode']=='static' else i]
            image=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
            divisor=255 if image.dtype==np.uint8 else 65535 if image.dtype==np.uint16 else 1
            image=image.astype(np.float32)/np.float32(divisor)
            target=keep if binding['role']=='force_foreground' else background
            target[:]=np.where(image>target,image,target)
        raised=np.where(alpha>keep,alpha,keep)
        ceiling=np.float32(1)-background
        corrected=np.where(raised<ceiling,raised,ceiling)
        if not np.array_equal(corrected,alpha):
            assert cv2.imwrite(str(p),corrected,[cv2.IMWRITE_EXR_TYPE,cv2.IMWRITE_EXR_TYPE_FLOAT,
                cv2.IMWRITE_EXR_COMPRESSION,cv2.IMWRITE_EXR_COMPRESSION_ZIP])
    worker_dir=args.data/'jobs'/job['id']/'published';worker=json.loads((worker_dir/'worker.json').read_text())
    assert worker['capture_raw'] and len(raw)==count and raw==worker['raw'],'Raw predictions differ'
    rendered={p.name:digest(p) for p in (out/'exr').rglob('*.exr')}
    actual={Path(f['path']).name:f['sha256'] for f in worker['files']}
    assert len(rendered)==len(actual)==count*2 and rendered==actual,'EXR bytes differ'
    assert all(digest(worker_dir/f['path'])==f['sha256'] for f in worker['files'])
    report=dict(name=item['name'],passed=True,raw_frames=count,exr_files=count*2,
        direct_runtime='previously accepted runtime' if legacy else 'extended direct runtime',
        independent_mask_oracle=True,processing_fingerprint=job['recipe']['processing_fingerprint'],
        peak_allocated_bytes=worker['peak_allocated_bytes'],peak_reserved_bytes=worker['peak_reserved_bytes'])
    save(out/'report.json',report);save(out/'reference.json',dict(raw=raw,rendered=rendered,processing_fingerprint=report['processing_fingerprint']))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
