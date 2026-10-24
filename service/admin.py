"""Local-only service provisioning. Tokens are written to files, never printed."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import secrets
import subprocess
import time

from .common import canonical, digest, save
from .store import Store


def provision(package, data, config, workers, capture_raw):
    package, data = package.resolve(), data.resolve()
    weight = package/'weights/corridorkey_v2.safetensors'
    foundations = {
        'foundation_stats': package/'weights/foundation_stats.safetensors',
        'rvm': package/'vfms/rvm/pretrain_rvm_small16_256_204031069.npz',
        'moge': package/'vfms/moge/model.pt',
    }
    # Resolve the bundle root explicitly: Path.rglob does not descend directory
    # symlinks used by the benchmark candidates.
    cradio=(package/'vfms/cradio/C-RADIOv4-SO400M').resolve()
    foundations.update({'cradio/'+str(p.relative_to(cradio)):p for p in sorted(cradio.rglob('*'))
                        if p.is_file() and '.cache' not in p.parts and '__pycache__' not in p.parts})
    runtime = {str(p.relative_to(package)):digest(p) for p in sorted(package.rglob('*.py'))
               if '.git' not in p.parts and '__pycache__' not in p.parts}
    stack = {n:importlib.metadata.version(n) for n in ('torch','numpy','opencv-python','triton')}
    stack['python'] = platform.python_version()
    foundation_digest = hashlib.sha256(canonical({name:digest(p) for name,p in foundations.items()})).hexdigest()
    model_digest = digest(weight)
    execution = hashlib.sha256(canonical(dict(runtime=runtime,stack=stack,checkpoint=model_digest,
        worker_sha256=digest(Path(__file__).with_name('worker.py')),
        foundations=foundation_digest,profile=dict(hann_chunk=24,hann_stride=12,stage2_batch=1,
        model_scale=1,low_vram=False,seed=0,torch_threads=2,opencv_threads=1,tf32_matmul=False,
        tf32_cudnn=True,cudnn_benchmark=False,warmup_frames=8)))).hexdigest()
    approved=json.loads(Path(__file__).with_name('qualification.json').read_text())
    hardware=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True).strip()
    if (execution!=approved['execution_fingerprint'] or
        hardware!=approved['gpu_class']+', '+approved['driver_version']):
        raise RuntimeError('Unqualified runtime, model, worker, GPU or driver. Complete exactness qualification before provisioning this profile.')
    model_revision = 'ck2-'+model_digest[:12]
    caps = dict(api_versions=['v1'],api_revision='0.1.0-pilot',capabilities_revision='pilot-'+execution[:16],
        models=[dict(revision=model_revision,name='CorridorKey2',checkpoint_sha256=model_digest,
            foundation_bundle_sha256=foundation_digest,runtime_commit='86df1a18bcfabc72296db6764ca1449aecf8011f')],
        profiles=[dict(id='native-720p-24x12',model_revision=model_revision,execution_fingerprint=execution,
            quality_policy='baseline_exact',hann_chunk=24,hann_stride=12,stage2_batch=1,model_scale=1,
            low_vram=False,max_width=1280,max_height=720,max_context_frames=72,max_output_frames=72,
            qualification=dict(state='qualified',suite_revision=approved['suite_revision'],
                gpu_class=approved['gpu_class'],execution_stack=json.dumps(stack,sort_keys=True)))],
        input_formats=['rgb8_srgb_png_sequence'],output_formats=['float32_zip_exr_alpha_fg'],
        features=dict(hints=False,vfr=False,trimmed_context=False,scaled_preview=False,premultiplied_output=False),
        limits=dict(max_active_gpu_jobs=workers,max_pending_jobs=8,max_active_jobs_per_user=1,
            max_pending_jobs_per_user=2,max_asset_bytes=1<<30,user_storage_quota_bytes=4<<30,
            max_patch_bytes=1<<20,max_validation_jobs=1,max_queue_age_seconds=600,max_run_seconds=600),
        retention=dict(incomplete_upload_seconds=86400,ready_asset_seconds=86400,result_seconds=86400,idempotency_seconds=604800),
        transfer=dict(tus_version='1.0.0',extensions=['expiration','termination'],checksum='sha256_whole_file',download_ranges=True),
        service_policy=dict(access='free',sponsor_processing_benefits=False))
    save(data/'capabilities.json',caps)
    save(data/'execution.json',dict(runtime=runtime,stack=stack,checkpoint_sha256=model_digest,foundation_bundle_sha256=foundation_digest))
    save(config,dict(package=str(package),data=str(data),capabilities=str(data/'capabilities.json'),workers=workers,
                     capture_raw=capture_raw,port=18080,cancel_grace_seconds=10,disk_floor_bytes=10<<30))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='command',required=True)
    init=sub.add_parser('init')
    init.add_argument('--package',type=Path,required=True)
    init.add_argument('--data',type=Path,required=True)
    init.add_argument('--config',type=Path,required=True)
    init.add_argument('--workers',type=int,choices=(1,2),default=1)
    init.add_argument('--capture-raw',action='store_true')
    token=sub.add_parser('token')
    token.add_argument('--data',type=Path,required=True)
    token.add_argument('--owner',required=True)
    token.add_argument('--output',type=Path,required=True)
    token.add_argument('--hours',type=float,default=72)
    revoke=sub.add_parser('revoke')
    revoke.add_argument('--data',type=Path,required=True)
    revoke.add_argument('--token-file',type=Path,required=True)
    args=ap.parse_args()
    os.umask(0o077)
    if args.command=='init':
        provision(args.package,args.data,args.config,args.workers,args.capture_raw)
    elif args.command=='token':
        value=secrets.token_urlsafe(32)
        args.output.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
        with args.output.open('x') as f:
            f.write(value+'\n')
        store=Store(args.data)
        with store.db:
            store.db.execute('INSERT INTO tokens VALUES(?,?,?)',
                             (hashlib.sha256(value.encode()).hexdigest(),args.owner,time.time()+3600*args.hours))
        store.db.close()
    else:
        value=args.token_file.read_text().strip()
        store=Store(args.data)
        with store.db:
            store.db.execute('DELETE FROM tokens WHERE hash=?',(hashlib.sha256(value.encode()).hexdigest(),))
        store.db.close()


if __name__=='__main__':
    main()
