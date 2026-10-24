"""Loopback-only pilot. Use an SSH tunnel; public hosting needs a hardened ingress."""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from email.utils import formatdate
import fcntl
from fractions import Fraction
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import re
import shutil
import signal
import socket
import threading
import time
import traceback
from urllib.parse import urlsplit

from jsonschema import Draft202012Validator, ValidationError

from .common import canonical, digest, public, save, stamp
from .store import Store, Problem, ident
from .worker import main as worker_main

def purge(path):
    if path.exists():
        shutil.rmtree(path)


class Controller:
    def __init__(self, config, start_workers=True):
        self.config = config
        self.root = Path(config['data']).resolve()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.lockfile = (self.root / 'server.lock').open('a')
        fcntl.flock(self.lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.event('startup',workers=config['workers'],capture_raw=config.get('capture_raw',False))
        self.store = Store(self.root)
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.validator = ThreadPoolExecutor(max_workers=1)
        self.validation_slots = threading.BoundedSemaphore(2)
        self.cap = json.loads(Path(config['capabilities']).read_text())
        self.cap['limits']['max_active_gpu_jobs'] = config['workers']
        self.schema = json.loads((Path(__file__).resolve().parents[1] / 'docs/service/openapi.json').read_text())
        self.slots = []
        self.owner_order = deque()
        self.allow_workers = start_workers
        self.readers = {}
        self.validating = set()
        self.ctx = mp.get_context('spawn')
        # A controller restart does not pretend old in-flight work succeeded.
        for owner, job in self.store.all('job'):
            if job['state'] in ('running', 'cancel_requested'):
                job.update(state='failed', error=self.error('controller_restarted'), updated_at=stamp())
                self.store.put('job', owner, job)
            elif job['state'] == 'queued':
                job['_accepted'] = time.monotonic() - max(0,time.time()-job['_created'])
                self.store.put('job', owner, job)
        for owner, asset in self.store.all('asset'):
            if asset['state'] == 'validating':
                self.validator.submit(self.validate_asset, owner, asset['id'])
        if start_workers:
            self.slots.append(self.spawn(0))
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    @staticmethod
    def error(code):
        return dict(type='about:blank', title=code, status=500, code=code,
                    retryable=True, request_id=ident('internal'))

    def event(self, name, **data):
        with (self.root/'events.jsonl').open('ab') as stream:
            stream.write(canonical(dict(event=name,monotonic=time.monotonic(),utc=time.time(),**data))+b'\n')
            stream.flush()
            os.fsync(stream.fileno())

    def validate(self, name, value):
        schema = dict(self.schema, **{'$ref': '#/components/schemas/' + name})
        try:
            Draft202012Validator(schema).validate(value)
        except ValidationError:
            raise Problem(422, 'invalid_schema', name) from None

    def spawn(self, number):
        inbox, events = self.ctx.Queue(), self.ctx.Queue()
        process = self.ctx.Process(target=worker_main,
            args=(self.config['package'], inbox, events, str(self.root / f'worker-{number}.log'),
                  self.config.get('capture_raw', False)), daemon=True)
        process.start()
        self.event('worker_spawn',worker=number,pid=process.pid)
        return dict(process=process, inbox=inbox, events=events, ready=False,
                    job=None, number=number, boot=time.monotonic())

    def reap(self, slot):
        p = slot['process']
        if p.is_alive():
            # setsid happens before model imports; never target the controller group.
            try:
                if os.getpgid(p.pid) == p.pid:
                    os.killpg(p.pid, signal.SIGKILL)
                else:
                    p.kill()
            except ProcessLookupError:
                pass
        p.join(timeout=5)
        if p.is_alive():
            raise RuntimeError('Worker could not be reaped; refusing replacement')
        slot['inbox'].close()
        slot['events'].close()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)
        with self.lock:
            for s in self.slots:
                self.reap(s)
        self.validator.shutdown(wait=True)
        self.event('shutdown')
        self.store.db.close()
        self.lockfile.close()

    def asset_dir(self, id):
        return self.root / 'assets' / id

    def job_dir(self, id):
        return self.root / 'jobs' / id

    def quota(self, owner, additional):
        used = sum(sum(f['bytes'] for f in a['manifest']['files']) for o,a in self.store.all('asset')
                   if o == owner and a['state'] not in ('deleted', 'expired'))
        used += sum(j.get('_reserved', 0) for o,j in self.store.all('job')
                    if o == owner and j['state'] not in ('deleted', 'expired', 'cancelled', 'failed'))
        if used + additional > self.cap['limits']['user_storage_quota_bytes']:
            raise Problem(429, 'storage_quota')
        reservations = sum(j.get('_reserved', 0)*2 for _,j in self.store.all('job') if j['state'] in ('queued','running'))
        pending_uploads = sum(sum(f['bytes'] for f in a['manifest']['files']) for _,a in self.store.all('asset') if a['state']=='uploading')
        if shutil.disk_usage(self.root).free - additional*2 - reservations - pending_uploads < self.config.get('disk_floor_bytes', 10 << 30):
            raise Problem(507, 'disk_reserve')

    def create_asset(self, owner, body, key, path):
        previous = self.store.replay(owner, path, key, body)
        if previous:
            return 200, previous
        self.validate('AssetCreate', body)
        n, w, h = body['frame_count'], body['width'], body['height']
        profile = self.cap['profiles'][0]
        if n > profile['max_context_frames'] or w > profile['max_width'] or h > profile['max_height']:
            raise Problem(422, 'input_limit')
        if sorted(f['frame_index'] for f in body['files']) != list(range(n)):
            raise Problem(422, 'frame_mapping')
        t = body['timing']
        if len(t['start_pts'])>20 or len(t['frame_duration_ticks'])>19:
            raise Problem(422,'timestamp_range')
        start_pts,duration=int(t['start_pts']),int(t['frame_duration_ticks'])
        if not -(1<<63)<=start_pts<=(1<<63)-1 or not 0<duration<=(1<<63)-1 or start_pts+n*duration>(1<<63)-1:
            raise Problem(422,'timestamp_range')
        rate, base = t['frame_rate'], t['time_base']
        if Fraction(base['num'],base['den']) * int(t['frame_duration_ticks']) != Fraction(rate['den'],rate['num']):
            raise Problem(422, 'cfr_timing')
        size = sum(f['bytes'] for f in body['files'])
        if size > self.cap['limits']['max_asset_bytes'] or any(f['bytes'] > 16 << 20 for f in body['files']):
            raise Problem(413, 'asset_too_large')
        self.quota(owner, size)
        id = ident('asset')
        expires = time.time() + self.cap['retention']['incomplete_upload_seconds']
        doc = dict(id=id, state='uploading', manifest=body, files=[], created_at=stamp(), expires_at=stamp(expires), _expires=expires)
        directory = self.asset_dir(id)
        (directory / 'frames').mkdir(mode=0o700, parents=True)
        for f in sorted(body['files'], key=lambda f:f['frame_index']):
            upload = ident('upload')
            doc['files'].append(dict(frame_index=f['frame_index'], file_id=ident('input'), upload_id=upload,
                upload_url='/v1/uploads/'+upload, bytes=f['bytes'], expires_at=stamp(expires)))
        # Slots are in the asset record, so creation and the idempotency key are atomic.
        doc['_offsets'] = {f['upload_id']:0 for f in doc['files']}
        self.store.remember(owner, path, key, body, 'asset', doc)
        return 201, doc

    def upload(self, owner, upload_id):
        for o,a in self.store.all('asset'):
            if o == owner:
                for slot in a['files']:
                    if slot['upload_id'] == upload_id:
                        if a['state'] in ('expired','deleted','deleting') or a['_expires'] <= time.time():
                            raise Problem(410, 'upload_expired')
                        return a, slot, self.asset_dir(a['id']) / 'frames' / f'{slot["frame_index"]:06d}.png'
        raise Problem(404, 'not_found')

    def complete(self, owner, id, key, path, body):
        old = self.store.replay(owner, path, key, body)
        if old:
            return 200, old
        if body != {}:
            raise Problem(422, 'invalid_schema')
        a = self.store.get('asset', id, owner)
        if a['state'] != 'uploading':
            if a['state'] in ('ready', 'validating', 'invalid'):
                self.store.remember(owner,path,key,body,'asset',a)
                return 200,a
            raise Problem(410, 'resource_gone')
        if any(a['_offsets'][f['upload_id']] != f['bytes'] for f in a['files']):
            raise Problem(409, 'incomplete_upload')
        if not self.validation_slots.acquire(blocking=False):
            raise Problem(429, 'validation_busy')
        a['state'] = 'validating'
        self.store.remember(owner,path,key,body,'asset',a)
        self.validator.submit(self.validate_asset, owner, id, True)
        return 202, a

    def validate_asset(self, owner, id, release=False):
        try:
            from PIL import Image
            with self.lock:
                self.validating.add(id)
                a = self.store.get('asset', id, owner)
            for f in a['manifest']['files']:
                p = self.asset_dir(id) / 'frames' / f'{f["frame_index"]:06d}.png'
                if p.stat().st_size != f['bytes'] or digest(p) != f['sha256']:
                    raise ValueError('checksum')
                # Check bounded IHDR before a decoder allocates any pixel memory.
                header = p.read_bytes()[:33]
                import struct
                w,h,depth,colour,compression,filtering,interlace = struct.unpack('>IIBBBBB', header[16:29])
                if header[:8] != b'\x89PNG\r\n\x1a\n' or header[12:16] != b'IHDR' or (w,h) != (a['manifest']['width'],a['manifest']['height']) or (depth,colour,compression,filtering,interlace) != (8,2,0,0,0):
                    raise ValueError('png_format')
                with Image.open(p) as image:
                    if image.format != 'PNG' or image.mode != 'RGB' or image.size != (w,h):
                        raise ValueError('geometry')
                    image.verify()
                with Image.open(p) as image:
                    image.load()
            with self.lock:
                current = self.store.get('asset', id, owner)
                if current['state'] == 'validating':
                    normalized=dict(a['manifest'],files=sorted(a['manifest']['files'],key=lambda f:f['frame_index']))
                    current.update(state='ready', content_sha256=hashlib.sha256(canonical(normalized)).hexdigest(),
                                   _expires=time.time()+self.cap['retention']['ready_asset_seconds'])
                    current['expires_at'] = stamp(current['_expires'])
                    self.store.put('asset', owner, current)
        except Exception:
            with self.lock:
                a = self.store.get('asset', id, owner)
                if a['state'] == 'validating':
                    a.update(state='invalid', error=self.error('invalid_media'))
                    self.store.put('asset', owner, a)
        finally:
            with self.lock:
                self.validating.discard(id)
            if release:
                self.validation_slots.release()

    def create_job(self, owner, body, key, path, retry_of=None):
        previous = self.store.replay(owner,path,key,body)
        if previous:
            return 200, previous
        if retry_of:
            old = self.store.get('job', retry_of, owner)
            if old['state'] not in ('failed','cancelled') or body != {}:
                raise Problem(409, 'not_retryable')
            request = old['request']
        else:
            request = body
        self.validate('JobCreate', request)
        profile = self.cap['profiles'][0]
        if (request['capabilities_revision'] != self.cap['capabilities_revision'] or
            request['model_revision'] != profile['model_revision'] or request['profile_id'] != profile['id']):
            raise Problem(409, 'unsupported_revision')
        a = self.store.get('asset', request['asset_id'], owner)
        if a['state'] != 'ready' or a['_expires'] <= time.time():
            raise Problem(409, 'asset_not_ready')
        r = request['output_range']
        if not 0 <= r['start'] < r['end'] <= a['manifest']['frame_count']:
            raise Problem(422, 'frame_range')
        pending = [(o,j) for o,j in self.store.all('job') if j['state']=='queued']
        limits = self.cap['limits']
        if len(pending) >= limits['max_pending_jobs'] or sum(o==owner for o,j in pending) >= limits['max_pending_jobs_per_user']:
            raise Problem(429, 'queue_full')
        reserve = a['manifest']['width'] * a['manifest']['height'] * a['manifest']['frame_count'] * 16
        self.quota(owner, reserve)
        recipe = dict(model_revision=profile['model_revision'], profile_id=profile['id'],
                      execution_fingerprint=profile['execution_fingerprint'], asset_content_sha256=a['content_sha256'],
                      processed_range=dict(start=0,end=a['manifest']['frame_count']), output_range=r, output=request['output'])
        j = dict(id=ident('job'), state='queued', request=request, recipe=recipe, attempt=0,
                 created_at=stamp(), updated_at=stamp(), _accepted=time.monotonic(), _created=time.time(), _reserved=reserve)
        if retry_of:
            j['retry_of'] = retry_of
        self.store.remember(owner,path,key,body,'job',j)
        return 202,j

    def get_job(self, owner, id):
        j = self.store.get('job',id,owner)
        waiting = [x['id'] for _,x in self.store.all('job') if x['state']=='queued']
        j['queue'] = dict(position_estimate=waiting.index(id)+1 if id in waiting else None,
                          poll_after_seconds=1,active_gpu_jobs=sum(s['job'] is not None for s in self.slots),
                          max_active_gpu_jobs=self.config['workers'])
        return j

    def cancel(self, owner, id, key, path, body):
        old = self.store.replay(owner,path,key,body)
        if old:
            return 200,old
        if body != {}:
            raise Problem(422, 'invalid_schema')
        j = self.store.get('job',id,owner)
        if j['state'] in ('deleting','deleted','expired'):
            raise Problem(410,'resource_gone')
        if j['state'] == 'queued':
            j['state'] = 'cancelled'
        elif j['state'] == 'running':
            j.update(state='cancel_requested', _cancel=time.monotonic())
        j['updated_at'] = stamp()
        self.store.remember(owner,path,key,body,'job',j)
        return (202 if j['state']=='cancel_requested' else 200),j

    def delete(self, owner, kind, id):
        doc = self.store.get(kind,id,owner)
        for o,d in self.store.all('deletion'):
            if o==owner and d['resource_id']==id:
                return d
        doc['state'] = 'deleting'
        self.store.put(kind,owner,doc)
        if kind=='asset':
            for o,j in self.store.all('job'):
                if o==owner and j['request']['asset_id']==id and j['state']!='deleted':
                    j['state']='deleting'
                    self.store.put('job',o,j)
        d = dict(id=ident('deletion'), resource_id=id,resource_type=kind,state='deleting',requested_at=stamp())
        d['status_url'] = '/v1/deletions/'+d['id']
        self.store.put('deletion',owner,d)
        return d

    def publish(self, slot, owner, j):
        attempt = self.job_dir(j['id']) / 'attempt-1'
        stats = json.loads((attempt / 'worker.json').read_text())
        asset = self.store.get('asset',j['request']['asset_id'],owner)
        if asset['state'] in ('deleting','deleted','expired'):
            j['state'] = 'cancelled'
            return
        expires = time.time()+self.cap['retention']['result_seconds']
        r = j['recipe']['output_range']
        timing, media = asset['manifest']['timing'], asset['manifest']
        files = []
        for f in stats['files']:
            i, output = f['source_frame'], f['output']
            if not r['start'] <= i < r['end']:
                continue
            alpha = output == 'alpha'
            id = ident('file')
            files.append(dict(id=id, source_frame=i, output=output,
                pts=str(int(timing['start_pts'])+i*int(timing['frame_duration_ticks'])),
                duration_ticks=timing['frame_duration_ticks'],channels=['Y'] if alpha else ['R','G','B'],
                width=media['width'],height=media['height'],pixel_type='float32',
                colour_space='coverage' if alpha else 'linear_srgb',alpha_association='coverage' if alpha else 'straight',
                bytes=f['bytes'],sha256=f['sha256'],download_url=f'/v1/jobs/{j["id"]}/files/{id}',
                expires_at=stamp(expires),_path=f['path']))
        if len(files) != 2*(r['end']-r['start']):
            raise RuntimeError('Incomplete publication')
        result = dict(schema_version='v1',job_id=j['id'],recipe=j['recipe'],timing=timing,
                      files=files,expires_at=stamp(expires))
        result['result_fingerprint'] = hashlib.sha256(canonical(public(result))).hexdigest()
        save(attempt / 'result.json', result)
        attempt.rename(self.job_dir(j['id']) / 'published')
        j.update(state='succeeded',_result=result,_expires=expires,result_expires_at=stamp(expires),
                 result_url='/v1/jobs/'+j['id']+'/result')
        j['server_timings']['service_seconds'] = time.monotonic()-j['_started']
        j['server_timings'].update(worker_inference_seconds=stats['inference_seconds'],
            service_start_monotonic=j['_started'],service_end_monotonic=time.monotonic(),
            warmup_seconds=stats['warmup_seconds'],worker_peak_allocated_bytes=stats['peak_allocated_bytes'],
            worker_peak_reserved_bytes=stats['peak_reserved_bytes'],worker_pid=stats['pid'],
            inference_start_monotonic=stats.get('inference_start_monotonic'),
            inference_end_monotonic=stats.get('inference_end_monotonic'))

    def loop(self):
        while not self.stop.wait(.1):
            try:
                with self.lock:
                    self.tick()
            except Exception:
                traceback.print_exc()

    def tick(self):
        # Stagger model loading; retain one CUDA process per configured lane.
        if (self.allow_workers and len(self.slots)<self.config['workers'] and
            self.slots and self.slots[-1]['ready']):
            self.slots.append(self.spawn(len(self.slots)))
        for index,s in enumerate(self.slots):
            while True:
                try:
                    event,data = s['events'].get_nowait()
                except queue.Empty:
                    break
                if event == 'ready':
                    s.update(ready=True,load=data)
                    self.event('worker_ready',worker=s['number'],**data)
                elif event in ('done','failed','progress') and s['job'] == data.get('id'):
                    owner = s['owner']
                    j = self.store.get('job',s['job'],owner)
                    if event == 'progress' and j['state']=='running':
                        j['progress'] = dict(phase=data['phase'], completed=data['completed'],
                            total=j['recipe']['processed_range']['end'],unit='frames',heartbeat_at=stamp(),eta_seconds=None)
                    elif event in ('done','failed'):
                        if j['state'] in ('cancel_requested','deleting'):
                            if j['state'] != 'deleting':
                                j['state'] = 'cancelled'
                        elif event=='done':
                            self.publish(s,owner,j)
                        else:
                            j.update(state='failed',error=self.error('worker_failed'))
                            self.reap(s)
                        j['updated_at'] = stamp()
                        self.store.put('job',owner,j)
                        self.event('job_terminal',job=j['id'],state=j['state'],worker=s['number'])
                        s['job'] = None
                    self.store.put('job',owner,j)
            if s['job']:
                j = self.store.get('job',s['job'],s['owner'])
                run_expired = time.monotonic()-j['_started'] > self.cap['limits']['max_run_seconds']
                if j['state']=='deleting' or run_expired or (j['state']=='cancel_requested' and time.monotonic()-j['_cancel']>=self.config.get('cancel_grace_seconds',10)):
                    self.reap(s)
                    self.event('worker_reaped',worker=s['number'],job=j['id'])
                    if j['state']!='deleting':
                        j['state'] = 'failed' if run_expired else 'cancelled'
                        if run_expired:
                            j['error'] = self.error('run_deadline')
                    j['updated_at'] = stamp()
                    self.store.put('job',s['owner'],j)
                    shutil.rmtree(self.job_dir(j['id'])/'attempt-1',ignore_errors=True)
                    s['job']=None
            if not s['process'].is_alive():
                if s['job']:
                    j=self.store.get('job',s['job'],s['owner'])
                    j.update(state='failed',error=self.error('worker_lost'),updated_at=stamp())
                    self.store.put('job',s['owner'],j)
                self.reap(s)
                self.slots[index]=self.spawn(s['number'])
                continue
            if not s['ready'] and time.monotonic()-s['boot']>180:
                self.reap(s)
                continue
            if s['ready'] and not s['job']:
                active_owners={x.get('owner') for x in self.slots if x['job']}
                pending=[(o,j) for o,j in self.store.all('job') if j['state']=='queued']
                owners=list(dict.fromkeys(o for o,j in pending))
                self.owner_order=deque(o for o in self.owner_order if o in owners)
                self.owner_order.extend(o for o in owners if o not in self.owner_order)
                next_owner=next((o for o in self.owner_order if o not in active_owners),None)
                jobs=[(o,j) for o,j in pending if o==next_owner]
                if jobs:
                    owner,j=jobs[0]
                    self.owner_order.remove(owner)
                    self.owner_order.append(owner)
                    a=self.store.get('asset',j['request']['asset_id'],owner)
                    if a['state'] not in ('ready',):
                        j['state']='cancelled'
                        self.store.put('job',owner,j)
                        continue
                    if time.time()-j['_created']>self.cap['limits']['max_queue_age_seconds']:
                        j.update(state='failed',error=self.error('queue_deadline'))
                        self.store.put('job',owner,j)
                        continue
                    attempt=self.job_dir(j['id'])/'attempt-1'
                    attempt.mkdir(mode=0o700,parents=True,exist_ok=False)
                    j.update(state='running',attempt=1,_started=time.monotonic(),updated_at=stamp(),
                        server_timings=dict(queue_seconds=max(0,time.monotonic()-j['_accepted']),
                            service_seconds=None,model_load_seconds_in_service=0))
                    self.store.put('job',owner,j)
                    s.update(job=j['id'],owner=owner)
                    self.event('job_start',job=j['id'],worker=s['number'],pid=s['process'].pid)
                    s['inbox'].put((j['id'],str(self.asset_dir(a['id'])/'frames'),str(attempt),a['manifest']['frame_count']))
        self.cleanup()

    def cleanup(self):
        active={s['job'] for s in self.slots if s['job']}
        for owner,j in self.store.all('job'):
            if j['state']=='queued' and time.time()-j['_created']>self.cap['limits']['max_queue_age_seconds']:
                j.update(state='failed',error=self.error('queue_deadline'),updated_at=stamp())
                self.store.put('job',owner,j)
        jobs=self.store.all('job')
        pinned={j['request']['asset_id'] for _,j in jobs if j['state'] in ('queued','running','cancel_requested')}
        for owner,a in self.store.all('asset'):
            if a['state']=='deleting':
                for o,j in jobs:
                    if o==owner and j['request']['asset_id']==a['id'] and j['state']!='deleted':
                        j['state']='deleting'
                        self.store.put('job',o,j)
            elif a['_expires']<=time.time() and a['id'] not in pinned and a['state'] not in ('deleted','expired','validating'):
                a['state']='expired'
                self.store.put('asset',owner,a)
            if a['state']=='expired':
                purge(self.asset_dir(a['id']))
        for owner,j in self.store.all('job'):
            if j['state']=='deleting' and j['id'] not in active and not self.readers.get(j['id']):
                purge(self.job_dir(j['id']))
                j['state']='deleted'
                j.pop('_result',None)
                self.store.put('job',owner,j)
            elif j['state']=='succeeded' and j['_expires']<=time.time():
                j['state']='expired'
                j.pop('_result',None)
                self.store.put('job',owner,j)
            if j['state']=='expired' and not self.readers.get(j['id']):
                purge(self.job_dir(j['id']))
            elif j['state'] in ('cancelled','failed') and j['id'] not in active:
                purge(self.job_dir(j['id']))
        for owner,a in self.store.all('asset'):
            if a['state']=='deleting' and a['id'] not in self.validating and not any(j['request']['asset_id']==a['id'] and j['state']!='deleted' for _,j in self.store.all('job')):
                purge(self.asset_dir(a['id']))
                a['state']='deleted'
                self.store.put('asset',owner,a)
        for owner,d in self.store.all('deletion'):
            if d['state']=='deleting' and self.store.get(d['resource_type'],d['resource_id'],owner)['state']=='deleted':
                d.update(state='deleted',completed_at=stamp())
                self.store.put('deletion',owner,d)


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    server_version = 'CorridorKeyPrivate/0.1'

    def setup(self):
        super().setup()
        self.connection.settimeout(30)
        self.connection.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)

    def log_message(self, *_args):
        pass  # Never log bearer credentials or user media identifiers.

    def reply(self,status,value=None,headers=None):
        data=b'' if value is None else canonical(public(value))
        self.send_response(status)
        self.send_header('X-Request-ID',self.request_id)
        self.send_header('Cache-Control','no-store')
        if self.close_connection:
            self.send_header('Connection','close')
        self.send_header('Content-Length',str(len(data)))
        if value is not None:
            self.send_header('Content-Type','application/json')
        for k,v in (headers or {}).items():
            self.send_header(k,str(v))
        self.end_headers()
        if self.command!='HEAD' and data:
            self.wfile.write(data)

    def body(self,limit):
        if self.headers.get('Transfer-Encoding'):
            raise Problem(400,'unsupported_transfer_encoding')
        try:
            size=int(self.headers.get('Content-Length','0'))
        except ValueError:
            raise Problem(400,'content_length') from None
        if not 0<=size<=limit:
            raise Problem(413,'body_limit')
        value=self.rfile.read(size)
        if len(value)!=size:
            raise Problem(400,'short_body')
        return value

    def handle_request(self):
        self.request_id=ident('request')
        c=self.server.controller
        try:
            parsed=urlsplit(self.path)
            if parsed.query or parsed.fragment or not re.fullmatch(r'/v1/[A-Za-z0-9_/-]+',parsed.path):
                raise Problem(404,'not_found')
            path=parsed.path
            auth=self.headers.get('Authorization','')
            with c.lock:
                owner=c.store.authenticate(auth[7:] if auth.startswith('Bearer ') else '')
            if '/files/' in path and self.command in ('GET','HEAD'):
                return self.download(c,owner,path)
            raw=self.body(c.cap['limits']['max_patch_bytes'] if self.command=='PATCH' else 128<<10)
            body=None
            if self.command=='POST':
                if self.headers.get_content_type()!='application/json':
                    raise Problem(415,'json_required')
                try:
                    body=json.loads(raw,parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
                except (ValueError,UnicodeError):
                    raise Problem(400,'invalid_json') from None
            with c.lock:
                status,value,headers=self.route(c,owner,path,body,raw)
            return self.reply(status,value,headers)
        except Problem as e:
            self.close_connection=True  # Unread/rejected request bodies cannot poison keep-alive.
            return self.reply(e.status,dict(type='about:blank',title=e.code,status=e.status,code=e.code,
                detail=e.detail,retryable=e.status in (429,503,507),request_id=self.request_id),
                {'Retry-After':'1'} if e.status in (429,503) else ({'WWW-Authenticate':'Bearer'} if e.status==401 else {}))
        except (BrokenPipeError,ConnectionResetError,TimeoutError):
            self.close_connection=True
        except Exception:
            traceback.print_exc()
            self.close_connection=True
            self.reply(500,dict(type='about:blank',title='internal_error',status=500,code='internal_error',
                              retryable=False,request_id=self.request_id))

    def route(self,c,owner,path,body,raw):
        method=self.command
        key=self.headers.get('Idempotency-Key')
        p=path.strip('/').split('/')
        if path=='/v1/capabilities' and method=='GET':
            cap=dict(c.cap,worker_pool=dict(ready=sum(s['ready'] and not s['job'] for s in c.slots),
                busy=sum(s['job'] is not None for s in c.slots),starting=sum(not s['ready'] for s in c.slots),
                capture_raw=c.config.get('capture_raw',False)))
            return 200,cap,{'ETag':'"'+hashlib.sha256(canonical(cap)).hexdigest()+'"'}
        if path=='/v1/assets' and method=='POST':
            status,value=c.create_asset(owner,body,key,path)
            return status,value,{'Location':'/v1/assets/'+value['id']}
        if path=='/v1/jobs' and method=='POST':
            status,value=c.create_job(owner,body,key,path)
            return status,value,{'Location':'/v1/jobs/'+value['id']}
        if len(p)>=3 and p[1]=='assets':
            a=c.store.get('asset',p[2],owner)
            if len(p)==3 and method=='GET':
                return 200,a,{'Retry-After':'1'}
            if len(p)==3 and method=='DELETE':
                return 202,c.delete(owner,'asset',p[2]),{}
            if len(p)==4 and p[3]=='complete' and method=='POST':
                status,value=c.complete(owner,p[2],key,path,body)
                return status,value,{}
        if len(p)==3 and p[1]=='uploads':
            a,f,target=c.upload(owner,p[2])
            offset=a['_offsets'][p[2]]
            headers={'Tus-Resumable':'1.0.0','Upload-Offset':offset,'Upload-Length':f['bytes'],
                     'Upload-Expires':formatdate(a['_expires'],usegmt=True),'Cache-Control':'no-store'}
            if method=='OPTIONS':
                headers.update({'Tus-Version':'1.0.0','Tus-Extension':'expiration,termination','Tus-Max-Size':16<<20})
                return 204,None,headers
            if self.headers.get('Tus-Resumable')!='1.0.0':
                raise Problem(412,'tus_version')
            if method=='HEAD':
                return 200,None,headers
            if a['state']!='uploading':
                raise Problem(409,'upload_frozen')
            if method=='PATCH':
                if self.headers.get('Content-Type')!='application/offset+octet-stream':
                    raise Problem(415,'patch_content_type')
                if self.headers.get('Upload-Offset')!=str(offset):
                    raise Problem(409,'offset_conflict')
                if offset+len(raw)>f['bytes'] or not raw:
                    raise Problem(413,'upload_length')
                # DB offset advances only after durable bytes; an uncommitted tail is
                # truncated on retry after an interrupted PATCH or controller crash.
                with target.open('r+b' if target.exists() else 'w+b') as stream:
                    stream.truncate(offset)
                    stream.seek(offset)
                    stream.write(raw)
                    stream.flush()
                    os.fsync(stream.fileno())
                directory=os.open(target.parent,os.O_RDONLY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
                a['_offsets'][p[2]]=offset+len(raw)
                c.store.put('asset',owner,a)
                headers['Upload-Offset']=offset+len(raw)
                return 204,None,headers
            if method=='DELETE':
                c.delete(owner,'asset',a['id'])
                return 204,None,headers
        if len(p)>=3 and p[1]=='jobs':
            j=c.get_job(owner,p[2])
            if len(p)==3 and method=='GET':
                return 200,j,{'Retry-After':'1'}
            if len(p)==3 and method=='DELETE':
                return 202,c.delete(owner,'job',p[2]),{}
            if len(p)==4 and p[3]=='cancel' and method=='POST':
                status,value=c.cancel(owner,p[2],key,path,body)
                return status,value,{}
            if len(p)==4 and p[3]=='retry' and method=='POST':
                status,value=c.create_job(owner,body,key,path,retry_of=p[2])
                return status,value,{}
            if len(p)==4 and p[3]=='result' and method=='GET':
                if j['state']!='succeeded' or j['_expires']<=time.time():
                    raise Problem(410 if j['state'] in ('deleted','expired','deleting') else 409,'result_unavailable')
                return 200,j['_result'],{}
        if len(p)==3 and p[1]=='deletions' and method=='GET':
            return 200,c.store.get('deletion',p[2],owner),{}
        raise Problem(404,'not_found')

    def download(self,c,owner,path):
        p=path.strip('/').split('/')
        if len(p)!=5 or p[1]!='jobs' or p[3]!='files':
            raise Problem(404,'not_found')
        with c.lock:
            j=c.store.get('job',p[2],owner)
            if j['state']!='succeeded' or j['_expires']<=time.time():
                raise Problem(410,'result_unavailable')
            f=next((f for f in j['_result']['files'] if f['id']==p[4]),None)
            if not f:
                raise Problem(404,'not_found')
            source=c.job_dir(j['id'])/'published'/f['_path']
            stream=source.open('rb')
            c.readers[j['id']]=c.readers.get(j['id'],0)+1
        try:
            size=f['bytes']
            tag='"'+f['sha256']+'"'
            start,end,status=0,size-1,200
            requested=self.headers.get('Range')
            if requested and self.headers.get('If-Range',tag)==tag:
                match=re.fullmatch(r'bytes=(\d+)-(\d*)',requested)
                if not match:
                    raise Problem(416,'invalid_range')
                start=int(match[1]); end=min(int(match[2]) if match[2] else size-1,size-1)
                if not 0<=start<=end<size:
                    raise Problem(416,'invalid_range')
                status=206
            self.send_response(status)
            for k,v in {'Content-Length':end-start+1,'Content-Type':'image/x-exr','ETag':tag,
                        'Accept-Ranges':'bytes','Cache-Control':'private, no-store','X-Request-ID':self.request_id,
                        **({'Content-Range':f'bytes {start}-{end}/{size}'} if status==206 else {})}.items():
                self.send_header(k,str(v))
            self.end_headers()
            if self.command=='HEAD':
                return
            stream.seek(start)
            remaining=end-start+1
            while remaining:
                with c.lock:
                    current=c.store.get('job',j['id'],owner)
                    if current['state']!='succeeded' or current['_expires']<=time.time():
                        self.close_connection=True
                        return
                block=stream.read(min(1<<20,remaining))
                if not block:
                    self.close_connection=True
                    return
                self.wfile.write(block)
                remaining-=len(block)
        finally:
            stream.close()
            with c.lock:
                c.readers[j['id']]-=1

    do_GET=do_HEAD=do_POST=do_PATCH=do_DELETE=do_OPTIONS=handle_request


class Server(ThreadingHTTPServer):
    daemon_threads=True
    request_queue_size=32
    def __init__(self,address,controller):
        self.controller=controller
        self.semaphore=threading.BoundedSemaphore(32)
        super().__init__(address,Handler)

    def process_request(self,request,address):
        if not self.semaphore.acquire(blocking=False):
            request.close()
            return
        try:
            super().process_request(request,address)
        except BaseException:
            self.semaphore.release()
            raise

    def process_request_thread(self,request,address):
        try:
            super().process_request_thread(request,address)
        finally:
            self.semaphore.release()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,required=True)
    args=ap.parse_args()
    os.umask(0o077)
    c=Controller(json.loads(args.config.read_text()))
    server=Server(('127.0.0.1',c.config.get('port',18080)),c)
    def shutdown(_sig,_frame):
        threading.Thread(target=server.shutdown,daemon=True).start()
    signal.signal(signal.SIGTERM,shutdown)
    signal.signal(signal.SIGINT,shutdown)
    try:
        server.serve_forever(poll_interval=.2)
    finally:
        server.server_close()
        c.close()


if __name__=='__main__':
    main()
