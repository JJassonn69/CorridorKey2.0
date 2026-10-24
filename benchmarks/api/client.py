#!/usr/bin/env python3
"""Mac-capable benchmark client for the proposed API; Python standard library only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import http.client
import json
import math
from pathlib import Path
import platform
import re
import ssl
import statistics
import time
from urllib.parse import urljoin, urlsplit
import uuid


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        tmp.chmod(0o600)
        json.dump(value, f, indent=2)
        f.write('\n')
        f.flush()
        import os
        os.fsync(f.fileno())
    tmp.replace(path)


def summaries(records):
    result = {}
    for operation in sorted({r['operation'] for r in records}):
        rows = [r for r in records if r['operation'] == operation]
        values = sorted(r['complete_ms'] for r in rows)
        result[operation] = dict(count=len(rows), errors=sum(r['error'] is not None or
                                r['status'] is None or r['status'] >= 400 for r in rows),
                                median_ms=statistics.median(values),
                                p95_ms=values[math.ceil(.95 * len(values)) - 1] if len(values) >= 20 else None,
                                min_ms=values[0], max_ms=values[-1])
    return result


class ApiError(RuntimeError):
    pass


class Client:
    def __init__(self, base, token, timeout=60, sleep=time.sleep):
        self.base = urlsplit(base.rstrip('/') + '/')
        if (self.base.scheme not in ('http', 'https') or not self.base.hostname
                or self.base.username or self.base.password or self.base.query
                or self.base.fragment or self.base.path != '/'):
            raise ValueError('Use a plain API origin without credentials/query/path')
        if self.base.scheme == 'http' and self.base.hostname not in ('localhost','127.0.0.1','::1'):
            raise ValueError('Plain HTTP is allowed only on loopback (for an SSH tunnel)')
        self.origin = self.base.geturl().rstrip('/')
        self.token, self.timeout, self.sleep = token, timeout, sleep
        self.records = []
        self.started = time.perf_counter()
        self.connection = None

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def path(self, path):
        target = urlsplit(urljoin(self.origin + '/', path))
        def origin(u):
            return u.scheme, u.hostname, u.port or (443 if u.scheme == 'https' else 80)
        if origin(target) != origin(self.base) or target.username or target.password or target.fragment:
            raise ValueError('Refusing to send credentials to a different origin')
        if target.query:
            raise ValueError('This pilot client accepts authenticated paths, not signed query URLs')
        return target.path

    def request(self, operation, method, path, payload=None, body=None, headers=None,
                expect=(200,), sink=None, limit=8 * 1024 * 1024):
        path = self.path(path)
        headers = dict(headers or {})
        if payload is not None:
            body = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
            headers['Content-Type'] = 'application/json'
        headers['Authorization'] = 'Bearer ' + self.token
        safe_retry = sink is None and (method in ('GET','HEAD','DELETE','OPTIONS') or (
            method == 'POST' and bool(headers.get('Idempotency-Key'))))
        for attempt in range(3):
            start = time.perf_counter()
            reused = self.connection is not None and self.connection.sock is not None
            record = dict(operation=operation, method=method, status=None,
                          started_at_utc=datetime.now(timezone.utc).isoformat(),
                          start_offset_seconds=start-self.started, request_id=None,
                          sent_bytes=len(body or b''), received_bytes=0,
                          reused_connection=reused, headers_ms=None, complete_ms=None, error=None)
            try:
                if self.connection is None:
                    kind = http.client.HTTPSConnection if self.base.scheme == 'https' else http.client.HTTPConnection
                    kw = {'context':ssl.create_default_context()} if self.base.scheme == 'https' else {}
                    self.connection = kind(self.base.hostname, self.base.port, timeout=self.timeout, **kw)
                self.connection.request(method, path, body=body, headers=headers)
                response = self.connection.getresponse()
                record['headers_ms'] = (time.perf_counter() - start) * 1000
                status = record['status'] = response.status
                received_headers = {k.lower():v for k,v in response.getheaders()}
                record['request_id'] = received_headers.get('x-request-id')
                parts = []
                stream = None
                if sink is not None and status in expect:
                    stream = Path(sink).open('ab' if status == 206 else 'wb')
                try:
                    while True:
                        block = response.read(256 * 1024)
                        if not block:
                            break
                        record['received_bytes'] += len(block)
                        if record['received_bytes'] > limit:
                            raise ApiError('Response exceeds its declared limit')
                        if stream is not None:
                            stream.write(block)
                        else:
                            parts.append(block)
                finally:
                    if stream is not None:
                        stream.close()
                record['complete_ms'] = (time.perf_counter() - start) * 1000
                self.records.append(record)
                if status in expect:
                    return status, received_headers, b''.join(parts)
                if not (safe_retry and status in (429,500,502,503,504) and attempt < 2):
                    raise ApiError(f'{operation}: HTTP {status}')
                delay = min(10.0, max(.1, float(received_headers.get('retry-after', .25 * 2 ** attempt))))
            except (OSError, http.client.HTTPException) as exc:
                record['error'] = type(exc).__name__
                record['complete_ms'] = (time.perf_counter() - start) * 1000
                self.records.append(record)
                self.close()
                if not safe_retry or attempt == 2 or isinstance(exc, ssl.SSLCertVerificationError):
                    raise
                delay = .25 * 2 ** attempt
            except ApiError:
                record['error'] = 'ApiError'
                if record['complete_ms'] is None:
                    record['complete_ms'] = (time.perf_counter() - start) * 1000
                    self.records.append(record)
                self.close()
                raise
            self.sleep(delay)
        raise ApiError('Retry budget exhausted')

    def json(self, operation, method, path, payload=None, key=None, expect=(200,)):
        headers = {'Idempotency-Key':key} if key else None
        _, h, data = self.request(operation, method, path, payload=payload, headers=headers, expect=expect)
        return json.loads(data), h

    def upload(self, slot, path):
        stalls = 0
        previous = -1
        while True:
            _, head, _ = self.request('upload_head','HEAD',slot['upload_url'],
                                      headers={'Tus-Resumable':'1.0.0'})
            offset = int(head['upload-offset'])
            if offset > previous:
                stalls = 0
                previous = offset
            if not 0 <= offset <= slot['bytes'] or int(head['upload-length']) != slot['bytes']:
                raise ApiError('Invalid server upload offset/length')
            if offset == slot['bytes']:
                return
            with Path(path).open('rb') as f:
                f.seek(offset)
                chunk = f.read(min(1024 * 1024, slot['bytes'] - offset))
            try:
                status, h, _ = self.request('upload_patch','PATCH',slot['upload_url'],body=chunk,
                    headers={'Tus-Resumable':'1.0.0','Upload-Offset':str(offset),
                             'Content-Type':'application/offset+octet-stream'}, expect=(204,409))
                if status == 204 and int(h['upload-offset']) != offset + len(chunk):
                    raise ApiError('Upload acknowledged an incorrect offset')
                stalls = stalls + 1 if status == 409 else 0
            except (OSError, http.client.HTTPException):
                self.close()
                stalls += 1  # A lost PATCH response is reconciled with HEAD, never blindly appended.
            if stalls > 3:
                raise ApiError('Repeated upload conflicts/connection failures')

    def download(self, item, target):
        target = Path(target)
        if target.exists() and target.stat().st_size == item['bytes'] and digest(target) == item['sha256']:
            return True
        _, h, _ = self.request('download_head','HEAD',item['download_url'])
        etag = h.get('etag')
        if not etag or etag.startswith('W/') or int(h['content-length']) != item['bytes']:
            raise ApiError('Result needs a strong ETag and the declared byte count')
        partial, tag = target.with_suffix('.part'), target.with_suffix('.part.etag')
        if not tag.exists() or tag.read_text() != etag or (partial.exists() and partial.stat().st_size > item['bytes']):
            partial.unlink(missing_ok=True)
        tag.write_text(etag)
        for attempt in range(3):
            offset = partial.stat().st_size if partial.exists() else 0
            if offset == item['bytes']:
                break
            headers = {'Range':f'bytes={offset}-','If-Range':etag} if offset else {}
            try:
                status, rh, _ = self.request('download_body','GET',item['download_url'],
                    headers=headers, expect=(200,206), sink=partial, limit=item['bytes'])
                if rh.get('etag') != etag:
                    raise ApiError('Result ETag changed during download')
                if status == 206:
                    content_range = rh.get('content-range','')
                    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', content_range)
                    if not match or tuple(map(int,match.groups())) != (offset,item['bytes']-1,item['bytes']):
                        partial.unlink(missing_ok=True)
                        raise ApiError('Invalid resumed-download byte range')
            except (OSError,http.client.HTTPException):
                self.close()
                if attempt == 2:
                    raise
        if not partial.exists() or partial.stat().st_size != item['bytes'] or digest(partial) != item['sha256']:
            partial.unlink(missing_ok=True)
            raise ApiError('Result checksum/length mismatch; cache was not published')
        import os
        with partial.open('rb') as f:
            os.fsync(f.fileno())
        partial.replace(target)
        tag.unlink(missing_ok=True)
        return False


def poll(client, operation, path, wanted, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value, headers = client.json(operation,'GET',path)
        if value['state'] in wanted:
            return value
        if value['state'] in ('invalid','failed','cancelled','expired','deleted'):
            raise ApiError(f'{operation}: terminal state {value["state"]}')
        client.sleep(min(10,max(.1,float(headers.get('retry-after',2)))))
    raise TimeoutError(operation)


def run_job(client, fixture_path, profile_id, out, resume=False, timeout=600,
            purpose='full', frame_range=None, route_label='unspecified'):
    started = time.perf_counter()
    record_start = len(client.records)
    out = Path(out)
    out.mkdir(mode=0o700, parents=True, exist_ok=True)
    report = dict(status='running', fixture=None, resumed=resume,
                  started_at_utc=datetime.now(timezone.utc).isoformat(),route_label=route_label,
                  client=dict(system=platform.system(), machine=platform.machine(), python=platform.python_version()),
                  connection_policy='persistent HTTP/1.1, verified TLS or loopback SSH tunnel',
                  stage_seconds={}, server_timings=None)
    last = started
    def stage(name):
        nonlocal last
        now = time.perf_counter()
        report['stage_seconds'][name] = now - last
        last = now
    try:
        fixture = json.loads(Path(fixture_path).read_text())
        report['fixture'] = fixture['id']
        asset_spec = fixture['asset_request']
        selected = frame_range or {'start':0,'end':asset_spec['frame_count']}
        if not 0 <= selected['start'] < selected['end'] <= asset_spec['frame_count']:
            raise ValueError('Output range is outside the fixture')
        delivery = dict(purpose=purpose,output_range=selected)
        report['delivery'] = delivery
        for f in asset_spec['files']:
            path = Path(fixture['frame_dir']) / f'{f["frame_index"]:06d}.png'
            if path.stat().st_size != f['bytes'] or digest(path) != f['sha256']:
                raise ApiError('Local fixture changed')
        signature = hashlib.sha256(json.dumps(asset_spec,sort_keys=True).encode()).hexdigest()
        state_path = out/'state.json'
        if state_path.exists():
            if not resume:
                raise ApiError('Use a fresh output directory or explicitly resume')
            state = json.loads(state_path.read_text())
            if state['input_signature'] != signature or state['origin'] != client.origin or state['profile_id'] != profile_id:
                raise ApiError('Resume state belongs to another input/origin/profile')
            if state['delivery'] != delivery:
                raise ApiError('Resume state belongs to another preview/full range')
        else:
            state = dict(input_signature=signature,origin=client.origin,profile_id=profile_id,delivery=delivery,
                         keys={n:str(uuid.uuid4()) for n in ('asset','complete','job')})
            save(state_path,state)
        stage('local_input_verification')
        cap, _ = client.json('capabilities','GET','/v1/capabilities')
        profiles = [p for p in cap['profiles'] if p['id'] == profile_id]
        if len(profiles) != 1:
            raise ApiError('Requested profile is not advertised')
        profile = profiles[0]
        if 'asset_id' not in state:
            asset, _ = client.json('asset_create','POST','/v1/assets',asset_spec,state['keys']['asset'],(200,201))
            state['asset_id'] = asset['id']
            save(state_path,state)
        else:
            asset, _ = client.json('asset_get','GET','/v1/assets/'+state['asset_id'])
        stage('capabilities_and_asset_declaration')
        if asset['state'] == 'uploading':
            slots = {f['frame_index']:f for f in asset['files']}
            for f in asset_spec['files']:
                client.upload(slots[f['frame_index']],Path(fixture['frame_dir']) / f'{f["frame_index"]:06d}.png')
        stage('upload')
        asset_url = '/v1/assets/' + state['asset_id']
        client.json('asset_complete','POST',asset_url+'/complete',{},state['keys']['complete'],(200,202))
        ready_asset = poll(client,'asset_poll',asset_url,{'ready'},timeout)
        stage('asset_validation_wait')
        if 'job_request' not in state:
            state['job_request'] = dict(asset_id=state['asset_id'], capabilities_revision=cap['capabilities_revision'],
                model_revision=profile['model_revision'],profile_id=profile_id,purpose=purpose,
                output_range=selected, context={'mode':'entire_asset'},
                output=dict(format='exr_sequence',pixel_type='float32',compression='zip',channels=['alpha','fg'],
                            colour_space='linear_srgb',foreground_alpha='straight'))
            save(state_path,state)
        job, _ = client.json('job_submit','POST','/v1/jobs',state['job_request'],state['keys']['job'],(200,202))
        if state.get('job_id', job['id']) != job['id']:
            raise ApiError('Idempotent submission returned a different job ID')
        state['job_id'] = job['id']
        save(state_path,state)
        stage('job_submission')
        job_url = '/v1/jobs/' + state['job_id']
        completed = poll(client,'job_poll',job_url,{'succeeded'},timeout)
        report['server_timings'] = completed.get('server_timings')
        stage('queue_processing_and_poll_detection')
        result, _ = client.json('result_manifest','GET',job_url+'/result')
        recipe = result['recipe']
        if (result['job_id'] != job['id'] or recipe['profile_id'] != profile_id
                or recipe['model_revision'] != state['job_request']['model_revision']
                or recipe['execution_fingerprint'] != profile['execution_fingerprint']
                or recipe['asset_content_sha256'] != ready_asset['content_sha256']
                or recipe['processed_range'] != {'start':0,'end':asset_spec['frame_count']}
                or recipe['output_range'] != selected or recipe['output'] != state['job_request']['output']
                or result['timing'] != asset_spec['timing']):
            raise ApiError('Result recipe/timing differs from the requested exact-output contract')
        expected = {(i,o) for i in range(selected['start'],selected['end']) for o in ('alpha','fg')}
        actual = {(f['source_frame'],f['output']) for f in result['files']}
        if expected != actual or len(actual) != len(result['files']):
            raise ApiError('Incomplete/duplicate result frame mapping')
        cache = out/'cache'
        cache.mkdir(mode=0o700,exist_ok=True)
        cached, paths = 0, []
        for f in result['files']:
            if not re.fullmatch(r'[0-9a-f]{64}',f['sha256']):
                raise ApiError('Invalid result digest')
            alpha = f['output'] == 'alpha'
            expected_pts = int(asset_spec['timing']['start_pts']) + f['source_frame'] * int(asset_spec['timing']['frame_duration_ticks'])
            if (f['width'] != asset_spec['width'] or f['height'] != asset_spec['height']
                    or f['pixel_type'] != 'float32'
                    or f['channels'] != (['A'] if alpha else ['R','G','B'])
                    or f['colour_space'] != ('coverage' if alpha else 'linear_srgb')
                    or f['alpha_association'] != ('coverage' if alpha else 'straight')
                    or int(f['pts']) != expected_pts
                    or f['duration_ticks'] != asset_spec['timing']['frame_duration_ticks']):
                raise ApiError('Result frame metadata differs from the requested media contract')
            path = cache/f'{f["source_frame"]:06d}_{f["output"]}.exr'
            cached += client.download(f,path)
            paths.append(dict(path=str(path.resolve()),sha256=f['sha256'],bytes=f['bytes']))
        save(cache/'manifest.json',dict(result=result,local_files=paths))
        report['cache_hits'] = cached
        report['result_files'] = len(paths)
        stage('result_manifest_download_and_local_verification')
        report['status'] = 'succeeded'
        return report
    except Exception as exc:
        report.update(status='failed',error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        report['end_to_end_seconds'] = time.perf_counter() - started
        report['requests'] = client.records[record_start:]
        report['request_summary'] = summaries(report['requests'])
        save(out/'report.json',report)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base-url',required=True)
    ap.add_argument('--token-file',type=Path,required=True)
    ap.add_argument('--route-label',default='unspecified',help='Describe the actual client-to-server route for the report.')
    sub = ap.add_subparsers(dest='command',required=True)
    probe = sub.add_parser('probe')
    probe.add_argument('--count',type=int,default=50)
    probe.add_argument('--output',type=Path,required=True)
    run = sub.add_parser('run')
    run.add_argument('--fixture',type=Path,required=True)
    run.add_argument('--profile-id',required=True)
    run.add_argument('--output',type=Path,required=True)
    run.add_argument('--resume',action='store_true')
    run.add_argument('--purpose',choices=('preview','full'),default='full')
    run.add_argument('--frames',help='Output presentation range START:END; full fixture remains the context.')
    args = ap.parse_args()
    if args.token_file.stat().st_mode & 0o077:
        ap.error('Token file permissions must be 0600 or stricter')
    token = args.token_file.read_text().strip()
    if not token or '\r' in token or '\n' in token:
        ap.error('Token file needs one non-empty service token')
    client = Client(args.base_url,token)
    try:
        if args.command == 'probe':
            if args.count < 1:
                ap.error('--count must be positive')
            for _ in range(args.count):
                client.json('capabilities','GET','/v1/capabilities')
            report = dict(kind='request_probe',origin=client.origin,requests=client.records,
                          route_label=args.route_label,
                          summary=summaries(client.records))
            save(args.output,report)
            print(json.dumps(report['summary'],indent=2))
        else:
            selected = None
            if args.frames:
                if not re.fullmatch(r'\d+:\d+',args.frames):
                    ap.error('--frames needs START:END')
                start,end = map(int,args.frames.split(':'))
                selected = {'start':start,'end':end}
            report = run_job(client,args.fixture,args.profile_id,args.output,args.resume,
                             purpose=args.purpose,frame_range=selected,route_label=args.route_label)
            print(json.dumps({k:v for k,v in report.items() if k not in ('requests','request_summary')},indent=2))
    finally:
        client.close()


if __name__ == '__main__':
    main()
