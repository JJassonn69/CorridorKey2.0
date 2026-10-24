"""Real loopback HTTP transport tests; no inference or performance claims."""
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import tempfile
import threading
import unittest

spec = importlib.util.spec_from_file_location('api_benchmark_client',
    Path(__file__).resolve().parents[1] / 'benchmarks/api/client.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *args):
        pass

    def send(self, status=200, value=None, headers=None, raw=None):
        body = raw if raw is not None else (json.dumps(value).encode() if value is not None else b'')
        self.send_response(status)
        self.send_header('Content-Length', str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, str(v))
        self.end_headers()
        if self.command != 'HEAD':
            self.wfile.write(body)

    def route(self):
        self.body = self.rfile.read(int(self.headers.get('Content-Length',0)))
        self.server.calls.append((self.command,self.path,dict(self.headers),self.body))
        self.server.callback(self)

    do_GET = do_HEAD = do_PATCH = do_POST = route


class BenchmarkClientTests(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.server.daemon_threads = True
        self.server.calls = []
        self.server.callback = lambda h:h.send(value={})
        self.thread = threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()
        self.client = module.Client(f'http://127.0.0.1:{self.server.server_port}',
                                    'disposable-test-token',sleep=lambda _:None)

    def tearDown(self):
        self.client.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_idempotent_post_retry_preserves_body_and_key(self):
        def route(h):
            h.send(503 if len(self.server.calls)==1 else 202, {'id':'same-job'}, {'Retry-After':'0.1'})
        self.server.callback = route
        value,_=self.client.json('submit','POST','/v1/jobs',{'input':'fixture'},'same-key',expect=(202,))
        self.assertEqual(value['id'],'same-job')
        self.assertEqual([c[3] for c in self.server.calls], [b'{"input":"fixture"}']*2)
        self.assertEqual([c[2]['Idempotency-Key'] for c in self.server.calls], ['same-key']*2)
        self.assertEqual(len(self.client.records),2)
        self.assertEqual(module.summaries(self.client.records)['submit']['errors'],1)

    def test_lost_patch_reply_resumes_without_duplicate_bytes(self):
        content=b'private fixture bytes'*100
        uploaded=bytearray()
        def route(h):
            if h.command=='HEAD':
                h.send(headers={'Upload-Offset':len(uploaded),'Upload-Length':len(content)})
            else:
                self.assertEqual(int(h.headers['Upload-Offset']),len(uploaded))
                uploaded.extend(h.body)
                h.close_connection=True
                h.connection.shutdown(socket.SHUT_RDWR)
                h.connection.close()
        self.server.callback=route
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'input';path.write_bytes(content)
            self.client.upload({'upload_url':'/v1/uploads/x','bytes':len(content)},path)
        self.assertEqual(bytes(uploaded),content)
        self.assertEqual(sum(c[0]=='PATCH' for c in self.server.calls),1)

    def test_resumed_result_download_and_local_cache(self):
        data=b'fake EXR transport payload, not an image'
        def route(h):
            if h.command=='HEAD':
                h.send(raw=data,headers={'ETag':'"stable"'})
            else:
                self.assertEqual(h.headers['Range'],'bytes=9-')
                self.assertEqual(h.headers['If-Range'],'"stable"')
                h.send(206,raw=data[9:],headers={'ETag':'"stable"','Content-Range':f'bytes 9-{len(data)-1}/{len(data)}'})
        self.server.callback=route
        item={'download_url':'/v1/jobs/j/files/f','bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'result.exr'
            target.with_suffix('.part').write_bytes(data[:9])
            target.with_suffix('.part.etag').write_text('"stable"')
            self.assertFalse(self.client.download(item,target))
            self.assertEqual(target.read_bytes(),data)
            count=len(self.server.calls)
            self.assertTrue(self.client.download(item,target))
            self.assertEqual(len(self.server.calls),count)

    def test_corrupt_download_never_publishes_cache(self):
        self.server.callback=lambda h:h.send(raw=b'bad',headers={'ETag':'"stable"'})
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bad.exr'
            with self.assertRaisesRegex(module.ApiError,'checksum'):
                self.client.download({'download_url':'/file','bytes':3,'sha256':'0'*64},path)
            self.assertFalse(path.exists())
            self.assertFalse(path.with_suffix('.part').exists())

    def test_origin_and_redirect_protection(self):
        with self.assertRaises(ValueError):
            self.client.request('x','GET','https://other.example.invalid/secret')
        self.assertFalse(self.server.calls)
        self.server.callback=lambda h:h.send(302,headers={'Location':'https://other.example.invalid/'})
        with self.assertRaises(module.ApiError):
            self.client.request('x','GET','/redirect')
        self.assertEqual(len(self.server.calls),1)
        self.assertEqual(module.summaries(self.client.records)['x']['errors'],1)

    def test_small_samples_do_not_claim_p95(self):
        self.client.json('probe','GET','/capabilities')
        self.assertIsNone(module.summaries(self.client.records)['probe']['p95_ms'])
        for _ in range(19):
            self.client.json('probe','GET','/capabilities')
        self.assertIsNotNone(module.summaries(self.client.records)['probe']['p95_ms'])
        self.assertTrue(all(r['complete_ms']>=r['headers_ms']>=0 for r in self.client.records))

    def test_complete_client_workflow_and_resume(self):
        media=b'fake PNG bytes for transport testing'
        exr=b'fake EXR bytes for transport testing'
        state={'uploaded':bytearray(),'ready':False,'submitted':False,'request':None,'bad_replay':False}
        timing={'mode':'cfr','frame_rate':{'num':24,'den':1},'time_base':{'num':1,'den':24},
                'start_pts':'9','frame_duration_ticks':'1'}
        def route(h):
            if h.path=='/v1/capabilities':
                h.send(value={'capabilities_revision':'test','profiles':[{'id':'p','model_revision':'m','execution_fingerprint':'c'*64}]})
            elif h.path=='/v1/assets' or h.path=='/v1/assets/a':
                h.send(201 if h.command=='POST' else 200,{'id':'a','state':'ready' if state['ready'] else 'uploading',
                    'content_sha256':'d'*64,
                    'files':[{'frame_index':0,'upload_url':'/v1/uploads/u','bytes':len(media)}]})
            elif h.path=='/v1/uploads/u':
                if h.command=='PATCH':state['uploaded'].extend(h.body)
                h.send(204 if h.command=='PATCH' else 200,headers={'Upload-Offset':len(state['uploaded']),'Upload-Length':len(media)})
            elif h.path=='/v1/assets/a/complete':
                state['ready']=True;h.send(202,{'id':'a','state':'ready'})
            elif h.path=='/v1/jobs':
                state['submitted']=True;state['request']=json.loads(h.body)
                h.send(202,{'id':'wrong' if state['bad_replay'] else 'j'})
            elif h.path=='/v1/jobs/j':
                h.send(value={'id':'j','state':'succeeded','server_timings':{'queue_seconds':1,'service_seconds':2}})
            elif h.path=='/v1/jobs/j/result':
                h.send(value={'job_id':'j','timing':timing,'recipe':{'profile_id':'p','model_revision':'m',
                    'execution_fingerprint':'c'*64,'asset_content_sha256':'d'*64,
                    'processed_range':{'start':0,'end':1},'output_range':state['request']['output_range'],
                    'output':state['request']['output']},'files':[{'source_frame':0,'output':o,'bytes':len(exr),
                    'width':1,'height':1,'pixel_type':'float32','channels':['Y'] if o=='alpha' else ['R','G','B'],
                    'colour_space':'coverage' if o=='alpha' else 'linear_srgb','alpha_association':'coverage' if o=='alpha' else 'straight',
                    'pts':'9','duration_ticks':'1',
                    'sha256':hashlib.sha256(exr).hexdigest(),'download_url':'/file/'+o} for o in ('alpha','fg')]})
            elif h.path.startswith('/file/'):
                h.send(raw=exr,headers={'ETag':'"stable"'})
            else:
                h.send(404)
        self.server.callback=route
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'000000.png').write_bytes(media)
            fixture={'id':'transport-test','frame_dir':str(root),'asset_request':{'frame_count':1,'width':1,'height':1,'timing':timing,
                'files':[{'frame_index':0,'bytes':len(media),'sha256':hashlib.sha256(media).hexdigest()}]}}
            path=root/'fixture.json';path.write_text(json.dumps(fixture))
            first=module.run_job(self.client,path,'p',root/'run')
            self.assertEqual(first['result_files'],2)
            self.assertEqual(first['server_timings']['queue_seconds'],1)
            second=module.run_job(self.client,path,'p',root/'run',resume=True)
            self.assertEqual(second['cache_hits'],2)
            self.assertEqual(sum(c[0]=='POST' and c[1]=='/v1/assets' for c in self.server.calls),1)
            self.assertEqual(bytes(state['uploaded']),media)
            report=json.loads((root/'run/report.json').read_text())
            self.assertNotIn('disposable-test-token',json.dumps(report))
            self.assertEqual(report['status'],'succeeded')
            state['bad_replay']=True
            with self.assertRaisesRegex(module.ApiError,'different job ID'):
                module.run_job(self.client,path,'p',root/'run',resume=True)


if __name__=='__main__':
    unittest.main()
