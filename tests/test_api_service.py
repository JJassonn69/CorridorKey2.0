"""Exercise the actual controller/HTTP boundary without importing CUDA."""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import queue
from types import SimpleNamespace
import unittest

from PIL import Image
from service.common import save
from service.server import Controller, Server

spec=importlib.util.spec_from_file_location('client',Path(__file__).resolve().parents[1]/'benchmarks/api/client.py')
client_module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(client_module)


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        caps=dict(capabilities_revision='test',profiles=[dict(id='p',model_revision='m',execution_fingerprint='a'*64,
            max_width=1280,max_height=720,max_context_frames=72)],limits=dict(max_active_gpu_jobs=1,
            max_pending_jobs=8,max_pending_jobs_per_user=2,max_patch_bytes=1<<20,max_asset_bytes=1<<30,
            user_storage_quota_bytes=4<<30,max_queue_age_seconds=600,max_run_seconds=600),
            retention=dict(incomplete_upload_seconds=86400,ready_asset_seconds=86400,result_seconds=86400))
        save(self.root/'caps.json',caps)
        self.c=Controller(dict(data=str(self.root/'data'),capabilities=str(self.root/'caps.json'),workers=1,
            disk_floor_bytes=0),start_workers=False)
        with self.c.store.db:
            for owner in ('a','b'):
                self.c.store.db.execute('INSERT INTO tokens VALUES(?,?,?)',
                    (hashlib.sha256(owner.encode()).hexdigest(),owner,time.time()+100))
        self.server=Server(('127.0.0.1',0),self.c)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()
        self.client=client_module.Client('http://127.0.0.1:'+str(self.server.server_port),'a',sleep=lambda _:time.sleep(.01))
        buf=io.BytesIO(); Image.new('RGB',(4,3),(20,180,40)).save(buf,format='PNG')
        self.png=buf.getvalue()
        self.source=self.root/'input.png';self.source.write_bytes(self.png)
        self.body=json.loads((Path(__file__).resolve().parents[1]/'docs/service/examples.json').read_text())['asset_create']
        self.body.update(width=4,height=3,frame_count=1,files=[dict(frame_index=0,bytes=len(self.png),
            sha256=hashlib.sha256(self.png).hexdigest(),media_type='image/png')])

    def tearDown(self):
        self.client.close();self.server.shutdown();self.server.server_close();self.thread.join()
        self.c.close();self.tmp.cleanup()

    def call(self,method,path,body=None,key=None,expect=(200,)):
        return self.client.json('test',method,path,body,key,expect)[0]

    def asset(self,ready=True,key='asset'):
        a=self.call('POST','/v1/assets',self.body,key,(201,))
        if ready:
            self.client.upload(a['files'][0],self.source)
            self.call('POST','/v1/assets/'+a['id']+'/complete',{},'complete',(202,))
            for _ in range(100):
                a=self.call('GET','/v1/assets/'+a['id'])
                if a['state']=='ready':break
                time.sleep(.01)
            self.assertEqual(a['state'],'ready')
        return a

    def request(self,a):
        body=json.loads((Path(__file__).resolve().parents[1]/'docs/service/examples.json').read_text())['job_create']
        body.update(asset_id=a['id'],capabilities_revision='test',model_revision='m',profile_id='p',output_range=dict(start=0,end=1))
        return body

    def test_idempotency_ownership_and_png_validation(self):
        a=self.asset()
        replay=self.call('POST','/v1/assets',self.body,'asset')
        self.assertEqual(a['id'],replay['id'])
        changed=copy.deepcopy(self.body);changed['width']=5
        self.call('POST','/v1/assets',changed,'asset',(409,))
        self.client.token='b'
        self.call('GET','/v1/assets/'+a['id'],expect=(404,))
        self.client.token='invalid'
        self.call('GET','/v1/capabilities',expect=(401,))

    def test_partial_upload_offset_conflict_freeze(self):
        a=self.asset(False);url=a['files'][0]['upload_url']
        headers={'Tus-Resumable':'1.0.0','Content-Type':'application/offset+octet-stream','Upload-Offset':'0'}
        self.client.request('partial','PATCH',url,body=self.png[:9],headers=headers,expect=(204,))
        _,response,_=self.client.request('conflict','PATCH',url,body=self.png[:9],headers=headers,expect=(409,))
        self.assertEqual(response['connection'],'close')
        headers['Upload-Offset']='9'
        self.client.request('continue_after_rejection','PATCH',url,body=self.png[9:],headers=headers,expect=(204,))
        self.call('POST','/v1/assets/'+a['id']+'/complete',{},'complete',(202,))
        self.client.request('frozen','PATCH',url,body=b'x',headers=headers,expect=(409,))
        self.assertEqual((self.c.asset_dir(a['id'])/'frames/000000.png').read_bytes(),self.png)

    def test_cancel_retry_queue_bound_and_delete_cascade(self):
        a=self.asset();body=self.request(a)
        j=self.call('POST','/v1/jobs',body,'j1',(202,))
        self.assertEqual(j['id'],self.call('POST','/v1/jobs',body,'j1')['id'])
        self.call('POST','/v1/jobs',body,'j2',(202,))
        self.call('POST','/v1/jobs',body,'j3',(429,))
        cancelled=self.call('POST','/v1/jobs/'+j['id']+'/cancel',{},'cancel',(200,))
        self.assertEqual(cancelled['state'],'cancelled')
        retry=self.call('POST','/v1/jobs/'+j['id']+'/retry',{},'retry',(202,))
        self.assertEqual(retry['retry_of'],j['id'])
        deletion=self.call('DELETE','/v1/assets/'+a['id'],expect=(202,))
        for _ in range(100):
            d=self.call('GET',deletion['status_url'])
            if d['state']=='deleted':break
            time.sleep(.02)
        self.assertEqual(d['state'],'deleted')
        self.assertFalse(self.c.asset_dir(a['id']).exists())
        self.call('POST','/v1/assets',self.body,'asset',(410,))

    def test_expiry_and_invalid_media_fail_closed(self):
        a=self.asset(False)
        with self.c.lock:
            a=self.c.store.get('asset',a['id'],'a');a['_expires']=time.time()-1
            self.c.store.put('asset','a',a);self.c.cleanup()
        self.assertEqual(self.call('GET','/v1/assets/'+a['id'])['state'],'expired')
        self.client.request('expired','HEAD',a['files'][0]['upload_url'],
                            headers={'Tus-Resumable':'1.0.0'},expect=(410,))
        self.body['files'][0]['sha256']='0'*64
        a=self.asset(False,'invalid');self.client.upload(a['files'][0],self.source)
        self.call('POST','/v1/assets/'+a['id']+'/complete',{},'invalid-complete',(202,))
        for _ in range(100):
            a=self.call('GET','/v1/assets/'+a['id'])
            if a['state']=='invalid':break
            time.sleep(.01)
        self.assertEqual(a['state'],'invalid')

    def test_limits_colour_and_cfr_are_validated_before_storage(self):
        for change in ('width','colour','timestamp','cfr'):
            body=copy.deepcopy(self.body)
            if change=='width':body['width']=1281
            elif change=='colour':body['colour']['transfer']='log'
            elif change=='timestamp':body['timing']['start_pts']='9'*100
            else:body['timing']['frame_duration_ticks']='999'
            self.call('POST','/v1/assets',body,change,(422,))
        self.assertEqual(self.c.store.all('asset'),[])

    def test_real_download_range_resume_and_revocation(self):
        a=self.asset();body=self.request(a)
        j=self.call('POST','/v1/jobs',body,'job',(202,))
        payload=b'private transport fixture; GPU EXR exactness is a separate gate'*100
        h=hashlib.sha256(payload).hexdigest()
        item=dict(id='f',bytes=len(payload),sha256=h,download_url=f'/v1/jobs/{j["id"]}/files/f',_path='f.exr')
        path=self.c.job_dir(j['id'])/'published/f.exr';path.parent.mkdir(parents=True);path.write_bytes(payload)
        with self.c.lock:
            j.update(state='succeeded',_expires=time.time()+100,_result=dict(files=[item]))
            self.c.store.put('job','a',j)
        target=self.root/'cached.exr'
        target.with_suffix('.part').write_bytes(payload[:73])
        target.with_suffix('.part.etag').write_text('"'+h+'"')
        self.assertFalse(self.client.download(item,target))
        self.assertEqual(target.read_bytes(),payload)
        self.assertTrue(any(r['status']==206 for r in self.client.records))
        self.call('DELETE','/v1/jobs/'+j['id'],expect=(202,))
        self.client.request('revoked','GET',item['download_url'],expect=(410,))
        before=len(self.client.records)
        self.assertTrue(self.client.download(item,target))
        self.assertEqual(len(self.client.records),before)

    def test_round_robin_three_users_fifo(self):
        self.c.stop.set();self.c.thread.join(timeout=2)
        with self.c.store.db:
            self.c.store.db.execute('INSERT INTO tokens VALUES(?,?,?)',
                (hashlib.sha256(b'c').hexdigest(),'c',time.time()+100))
        for owner in ('a','b','c'):
            self.client.token=owner
            a=self.asset(key='asset-'+owner)
            for i in range(2):self.call('POST','/v1/jobs',self.request(a),owner+str(i),(202,))
        slot=dict(process=SimpleNamespace(is_alive=lambda:True,pid=123456789),events=queue.Queue(),inbox=queue.Queue(),
                  ready=True,job=None,number=0,boot=time.monotonic())
        self.c.slots=[slot]
        order=[]
        try:
            for _ in range(6):
                with self.c.lock:
                    self.c.tick();order.append(slot['owner'])
                    j=self.c.store.get('job',slot['job'],slot['owner'])
                    j['state']='cancelled';self.c.store.put('job',slot['owner'],j);slot['job']=None
            self.assertEqual(order,['a','b','c','a','b','c'])
        finally:self.c.slots=[]


if __name__=='__main__':unittest.main()
