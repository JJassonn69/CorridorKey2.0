"""Editor input contracts, cache identity, constraints and decoding without CUDA."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import queue
import time
from types import SimpleNamespace
import unittest

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import cv2
import numpy as np
from PIL import Image

import test_api_service as base
from service.common import digest
from service.media import apply_constraints,exr_header,validate_file


class EditorTests(unittest.TestCase):
    setUp=base.ServiceTests.setUp
    tearDown=base.ServiceTests.tearDown
    call=base.ServiceTests.call
    request=base.ServiceTests.request

    def upload(self,body,files):
        key=str(time.monotonic_ns())
        a=self.call('POST','/v1/assets',body,key,(201,))
        for slot in a['files']:self.client.upload(slot,files[slot['frame_index']])
        self.call('POST','/v1/assets/'+a['id']+'/complete',{},key,(202,))
        for _ in range(200):
            a=self.call('GET','/v1/assets/'+a['id'])
            if a['state'] in ('ready','invalid'):return a
            time.sleep(.01)
        self.fail('Validation did not finish')

    def source_asset(self,count=4):
        b=copy.deepcopy(self.body);b['frame_count']=count
        b['files']=[dict(b['files'][0],frame_index=i) for i in range(count)]
        return self.upload(b,{i:self.source_file for i in range(count)})

    @property
    def source_file(self):return self.root/'input.png'

    def matte(self,indices,count=4,value=96,depth=8,start=None):
        b=copy.deepcopy(self.body);b.update(kind='matte_sequence',frame_count=count,
            colour=dict(interpretation='coverage',pixel_type='uint8' if depth==8 else 'uint16'))
        if start is not None:b['timing']['start_pts']=start
        image=np.full((3,4),value,dtype=np.uint8 if depth==8 else np.uint16)
        p=self.root/f'matte-{time.monotonic_ns()}.png';Image.fromarray(image).save(p)
        b['files']=[dict(frame_index=i,bytes=p.stat().st_size,sha256=digest(p),media_type='image/png') for i in indices]
        return self.upload(b,{i:p for i in indices})

    def job(self,source,**extras):
        r=self.request(source);r['output_range']['end']=source['manifest']['frame_count'];r.update(extras)
        return r

    def submit(self,request,status=202):
        return self.call('POST','/v1/jobs',request,str(time.monotonic_ns()),(status,))

    def cancel(self,job):self.call('POST','/v1/jobs/'+job['id']+'/cancel',{},str(time.monotonic_ns()),(200,))

    def test_sparse_alignment_confidence_and_missing_frames(self):
        source=self.source_asset();hint=self.matte([0,2],depth=16,value=32768)
        self.assertEqual(hint['state'],'ready')
        j=self.submit(self.job(source,hint=dict(asset_id=hint['id'],mode='keyframes',confidence=.8,
            frame_confidence=[dict(frame_index=2,confidence=.25)]),carry=dict(enabled=False,confidence=.6)))
        self.assertEqual(j['recipe']['guidance']['hint']['frames'],[dict(frame_index=0,confidence=.8),dict(frame_index=2,confidence=.25)])
        private=self.c.store.get('job',j['id'],'a')
        self.assertEqual(set(private['_work']['hint']['frames']),{'0','2'})
        self.assertEqual(len(list((self.c.asset_dir(hint['id'])/'frames').glob('*'))),2)
        self.cancel(j)
        self.submit(self.job(source,hint=dict(asset_id=hint['id'],mode='sequence')),422)
        self.submit(self.job(source,hint=dict(asset_id=hint['id'],mode='first_frame')),422)
        self.submit(self.job(source,hint=dict(asset_id=hint['id'],mode='keyframes',
            frame_confidence=[dict(frame_index=1,confidence=.5)])),422)
        shifted=self.matte([0],start='49000')
        self.submit(self.job(source,hint=dict(asset_id=shifted['id'],mode='first_frame')),422)

    def test_cache_identity_covers_every_effective_input(self):
        source=self.source_asset();h1=self.matte([0]);h2=self.matte([0],value=128)
        first=self.job(source,hint=dict(asset_id=h1['id'],mode='first_frame'),
            masks=[dict(asset_id=h1['id'],mode='static',role='force_background',label='garbage')])
        def signature(r):
            j=self.submit(r);self.cancel(j);return j['recipe']['processing_fingerprint']
        original=signature(first)
        labelled=copy.deepcopy(first);labelled['masks'][0]['label']='holdout'
        self.assertEqual(signature(labelled),original)
        for label in ('hint','confidence','carry','priority','mask','role','range'):
            r=copy.deepcopy(first)
            if label=='hint':r['hint']['asset_id']=h2['id']
            elif label=='confidence':r['hint']['confidence']=.5
            elif label=='carry':r['carry']=dict(enabled=False)
            elif label=='priority':r['carry']=dict(precedence='carried')
            elif label=='mask':r['masks'][0]['asset_id']=h2['id']
            elif label=='role':r['masks'][0]['role']='force_foreground'
            else:r['output_range']=dict(start=1,end=3)
            self.assertNotEqual(signature(r),original,label)

    def test_guidance_is_owner_checked_pinned_and_cascades(self):
        source=self.source_asset();hint=self.matte([0])
        j=self.submit(self.job(source,hint=dict(asset_id=hint['id'],mode='first_frame')))
        with self.c.lock:
            doc=self.c.store.get('asset',hint['id'],'a');doc['_expires']=time.time()-1
            self.c.store.put('asset','a',doc);self.c.cleanup()
            self.assertEqual(self.c.store.get('asset',hint['id'],'a')['state'],'ready')
        d=self.call('DELETE','/v1/assets/'+hint['id'],expect=(202,))
        for _ in range(100):
            value=self.call('GET',d['status_url'])
            if value['state']=='deleted':break
            time.sleep(.01)
        self.assertEqual(value['state'],'deleted')
        self.assertEqual(self.call('GET','/v1/jobs/'+j['id'])['state'],'deleted')
        self.assertEqual(self.call('GET','/v1/assets/'+source['id'])['state'],'ready')
        self.client.token='b';other=self.matte([0]);self.client.token='a'
        self.submit(self.job(source,hint=dict(asset_id=other['id'],mode='first_frame')),404)

    def test_linear_exr_half_float_hdr_and_finite_validation(self):
        for pixel,code in [('float16',cv2.IMWRITE_EXR_TYPE_HALF),('float32',cv2.IMWRITE_EXR_TYPE_FLOAT)]:
            p=self.root/(pixel+'.exr');values=np.full((3,4,3),[-.25,2,4],np.float32)
            cv2.imwrite(str(p),values,[cv2.IMWRITE_EXR_TYPE,code])
            b=copy.deepcopy(self.body);b['colour'].update(transfer='linear',pixel_type=pixel)
            b['files']=[dict(frame_index=0,bytes=p.stat().st_size,sha256=digest(p),media_type='image/x-exr')]
            a=self.upload(b,{0:p});self.assertEqual(a['state'],'ready')
            self.assertEqual(set(exr_header(p)['channels'].values()),{pixel})
            np.testing.assert_array_equal(cv2.imread(str(p),cv2.IMREAD_UNCHANGED),values)
        values[0,0,0]=np.nan;cv2.imwrite(str(p),values,[cv2.IMWRITE_EXR_TYPE,cv2.IMWRITE_EXR_TYPE_FLOAT])
        b['files'][0].update(bytes=p.stat().st_size,sha256=digest(p))
        self.assertEqual(self.upload(b,{0:p})['state'],'invalid')

    def test_constraints_are_explicit_exclusion_wins_and_rgb_is_untouched(self):
        out=self.root/'out';alpha_dir=out/'source_alpha_exr';fg_dir=out/'source_fg_exr'
        alpha_dir.mkdir(parents=True);fg_dir.mkdir()
        alpha=np.array([[.2,.8,.4],[.5,0,1]],np.float32)
        p=alpha_dir/'alpha_000000.exr';cv2.imwrite(str(p),alpha)
        fg=fg_dir/'fg_000000.exr';cv2.imwrite(str(fg),np.full((2,3,3),2.5,np.float32));fg_hash=digest(fg)
        keep=self.root/'keep.png';exclude=self.root/'exclude.png'
        cv2.imwrite(str(keep),np.full((2,3),255,np.uint8));cv2.imwrite(str(exclude),np.array([[0,255,128],[0,0,255]],np.uint8))
        apply_constraints(out,[dict(role='force_foreground',mode='static',frames={0:str(keep)}),
                               dict(role='force_background',mode='static',frames={0:str(exclude)})])
        expected=np.float32(1)-cv2.imread(str(exclude),cv2.IMREAD_UNCHANGED).astype(np.float32)/np.float32(255)
        np.testing.assert_array_equal(cv2.imread(str(p),cv2.IMREAD_UNCHANGED),expected)
        self.assertEqual(digest(fg),fg_hash)
        before=digest(p);zero=self.root/'zero.png';cv2.imwrite(str(zero),np.zeros((2,3),np.uint8))
        apply_constraints(out,[dict(role='force_background',mode='static',frames={0:str(zero)})])
        self.assertEqual(digest(p),before)

    def test_coverage_exr_is_single_channel_and_bounded(self):
        for pixel,code in [('float16',cv2.IMWRITE_EXR_TYPE_HALF),('float32',cv2.IMWRITE_EXR_TYPE_FLOAT)]:
            p=self.root/('coverage-'+pixel+'.exr')
            cv2.imwrite(str(p),np.full((3,4),.5,np.float32),[cv2.IMWRITE_EXR_TYPE,code])
            b=copy.deepcopy(self.body)
            b.update(kind='matte_sequence',colour=dict(interpretation='coverage',pixel_type=pixel))
            b['files']=[dict(frame_index=0,bytes=p.stat().st_size,sha256=digest(p),media_type='image/x-exr')]
            self.assertEqual(self.upload(b,{0:p})['state'],'ready')
            for invalid in (np.full((3,4),1.5,np.float32),np.full((3,4,3),.5,np.float32)):
                cv2.imwrite(str(p),invalid,[cv2.IMWRITE_EXR_TYPE,code])
                b['files'][0].update(bytes=p.stat().st_size,sha256=digest(p))
                self.assertEqual(self.upload(b,{0:p})['state'],'invalid')

    def test_large_profile_cannot_overlap_a_second_gpu_job(self):
        self.c.stop.set();self.c.thread.join(timeout=2)
        self.c.cap['profiles'][0]['max_parallel_jobs']=2
        self.c.cap['profiles'].append(dict(self.c.cap['profiles'][0],id='large',max_parallel_jobs=1))
        a=self.source_asset();large=self.job(a);large['profile_id']='large';j1=self.submit(large)
        self.client.token='b';b=self.source_asset();j2=self.submit(self.job(b))
        slots=[dict(process=SimpleNamespace(is_alive=lambda:True,pid=123456789+i),events=queue.Queue(),inbox=queue.Queue(),
            ready=True,job=None,number=i,boot=time.monotonic()) for i in range(2)]
        self.c.slots=slots
        try:
            with self.c.lock:
                self.c.tick()
                self.assertEqual([s['job'] for s in slots],[j1['id'],None])
                self.assertEqual(self.c.store.get('job',j2['id'],'b')['state'],'queued')
                finished=self.c.store.get('job',j1['id'],'a');finished['state']='cancelled'
                self.c.store.put('job','a',finished);slots[0]['job']=None
                self.c.tick()
                self.assertEqual(slots[0]['job'],j2['id'])
        finally:self.c.slots=[]


if __name__=='__main__':unittest.main()
