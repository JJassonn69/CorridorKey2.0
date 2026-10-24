"""Private GPU pilot checks through the same Mac HTTP client."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import time
import uuid

spec=importlib.util.spec_from_file_location('client',Path(__file__).with_name('client.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tokens',nargs=2,type=Path,required=True)
    ap.add_argument('--fixture',type=Path,required=True)
    ap.add_argument('--state',type=Path,required=True)
    ap.add_argument('--reference',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    args.output.mkdir(mode=0o700,parents=True,exist_ok=False)
    c=m.Client('http://127.0.0.1:18080',args.tokens[0].read_text().strip())
    other=m.Client(c.origin,args.tokens[1].read_text().strip())
    fixture=json.loads(args.fixture.read_text());state=json.loads(args.state.read_text())
    reference=json.loads(args.reference.read_text())
    expected={Path(f['file']).name:f['sha256'] for f in reference['rendered']}
    checks={}
    def call(method,path,body=None,expect=(200,),key=None):
        return c.json('lifecycle',method,path,body,key or (str(uuid.uuid4()) if method=='POST' else None),expect)[0]
    def wait(path,states,timeout=600):
        return m.poll(c,'lifecycle_poll',path,states,timeout)
    try:
        invalid=m.Client(c.origin,'invalid')
        invalid.json('auth','GET','/v1/capabilities',expect=(401,));invalid.close()
        other.json('isolation','GET','/v1/assets/'+state['asset_id'],expect=(404,))
        checks['auth_and_owner_isolation']=True
        body=copy.deepcopy(fixture['asset_request']);body['frame_count']=1;body['files']=body['files'][:1]
        key=str(uuid.uuid4())
        asset=call('POST','/v1/assets',body,(201,),key)
        replay=call('POST','/v1/assets',body,(200,),key)
        assert asset['id']==replay['id']
        changed=copy.deepcopy(body);changed['width']+=1
        call('POST','/v1/assets',changed,(409,),key)
        slot=asset['files'][0];upload=slot['upload_url']
        source=Path(fixture['frame_dir'])/'000000.png'
        headers={'Tus-Resumable':'1.0.0','Content-Type':'application/offset+octet-stream','Upload-Offset':'0'}
        c.request('partial_upload','PATCH',upload,body=source.read_bytes()[:137],headers=headers,expect=(204,))
        c.request('stale_offset','PATCH',upload,body=b'x',headers=headers,expect=(409,))
        other.request('upload_isolation','HEAD',upload,headers={'Tus-Resumable':'1.0.0'},expect=(404,))
        c.upload(slot,source)
        call('POST','/v1/assets/'+asset['id']+'/complete',{},(202,))
        wait('/v1/assets/'+asset['id'],{'ready'})
        checks['real_upload_resume_idempotency_and_conflict']=True
        caps=call('GET','/v1/capabilities');profile=caps['profiles'][0]
        request=copy.deepcopy(state['job_request'])
        request.update(capabilities_revision=caps['capabilities_revision'],model_revision=profile['model_revision'],profile_id=profile['id'])
        active=call('POST','/v1/jobs',request,(202,))
        active_url='/v1/jobs/'+active['id']
        wait(active_url,{'running'})
        pending=call('POST','/v1/jobs',request,(202,))
        pending_url='/v1/jobs/'+pending['id']
        cancelled=call('POST',pending_url+'/cancel',{},(200,202))
        assert cancelled['state']=='cancelled'
        retry=call('POST',pending_url+'/retry',{},(202,))
        assert retry['retry_of']==pending['id'] and retry['id']!=pending['id']
        call('POST','/v1/jobs/'+retry['id']+'/cancel',{},(200,202))
        checks['queued_cancel_and_explicit_retry']=True
        cancel_start=time.monotonic()
        call('POST',active_url+'/cancel',{},(200,202))
        wait(active_url,{'cancelled'},timeout=60)
        checks['running_cancel_seconds']=time.monotonic()-cancel_start
        call('GET',active_url+'/result',expect=(409,))
        retry=call('POST',active_url+'/retry',{},(202,))
        retry_url='/v1/jobs/'+retry['id']
        wait(retry_url,{'succeeded'})
        result=call('GET',retry_url+'/result')
        assert all(expected[f'{f["output"]}_{f["source_frame"]:06d}.exr']==f['sha256'] for f in result['files'])
        assert len(result['files'])==144
        checks['post_cancel_model_reload_exact_exr_files']=144
        item=result['files'][0]
        other.json('file_isolation','GET',item['download_url'],expect=(404,))
        target=args.output/'resumed.exr'
        _,h,partial=c.request('partial_download','GET',item['download_url'],headers={'Range':'bytes=0-136'},expect=(206,))
        target.with_suffix('.part').write_bytes(partial)
        target.with_suffix('.part.etag').write_text(h['etag'])
        c.download(item,target)
        assert m.digest(target)==item['sha256']
        offline=m.Client('http://127.0.0.1:1','unused')
        assert offline.download(item,target)
        offline.close()
        checks['real_download_resume_and_offline_cache']=True
        preview=copy.deepcopy(request);preview.update(purpose='preview',output_range=dict(start=24,end=36))
        job=call('POST','/v1/jobs',preview,(202,));url='/v1/jobs/'+job['id']
        wait(url,{'succeeded'})
        manifest=call('GET',url+'/result')
        assert manifest['recipe']['processed_range']==dict(start=0,end=72)
        assert len(manifest['files'])==24
        cache=args.output/'preview';cache.mkdir(mode=0o700)
        for f in manifest['files']:
            assert expected[f'{f["output"]}_{f["source_frame"]:06d}.exr']==f['sha256']
            c.download(f,cache/f'{f["source_frame"]:06d}_{f["output"]}.exr')
        m.save(cache/'manifest.json',manifest)
        checks['preview_exact_full_context_files']=24
        deletion=call('DELETE',url,expect=(202,))
        wait(deletion['status_url'],{'deleted'})
        call('GET',manifest['files'][0]['download_url'],expect=(410,))
        deletion=call('DELETE','/v1/assets/'+asset['id'],expect=(202,))
        wait(deletion['status_url'],{'deleted'})
        checks['deletion_and_download_revocation']=True
        checks['passed']=True
    finally:
        m.save(args.output/'report.json',dict(checks=checks,requests=c.records,other_user_requests=other.records))
        c.close();other.close()
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
