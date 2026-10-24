"""Resolve owner-checked assets into immutable, content-addressed job recipes."""
import hashlib

from .common import canonical
from .media import frame_path
from .store import Problem

DEFAULT_CARRY=dict(enabled=True,confidence=.95,precedence='provided')


def dependencies(job):
    return job.get('_assets',[job['request']['asset_id']])


def resolve(request,source,profile,get_asset,asset_dir):
    source_spec=source['manifest'];ids=[source['id']]
    if source_spec['kind']!='image_sequence':raise Problem(422,'source_must_be_rgb')
    carry=dict(DEFAULT_CARRY,**request.get('carry',{}))
    guidance=dict(hint=None,carry=carry,masks=[])
    work=dict(carry=carry,hint=None,masks=[],linear=source_spec['colour']['transfer']=='linear')
    def aligned(id):
        asset=get_asset(id);spec=asset['manifest']
        if asset['state']!='ready':raise Problem(409,'guidance_not_ready')
        if spec['kind']!='matte_sequence':raise Problem(422,'guidance_must_be_coverage')
        for name in ('width','height','frame_count','geometry'):
            if spec[name]!=source_spec[name]:raise Problem(422,'guidance_alignment',name)
        for name in ('mode','frame_rate','time_base','start_pts','frame_duration_ticks'):
            if spec['timing'][name]!=source_spec['timing'][name]:raise Problem(422,'guidance_timing',name)
        ids.append(id)
        files={f['frame_index']:str(frame_path(asset_dir(id)/'frames',f)) for f in spec['files']}
        return asset,files
    if 'hint' in request:
        hint=request['hint'];asset,frames=aligned(hint['asset_id']);indices=sorted(frames)
        mode=hint['mode'];confidence=hint.get('confidence',.95)
        if mode=='first_frame' and indices!=[0]:raise Problem(422,'first_hint_frame')
        if mode=='sequence' and indices!=list(range(source_spec['frame_count'])):raise Problem(422,'hint_sequence_incomplete')
        quality={i:confidence for i in indices}
        overrides=hint.get('frame_confidence',[])
        if mode!='keyframes' and overrides:raise Problem(422,'per_frame_confidence_requires_keyframes')
        if len({f['frame_index'] for f in overrides})!=len(overrides):raise Problem(422,'duplicate_hint_confidence')
        for f in overrides:
            if f['frame_index'] not in frames:raise Problem(422,'confidence_without_hint')
            quality[f['frame_index']]=f['confidence']
        guidance['hint']=dict(content_sha256=asset['content_sha256'],mode=mode,
            confidence=float(confidence),frames=[dict(frame_index=i,confidence=float(quality[i])) for i in indices])
        work['hint']=dict(mode=mode,confidence=confidence,frames=frames,quality=quality)
    for mask in request.get('masks',[]):
        asset,frames=aligned(mask['asset_id'])
        expected=[0] if mask['mode']=='static' else list(range(source_spec['frame_count']))
        if sorted(frames)!=expected:raise Problem(422,'mask_frame_mapping')
        guidance['masks'].append(dict(content_sha256=asset['content_sha256'],role=mask['role'],mode=mask['mode']))
        work['masks'].append(dict(frames=frames,role=mask['role'],mode=mask['mode']))
    guidance['masks'].sort(key=lambda m:(m['role'],m['content_sha256'],m['mode']))
    recipe=dict(model_revision=profile['model_revision'],profile_id=profile['id'],execution_fingerprint=profile['execution_fingerprint'],
        asset_content_sha256=source['content_sha256'],processed_range=dict(start=0,end=source_spec['frame_count']),
        output_range=request['output_range'],output=request['output'],guidance=guidance,recipe_version='editor-v1')
    recipe['processing_fingerprint']=hashlib.sha256(canonical(recipe)).hexdigest()
    return recipe,work,list(dict.fromkeys(ids))
