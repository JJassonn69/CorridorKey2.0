"""Prepare private editor-input fixtures from the user's existing footage/caches."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import cv2
import numpy as np
from PIL import Image

spec=importlib.util.spec_from_file_location('prepare_vfx',Path(__file__).with_name('prepare_vfx.py'))
vfx=importlib.util.module_from_spec(spec);spec.loader.exec_module(vfx)


def save(path,value):
    path.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base-fixture',type=Path,required=True)
    ap.add_argument('--baseline-cache',type=Path,required=True);ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();os.umask(0o077)
    args.output.mkdir(mode=0o700,parents=True,exist_ok=False)
    base=json.loads(args.base_fixture.read_text());geometry=base['asset_request']['geometry'];timing=base['asset_request']['timing']
    fixtures={}
    def fixture(name,count,files,colour,width=1024,height=578,kind='image_sequence',directory=None,**metadata):
        directory=directory or args.output/name/'frames'
        result=dict(id=name,frame_dir=str(directory.resolve()),asset_request=dict(kind=kind,width=width,height=height,
            frame_count=count,timing=timing,geometry=geometry,colour=colour,files=files),
            usage='Private personal tests only; no redistribution or training.',**metadata)
        path=args.output/name/'fixture.json';save(path,result);fixtures[name]=path
        return result
    def file(path,index,media='image/png'):
        return dict(frame_index=index,bytes=path.stat().st_size,sha256=vfx.sha256(path),media_type=media)
    rgb=fixture('rgb36',36,base['asset_request']['files'][:36],base['asset_request']['colour'],directory=Path(base['frame_dir']))
    matte_dir=args.output/'matte36/frames';matte_dir.mkdir(parents=True)
    matte_files=[]
    for i in range(36):
        alpha=cv2.imread(str(args.baseline_cache/f'{i:06d}_alpha.exr'),cv2.IMREAD_UNCHANGED)
        if alpha is None:raise ValueError('Missing verified baseline alpha')
        path=matte_dir/f'{i:06d}.png';cv2.imwrite(str(path),np.rint(np.clip(alpha,0,1)*65535).astype(np.uint16))
        matte_files.append(file(path,i))
    coverage=dict(interpretation='coverage',pixel_type='uint16')
    fixture('matte36',36,matte_files,coverage,kind='matte_sequence',directory=matte_dir,
            note='Prior model predictions are guidance test inputs, not ground-truth mattes.')
    for name,count,indices in [('first36',36,[0]),('keys36',36,[0,12,35]),('first8',8,[0])]:
        fixture(name,count,[matte_files[i] for i in indices],coverage,kind='matte_sequence',directory=matte_dir)
    for name,role in [('garbage','force_background'),('holdout','force_background'),('inside','force_foreground')]:
        directory=args.output/name/'frames';directory.mkdir(parents=True)
        frames=[]
        for i in range(36 if name=='holdout' else 1):
            mask=np.zeros((578,1024),np.uint8)
            if name=='inside':mask[180:420,350:700]=255
            elif name=='garbage':mask[250:500,500:820]=255
            else:mask[100:450,320+i*4:390+i*4]=128
            p=directory/f'{i:06d}.png';cv2.imwrite(str(p),mask);frames.append(file(p,i))
        fixture(name,36,frames,dict(interpretation='coverage',pixel_type='uint8'),kind='matte_sequence',directory=directory)
    import imageio_ffmpeg
    import PyOpenColorIO as ocio
    config=ocio.Config.CreateFromBuiltinConfig(vfx.CONFIG)
    processor=config.getProcessor(vfx.SOURCE_SPACE,vfx.TARGET_SPACE).getDefaultCPUProcessor()
    directories={name:args.output/name/'frames' for name in ['linear36','half8','large144']}
    for directory in directories.values():directory.mkdir(parents=True)
    lists={name:[] for name in directories};minimum=float('inf');maximum=-float('inf')
    # Source frames 48:192 are available in the 204-frame Escape master.
    command=[imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-nostdin','-threads','2','-i',str(args.source),'-map','0:v:0','-an',
        '-vf',r'select=between(n\,48\,191),scale=in_color_matrix=bt709:in_range=tv:out_range=full,format=rgb48le',
        '-frames:v','144','-fps_mode','passthrough','-f','rawvideo','pipe:1']
    with (args.output/'decode.log').open('wb') as log:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=log,stdin=subprocess.DEVNULL)
        try:
            for i in range(144):
                remaining=2048*1156*3*2;blocks=[]
                while remaining:
                    block=process.stdout.read(remaining)
                    if not block:raise RuntimeError('Truncated source frame')
                    blocks.append(block);remaining-=len(block)
                image=np.frombuffer(b''.join(blocks),'<u2').reshape(1156,2048,3).astype(np.float32)/np.float32(65535)
                processor.applyRGB(image)
                if not np.isfinite(image).all():raise ValueError('Nonfinite OCIO output')
                if i<36:
                    linear=np.stack([np.asarray(Image.fromarray(image[:,:,c]).resize((1024,578),Image.Resampling.LANCZOS)) for c in range(3)],axis=-1)
                    minimum=min(minimum,float(linear.min()));maximum=max(maximum,float(linear.max()))
                    p=directories['linear36']/f'{i:06d}.exr'
                    cv2.imwrite(str(p),linear[:,:,::-1],[cv2.IMWRITE_EXR_TYPE,cv2.IMWRITE_EXR_TYPE_FLOAT,cv2.IMWRITE_EXR_COMPRESSION,cv2.IMWRITE_EXR_COMPRESSION_ZIP])
                    lists['linear36'].append(file(p,i,'image/x-exr'))
                    if i<8:
                        p=directories['half8']/f'{i:06d}.exr'
                        cv2.imwrite(str(p),linear[:,:,::-1],[cv2.IMWRITE_EXR_TYPE,cv2.IMWRITE_EXR_TYPE_HALF,cv2.IMWRITE_EXR_COMPRESSION,cv2.IMWRITE_EXR_COMPRESSION_ZIP])
                        lists['half8'].append(file(p,i,'image/x-exr'))
                large=np.stack([np.asarray(Image.fromarray(image[:,:,c]).resize((1913,1080),Image.Resampling.LANCZOS)) for c in range(3)],axis=-1)
                large=np.pad(large,((0,0),(3,4),(0,0)))
                srgb=np.where(large<=.0031308,large*12.92,1.055*np.maximum(large,0)**(1/2.4)-.055)
                p=directories['large144']/f'{i:06d}.png';cv2.imwrite(str(p),np.rint(np.clip(srgb,0,1)*255).astype(np.uint8)[:,:,::-1])
                lists['large144'].append(file(p,i))
                if i%36==0:print(json.dumps(dict(prepared_source_frames=i+1)),flush=True)
            if process.stdout.read(1) or process.wait()!=0:raise RuntimeError('Source decoder failed')
        finally:
            process.stdout.close()
            if process.poll() is None:process.kill();process.wait()
    transform='editor-linear-'+hashlib.sha256(config.serialize().encode()).hexdigest()
    for name,count,pixel in [('linear36',36,'float32'),('half8',8,'float16')]:
        colour=copy.deepcopy(base['asset_request']['colour']);colour.update(transfer='linear',pixel_type=pixel,transform_id=transform)
        fixture(name,count,lists[name],colour,hdr_range=[minimum,maximum],transform='Same recorded SLog3/SGamut3.Cine interpretation as the prepared SDR fixtures, without transfer encoding or clipping.')
    large_transform='Linear Lanczos to 1913x1080, black pad left3/right4, then sRGB encoding and RGB8 clipping.'
    large_colour=copy.deepcopy(base['asset_request']['colour'])
    large_colour['transform_id']='editor-large-'+hashlib.sha256(large_transform.encode()).hexdigest()
    fixture('large144',144,lists['large144'],large_colour,width=1920,height=1080,transform=large_transform)
    directory=args.output/'first144/frames';directory.mkdir(parents=True)
    alpha=cv2.imread(str(args.baseline_cache/'000000_alpha.exr'),cv2.IMREAD_UNCHANGED)
    alpha=np.pad(np.asarray(Image.fromarray(alpha).resize((1913,1080),Image.Resampling.LANCZOS)),((0,0),(3,4)))
    p=directory/'000000.png';cv2.imwrite(str(p),np.rint(np.clip(alpha,0,1)*65535).astype(np.uint16))
    fixture('first144',144,[file(p,0)],coverage,width=1920,height=1080,kind='matte_sequence',directory=directory)
    def hint(name,mode,**kwargs):return dict(fixture=str(fixtures[name]),mode=mode,**kwargs)
    masks=[dict(fixture=str(fixtures[n]),mode='sequence' if n=='holdout' else 'static',role=r,label=n)
           for n,r in [('garbage','force_background'),('holdout','force_background'),('inside','force_foreground')]]
    sparse=dict(hint=hint('keys36','keyframes',confidence=.85,frame_confidence=[dict(frame_index=12,confidence=.35),dict(frame_index=35,confidence=.95)]),carry=dict(enabled=True,confidence=.6,precedence='provided'))
    cases=[('first','rgb36',dict(hint=hint('first36','first_frame',confidence=.9),carry=dict(enabled=True,confidence=.85,precedence='carried'))),
           ('dense-carried','rgb36',dict(hint=hint('matte36','sequence',confidence=.7),carry=dict(enabled=True,confidence=.55,precedence='carried'))),
           ('dense-provided','rgb36',dict(hint=hint('matte36','sequence',confidence=.7),carry=dict(enabled=True,confidence=.55,precedence='provided'))),
           ('sparse','rgb36',sparse),('sparse-no-carry','rgb36',dict(sparse,carry=dict(enabled=False,confidence=.6,precedence='provided'))),
           ('constraints','rgb36',dict(masks=masks)),('linear','linear36',dict(sparse,masks=masks)),
           ('half','half8',dict(hint=hint('first8','first_frame',confidence=.75))),
           ('large','large144',dict(hint=hint('first144','first_frame',confidence=.85)))]
    matrix=[]
    for name,source,controls in cases:
        p=args.output/'controls'/(name+'.json');save(p,controls)
        matrix.append(dict(name=name,fixture=str(fixtures[source]),guidance=str(p),
            profile='native-1080p-24x12' if name=='large' else 'native-720p-24x12',output_range=dict(start=24,end=26) if name=='large' else None))
    save(args.output/'matrix.json',matrix)
    print(json.dumps(dict(cases=len(matrix),linear_range=[minimum,maximum],upload_bytes={k:sum(f['bytes'] for f in v) for k,v in lists.items()})),flush=True)


if __name__=='__main__':main()
