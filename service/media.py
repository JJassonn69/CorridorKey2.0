"""Bounded canonical PNG/EXR inputs and explicit coverage constraints. No CUDA."""
import os
from pathlib import Path
import struct

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')


def extension(file):
    return {'image/png':'.png','image/x-exr':'.exr'}[file['media_type']]


def frame_path(directory,file):
    return Path(directory)/f'{file["frame_index"]:06d}{extension(file)}'


def exr_header(path):
    """Reject oversized/multipart/tiled/subsampled inputs before native decoding."""
    with Path(path).open('rb') as stream:
        if stream.read(4)!=b'v/1\x01' or stream.read(4)!=struct.pack('<I',2):
            raise ValueError('Only single-part scanline EXR v2 is supported')
        def string():
            data=bytearray()
            while len(data)<256:
                b=stream.read(1)
                if not b:raise ValueError('Truncated EXR header')
                if b==b'\0':return bytes(data)
                data.extend(b)
            raise ValueError('Oversized EXR attribute name')
        attributes={}
        while stream.tell()<65536:
            name=string()
            if not name:break
            kind=string();length=struct.unpack('<I',stream.read(4))[0]
            if length>32768 or name in attributes:raise ValueError('Invalid EXR attribute')
            data=stream.read(length)
            if len(data)!=length:raise ValueError('Truncated EXR attribute')
            attributes[name]=(kind,data)
        else:raise ValueError('Oversized EXR header')
    def attribute(name,kind):
        k,v=attributes[name]
        if k!=kind:raise ValueError('Invalid EXR attribute type')
        return v
    window=struct.unpack('<4i',attribute(b'dataWindow',b'box2i'))
    if window[:2]!=(0,0) or attribute(b'displayWindow',b'box2i')!=attribute(b'dataWindow',b'box2i'):
        raise ValueError('EXR windows must match and start at zero')
    if attribute(b'compression',b'compression') not in (b'\x00',b'\x01',b'\x02',b'\x03'):
        raise ValueError('Only lossless NONE/RLE/ZIPS/ZIP EXR compression is accepted')
    data=attribute(b'channels',b'chlist');i=0;channels={}
    while i<len(data) and data[i]:
        end=data.index(0,i);name=data[i:end].decode('ascii');i=end+1
        pixel,linear,x,y=struct.unpack_from('<iB3xii',data,i);i+=16
        if name in channels or pixel not in (1,2) or (x,y)!=(1,1):
            raise ValueError('Unsupported EXR channel type or sampling')
        channels[name]='float16' if pixel==1 else 'float32'
    if i!=len(data)-1 or not channels:raise ValueError('Invalid EXR channel list')
    return dict(width=window[2]+1,height=window[3]+1,channels=channels,attributes=attributes)


def read_coverage(path):
    import cv2
    import numpy as np
    image=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim!=2:raise ValueError('Coverage must be one channel')
    if image.dtype==np.uint8:image=image.astype(np.float32)/np.float32(255)
    elif image.dtype==np.uint16:image=image.astype(np.float32)/np.float32(65535)
    else:image=image.astype(np.float32,copy=False)
    if not np.isfinite(image).all() or np.any(image<0) or np.any(image>1):
        raise ValueError('Coverage must be finite and in [0,1]')
    return image


def validate_file(path,manifest,file):
    import numpy as np
    matte=manifest['kind']=='matte_sequence'
    pixel=manifest['colour']['pixel_type']
    expected=(manifest['width'],manifest['height'])
    if file['media_type']=='image/png':
        from PIL import Image
        with Path(path).open('rb') as stream:header=stream.read(33)
        w,h,depth,colour,compression,filtering,interlace=struct.unpack('>IIBBBBB',header[16:29])
        wanted=(8 if pixel=='uint8' else 16,0 if matte else 2,0,0,0)
        if header[:8]!=b'\x89PNG\r\n\x1a\n' or header[12:16]!=b'IHDR' or (w,h)!=expected or (depth,colour,compression,filtering,interlace)!=wanted:
            raise ValueError('PNG type or dimensions differ from declared media')
        with Image.open(path) as image:
            if image.format!='PNG' or image.size!=expected:raise ValueError('Invalid PNG')
            image.verify()
        with Image.open(path) as image:image.load()
    else:
        h=exr_header(path)
        names={'Y'} if matte else {'R','G','B'}
        if (h['width'],h['height'])!=expected or set(h['channels'])!=names or set(h['channels'].values())!={pixel}:
            raise ValueError('EXR type, channels or dimensions differ from declared media')
        if not matte and b'chromaticities' in h['attributes']:
            kind,data=h['attributes'][b'chromaticities']
            if kind!=b'chromaticities' or not np.allclose(struct.unpack('<8f',data),
                [.64,.33,.30,.60,.15,.06,.3127,.3290],rtol=0,atol=1e-6):
                raise ValueError('Convert EXR primaries to linear sRGB before upload')
        import cv2
        image=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
        if image is None or not np.isfinite(image).all():raise ValueError('Invalid EXR pixels')
        if image.shape!=(expected[1],expected[0])+(tuple() if matte else (3,)):
            raise ValueError('Unexpected decoded EXR shape')
    if matte:read_coverage(path)


def apply_constraints(output,masks):
    """Keep raises coverage; exclusion lowers it last. Straight foreground is unchanged."""
    if not masks:return
    import cv2
    import numpy as np
    static={}
    for path in sorted(Path(output).glob('*_alpha_exr/alpha_*.exr')):
        index=int(path.stem.rsplit('_',1)[1]);alpha=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
        keep=np.zeros_like(alpha);exclude=np.zeros_like(alpha)
        for mask in masks:
            source=mask['frames'][0 if mask['mode']=='static' else index]
            if mask['mode']=='static':
                if source not in static:static[source]=read_coverage(source)
                image=static[source]
            else:image=read_coverage(source)
            if image.shape!=alpha.shape:raise ValueError('Constraint alignment changed')
            if mask['role']=='force_foreground':np.maximum(keep,image,out=keep)
            else:np.maximum(exclude,image,out=exclude)
        corrected=np.minimum(np.maximum(alpha,keep),np.float32(1)-exclude)
        if not np.array_equal(corrected,alpha):
            if not cv2.imwrite(str(path),corrected,[cv2.IMWRITE_EXR_TYPE,cv2.IMWRITE_EXR_TYPE_FLOAT,
                cv2.IMWRITE_EXR_COMPRESSION,cv2.IMWRITE_EXR_COMPRESSION_ZIP]):
                raise RuntimeError('Constraint export failed')
