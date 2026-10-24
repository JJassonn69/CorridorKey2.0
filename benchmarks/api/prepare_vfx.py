#!/usr/bin/env python3
"""Prepare private, reproducible VFX Assets fixtures; never modify the masters."""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import importlib.metadata
import json
from pathlib import Path
import struct
import subprocess

CONFIG = 'studio-config-v2.2.0_aces-v1.3_ocio-v2.4'
SOURCE_SPACE = 'S-Log3 S-Gamut3.Cine'
TARGET_SPACE = 'Linear Rec.709 (sRGB)'
CLIPS = (
    ('escape-s001', 'Woman in White Stands Holding Pistol', 48),
    ('saber-s009', 'Saber Lock Struggle Closeup', 4),
    ('ladder-s001', 'Man Climbs Ladder, Wide', 24),
)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def mov_timing(path):
    """Read the CFR video sample table without rounding a frame-rate display."""
    with open(path, 'rb') as f:
        def boxes(start, end):
            while start + 8 <= end:
                f.seek(start)
                size, kind = struct.unpack('>I4s', f.read(8))
                header = 8
                if size == 1:
                    size = struct.unpack('>Q', f.read(8))[0]
                    header = 16
                elif size == 0:
                    size = end - start
                if size < header or start + size > end:
                    raise ValueError('Invalid MOV atom bounds')
                yield kind, start + header, start + size
                start += size

        def child(parent, kind):
            return next((b[1:] for b in boxes(*parent) if b[0] == kind), None)

        moov = child((0, Path(path).stat().st_size), b'moov')
        if moov is None:
            raise ValueError('MOV has no metadata atom')
        for kind, start, end in boxes(*moov):
            if kind != b'trak':
                continue
            mdia = child((start, end), b'mdia')
            if mdia is None:
                continue
            hdlr = child(mdia, b'hdlr')
            f.seek(hdlr[0] + 8)
            if f.read(4) != b'vide':
                continue
            mdhd = child(mdia, b'mdhd')
            f.seek(mdhd[0])
            version = f.read(1)[0]
            f.seek(mdhd[0] + (20 if version == 1 else 12))
            timescale = struct.unpack('>I', f.read(4))[0]
            table = child(child(mdia, b'minf'), b'stbl')
            if child(table, b'ctts') is not None:
                raise ValueError('Reordered video needs an explicit PTS probe')
            stts = child(table, b'stts')
            f.seek(stts[0] + 4)
            entries = struct.unpack('>I', f.read(4))[0]
            if entries > (stts[1] - stts[0] - 8) // 8:
                raise ValueError('Invalid sample timing table')
            samples = [struct.unpack('>II', f.read(8)) for _ in range(entries)]
            durations = {duration for count, duration in samples if count}
            if len(durations) != 1 or not timescale:
                raise ValueError('Only verified CFR masters are supported')
            duration = durations.pop()
            rate = Fraction(timescale, duration)
            return dict(frame_count=sum(n for n, _ in samples),
                        time_base={'num': 1, 'den': timescale},
                        frame_rate={'num': rate.numerator, 'den': rate.denominator},
                        frame_duration_ticks=str(duration))
    raise ValueError('MOV has no video track')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--downloads', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    import imageio_ffmpeg
    import numpy as np
    from PIL import Image, PngImagePlugin
    import PyOpenColorIO as ocio

    args.output.mkdir(mode=0o700, parents=True, exist_ok=True)
    config = ocio.Config.CreateFromBuiltinConfig(CONFIG)
    processor = config.getProcessor(SOURCE_SPACE, TARGET_SPACE).getDefaultCPUProcessor()
    transform = dict(ocio_version=ocio.__version__, config=CONFIG,
                     config_sha256=hashlib.sha256(config.serialize().encode()).hexdigest(),
                     source=SOURCE_SPACE, destination=TARGET_SPACE,
                     decode='FFmpeg ProRes -> RGB48; BT.709 YUV matrix, limited-to-full range',
                     range_note='Range is an explicit fixture interpretation of the untagged ProRes range, not a colour-critical source certification.',
                     resize='linear RGB, Pillow float-channel Lanczos, 2048x1156 -> 1024x578',
                     encode='sRGB transfer; clip to [0,1]; nearest integer uint8',
                     view_transform='none')
    transform_id = 'vfx-fixture-' + hashlib.sha256(json.dumps(transform, sort_keys=True).encode()).hexdigest()
    versions = {p: importlib.metadata.version(p) for p in
                ('imageio-ffmpeg', 'opencolorio', 'numpy', 'pillow')}
    index = []
    for clip_id, title, first in CLIPS:
        source = args.downloads / (clip_id + '_prores2k.mov')
        timing = mov_timing(source)
        if first + 72 > timing['frame_count']:
            raise ValueError(f'{clip_id}: insufficient source frames')
        target = args.output / clip_id
        target.mkdir(exist_ok=False)
        frames = target / 'frames'
        frames.mkdir()
        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-nostdin', '-threads', '2',
               '-i', str(source), '-map', '0:v:0', '-an',
               '-vf', f'select=between(n\\,{first}\\,{first+71}),scale=in_color_matrix=bt709:in_range=tv:out_range=full,format=rgb48le',
               '-frames:v', '72', '-fps_mode', 'passthrough', '-f', 'rawvideo', 'pipe:1']
        clipped_low = clipped_high = total_samples = 0
        files = []
        with (target / 'decode.log').open('wb') as log:
            proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=log)
            try:
                size = 2048 * 1156 * 3 * 2
                for i in range(72):
                    chunks, remaining = [], size
                    while remaining:
                        chunk = proc.stdout.read(remaining)
                        if not chunk:
                            raise RuntimeError(f'{clip_id}: truncated frame {i}')
                        chunks.append(chunk)
                        remaining -= len(chunk)
                    rgb = np.frombuffer(b''.join(chunks), dtype='<u2').reshape(1156, 2048, 3).astype(np.float32) / 65535.0
                    processor.applyRGB(rgb)
                    linear = np.stack([np.asarray(Image.fromarray(rgb[:, :, c]).resize(
                        (1024, 578), Image.Resampling.LANCZOS)) for c in range(3)], axis=-1)
                    if not np.isfinite(linear).all():
                        raise ValueError('Non-finite colour conversion')
                    clipped_low += int((linear < 0).sum())
                    clipped_high += int((linear > 1).sum())
                    total_samples += linear.size
                    srgb = np.where(linear <= .0031308, linear * 12.92,
                                    1.055 * np.maximum(linear, 0) ** (1 / 2.4) - .055)
                    image = np.rint(np.clip(srgb, 0, 1) * 255).astype(np.uint8)
                    path = frames / f'{i:06d}.png'
                    info = PngImagePlugin.PngInfo()
                    info.add(b'sRGB', b'\x00')
                    Image.fromarray(image).save(path, pnginfo=info, compress_level=6)
                    files.append(dict(frame_index=i, bytes=path.stat().st_size,
                                      sha256=sha256(path), media_type='image/png'))
                if proc.stdout.read(1) or proc.wait() != 0:
                    raise RuntimeError(f'{clip_id}: decoder failed or produced excess data')
            finally:
                proc.stdout.close()
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
        api_timing = dict(mode='cfr', frame_rate=timing['frame_rate'],
                          time_base=timing['time_base'],
                          start_pts=str(first * int(timing['frame_duration_ticks'])),
                          frame_duration_ticks=timing['frame_duration_ticks'])
        asset = dict(kind='image_sequence', width=1024, height=578, frame_count=72,
                     timing=api_timing, colour=dict(primaries='bt709', white_point='D65',
                     transfer='srgb', matrix='rgb', range='full', pixel_type='uint8',
                     alpha='none', transform_id=transform_id),
                     geometry=dict(pixel_aspect={'num':1,'den':1}, scan='progressive', rotation_degrees=0), files=files)
        fixture = dict(id=clip_id, title=title, source_url='https://vfxassets.com/a/' + clip_id,
                       source_path=str(source.resolve()), source_bytes=source.stat().st_size,
                       source_sha256=sha256(source), source_timing=timing,
                       source_range={'start':first,'end':first+72}, context='This 72-frame extract only; not equivalent to the full master temporal context.',
                       usage='Private personal tests only; no redistribution, public media or model training.',
                       transform=transform, versions=versions,
                       clipping={'negative_samples':clipped_low,'above_one_samples':clipped_high,'total_samples':total_samples},
                       frame_dir=str(frames.resolve()), asset_request=asset)
        (target / 'fixture.json').write_text(json.dumps(fixture, indent=2) + '\n')
        index.append(dict(id=clip_id, fixture=str((target/'fixture.json').resolve()),
                          source_sha256=fixture['source_sha256'], upload_bytes=sum(f['bytes'] for f in files)))
        print(json.dumps(index[-1]), flush=True)
    (args.output / 'index.json').write_text(json.dumps(index, indent=2) + '\n')


if __name__ == '__main__':
    main()
