"""Persistent serial worker. Only this module imports the inference runtime."""
import contextlib
import dataclasses
import hashlib
import io
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback

from .common import digest, save


def main(package, inbox, events, log, capture_raw=False):
    os.setsid()
    os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    os.environ.setdefault('MKL_NUM_THREADS', '2')
    with open(log, 'a', buffering=1) as stream, contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        try:
            sys.path.insert(0, str(package))
            import numpy as np
            import torch
            import cv2
            from corridorkey.runtime import InferenceConfig, RuntimePaths, WriterConfig, load_model, run_inference
            torch.set_num_threads(2)
            cv2.setNumThreads(1)
            torch.set_float32_matmul_precision('highest')
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = False
            paths = RuntimePaths.from_package(package)
            before = time.monotonic()
            model = load_model(paths, device='cuda')
            torch.cuda.synchronize()
            events.put(('ready', {'model_load_seconds': time.monotonic() - before, 'pid': os.getpid()}))
            while True:
                work = inbox.get()
                if work is None:
                    return
                job_id, source, attempt, count = work
                attempt = Path(attempt)
                try:
                    random.seed(0)
                    np.random.seed(0)
                    torch.manual_seed(0)
                    torch.cuda.manual_seed_all(0)
                    cfg = InferenceConfig(hann_chunk=24, hann_stride=12, num_frames=count,
                                          stage2_batch=1, auto_vram=False, model_scale=1,
                                          low_vram=False, output_format='exr', exr_dtype='float',
                                          outputs={'alpha', 'fg'}, temp_dir=attempt / 'scratch')
                    # The same eight-frame warmup as the direct benchmark. Its cost is
                    # included in service latency. No numerical runtime code is changed.
                    events.put(('progress', {'id': job_id, 'phase': 'prepass', 'completed': 0}))
                    warm_start = time.monotonic()
                    run_inference(model, source, str(attempt / 'warmup'),
                                  dataclasses.replace(cfg, num_frames=min(8, count)), WriterConfig(),
                                  initial_hint_path=None, hint_video_path=None)
                    torch.cuda.synchronize()
                    shutil.rmtree(attempt / 'warmup')
                    warm_seconds = time.monotonic() - warm_start
                    raw = []

                    def capture(_module, _inputs, result):
                        logits = result[0] if isinstance(result, tuple) else result
                        for frame in logits[0]:
                            item = {'index': len(raw), 'shape': list(frame.shape), 'model_dtype': str(frame.dtype)}
                            if capture_raw:
                                array = frame.detach().float().cpu().numpy()
                                if not np.isfinite(array).all():
                                    raise ValueError('Nonfinite raw prediction')
                                buffer = io.BytesIO()
                                np.save(buffer, array, allow_pickle=False)
                                item['sha256'] = hashlib.sha256(buffer.getbuffer()).hexdigest()
                            raw.append(item)
                        events.put(('progress', {'id': job_id, 'phase': 'stage2', 'completed': len(raw)}))

                    hook = model.stage2.register_forward_hook(capture)
                    torch.cuda.reset_peak_memory_stats()
                    start = time.monotonic()
                    try:
                        run_inference(model, source, str(attempt / 'exr'), cfg, WriterConfig(),
                                      initial_hint_path=None, hint_video_path=None)
                        torch.cuda.synchronize()
                    finally:
                        hook.remove()
                    seconds = time.monotonic() - start
                    if len(raw) != count:
                        raise ValueError('Incomplete raw prediction count')
                    files = []
                    for output in ('alpha', 'fg'):
                        matches = sorted((attempt / 'exr').glob(f'*_{output}_exr/{output}_*.exr'))
                        if len(matches) != count:
                            raise ValueError('Incomplete EXR count')
                        for i, path in enumerate(matches):
                            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                            if image is None or image.dtype != np.float32 or not np.isfinite(image).all():
                                raise ValueError('Invalid float EXR')
                            with path.open('rb') as f:
                                os.fsync(f.fileno())
                            files.append(dict(source_frame=i, output=output,
                                              path=str(path.relative_to(attempt)), bytes=path.stat().st_size,
                                              sha256=digest(path)))
                    stats = dict(raw=raw, files=files, inference_seconds=seconds, warmup_seconds=warm_seconds,
                                 inference_start_monotonic=start, inference_end_monotonic=start+seconds,
                                 peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                 peak_reserved_bytes=torch.cuda.max_memory_reserved(), capture_raw=capture_raw,
                                 precision=dict(matmul_precision=torch.get_float32_matmul_precision(),
                                                tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                                                tf32_cudnn=torch.backends.cudnn.allow_tf32), pid=os.getpid())
                    save(attempt / 'worker.json', stats)
                    shutil.rmtree(attempt / 'scratch', ignore_errors=True)
                    events.put(('done', {'id': job_id}))
                except BaseException:
                    traceback.print_exc()
                    events.put(('failed', {'id': job_id}))
                    # Unknown model/CUDA state must never be reused.
                    return
        except BaseException:
            traceback.print_exc()
            events.put(('boot_failed', {}))
