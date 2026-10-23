"""Parallel EXR export must preserve data and terminate cleanly on I/O failures."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest import mock

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
import cv2
import numpy as np
from corridorkey.runtime import _release_output_writers
from corridorkey.video_io import ExrSequenceWriter as Writer


class ExrWriterTests(unittest.TestCase):
    def test_parallel_matches_serial_bytes_for_hdr_and_half(self):
        rng = np.random.default_rng(4)
        with tempfile.TemporaryDirectory() as tmp:
            for channels in (1, 3, 4):
                for dtype in ('half', 'float'):
                    frames = (rng.normal(size=(7, 19, 23, channels)) * 80000).astype(np.float32)
                    outputs = []
                    for workers in (1, 4, 6):
                        path = Path(tmp) / f'{channels}-{dtype}-{workers}'
                        writer = Writer(str(path), 24, (23, 19), channels=channels,
                                        dtype=dtype, workers=workers)
                        for start, stop in ((0, 2), (2, 3), (3, 6), (6, 7)):
                            writer.write_many(frames[start:stop])
                        writer.release()
                        writer.release()
                        files = sorted(path.glob('*.exr'))
                        self.assertEqual([p.name for p in files], [f'frame_{i:06d}.exr' for i in range(7)])
                        for file in files:
                            self.assertTrue(np.isfinite(cv2.imread(str(file), cv2.IMREAD_UNCHANGED)).all())
                        outputs.append([p.read_bytes() for p in files])
                        with self.assertRaisesRegex(RuntimeError, 'closed'):
                            writer.write_many(frames[:1])
                    self.assertEqual(outputs[0], outputs[1])
                    self.assertEqual(outputs[0], outputs[2])

    def test_failed_writes_raise_without_deadlock(self):
        frame = np.zeros((1, 4, 4, 3), np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            for failure in (False, OSError('disk failure')):
                writer = Writer(tmp, 24, (4, 4), dtype='float', workers=2, maxsize=1)
                kw = {'side_effect': failure} if isinstance(failure, Exception) else {'return_value': False}
                with mock.patch.object(cv2, 'imwrite', **kw):
                    for _ in range(3):
                        writer.write_many(frame)
                    with self.assertRaises((RuntimeError, OSError)):
                        writer.write_many(frame)
                    with self.assertRaises((RuntimeError, OSError)):
                        writer.release()
                self.assertTrue(all(not t.is_alive() for t in writer._pool._threads))

    def test_all_outputs_release_after_one_writer_fails(self):
        first = OSError('first disk failure')
        outputs = {i: mock.Mock() for i in range(3)}
        outputs[0].release.side_effect = first
        outputs[2].release.side_effect = OSError('second disk failure')
        self.assertIs(_release_output_writers(outputs), first)
        for output in outputs.values():
            output.release.assert_called_once_with()

    def test_backpressure_bounds_pending_work(self):
        frame = np.zeros((1, 4, 4, 3), np.float32)
        gate = threading.Event()
        entered = threading.Event()
        def write(*args):
            entered.set()
            if not gate.wait(5):
                raise RuntimeError('test writer timed out')
            return True
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(cv2, 'imwrite', side_effect=write):
            writer = Writer(tmp, 24, (4, 4), workers=2, maxsize=1)
            for _ in range(3):
                writer.write_many(frame)
            self.assertTrue(entered.wait(2))
            with ThreadPoolExecutor(max_workers=1) as caller:
                blocked = caller.submit(writer.write_many, frame)
                try:
                    time.sleep(.05)
                    self.assertFalse(blocked.done())
                    self.assertEqual(len(writer._pending), 2)  # oldest is being awaited
                finally:
                    gate.set()
                blocked.result(timeout=3)
            writer.release()
            self.assertEqual(writer.frame_idx, 4)


if __name__ == '__main__':
    unittest.main()
