"""Prefetch must preserve window order, bound lookahead, and close after failures."""
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import unittest

import torch
from corridorkey.video_io import (
    PrefetchedModelWindowReader as Prefetch,
    SequentialVideoWindowReader,
    SequentialFrameDirWindowReader,
)

class Reader:
    def __init__(self):
        self.calls = []
        self.closed = False
        self.finished = threading.Event()
    def window(self, t0, t1, *, model_only=False):
        assert model_only and not self.closed
        self.calls.append((t0,t1))
        self.finished.set()
        return torch.arange(t0,t1), None, None
    def close(self):
        self.closed = True

class InputPrefetchTests(unittest.TestCase):
    def test_order_overlap_tail_and_bounded_lookahead(self):
        reader = Reader()
        windows = [(0,4),(2,6),(3,7)]
        prefetch = Prefetch(reader, windows)
        try:
            self.assertTrue(reader.finished.wait(2))
            self.assertEqual(reader.calls, windows[:1])
            for i, window in enumerate(windows):
                result = prefetch.window(*window)
                self.assertEqual(result[0].tolist(), list(range(*window)))
                self.assertEqual(result[1:], (None,None))
                time.sleep(.02)
                self.assertEqual(reader.calls, windows[:min(i+2,len(windows))])
        finally:
            prefetch.close()
        prefetch.close()
        self.assertTrue(reader.closed)
        self.assertTrue(all(not t.is_alive() for t in prefetch._pool._threads))

    def test_decoder_failure_and_early_close(self):
        class Broken(Reader):
            def window(self, *args, **kwargs):
                raise OSError('decode failed')
        broken = Broken()
        prefetch = Prefetch(broken, [(0,2),(1,3)])
        with self.assertRaisesRegex(OSError,'decode failed'):
            prefetch.window(0,2)
        prefetch.close()
        self.assertTrue(broken.closed)
        entered, finish = threading.Event(), threading.Event()
        class Slow(Reader):
            def window(self, *args, **kwargs):
                entered.set()
                if not finish.wait(3):
                    raise RuntimeError('test timed out')
                self.assert_open = not self.closed
                return super().window(*args, **kwargs)
        slow = Slow()
        prefetch = Prefetch(slow, [(0,2),(1,3)])
        self.assertTrue(entered.wait(2))
        with ThreadPoolExecutor(max_workers=1) as caller:
            closing = caller.submit(prefetch.close)
            try:
                time.sleep(.03)
                self.assertFalse(closing.done())
                self.assertFalse(slow.closed)
            finally:
                finish.set()
            closing.result(timeout=2)
        self.assertTrue(slow.assert_open and slow.closed)
        self.assertEqual(slow.calls, [(0,2)])

    def test_model_only_stack_matches_original_reader(self):
        for cls in (SequentialVideoWindowReader, SequentialFrameDirWindowReader):
            reader = cls.__new__(cls)
            def populate(end):
                for i in range(end):
                    reader.cache.setdefault(i, (torch.full((3,5,7),i,dtype=torch.bfloat16),
                                                torch.ones(3,5,7),torch.zeros(3,5,7)))
            reader._read_until = populate
            reader.cache = {}
            original = reader.window(1,4)[0]
            model, foundation, review = reader.window(1,4,model_only=True)
            self.assertTrue(torch.equal(original,model))
            self.assertIsNone(foundation)
            self.assertIsNone(review)
            self.assertNotIn(0,reader.cache)

if __name__ == '__main__':
    unittest.main()
