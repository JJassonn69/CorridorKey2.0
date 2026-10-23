"""Foundation tails preserve the original batch shape and frame coverage."""
import unittest
import torch
from corridorkey.runtime import _full_foundation_batches


class FoundationBatchTests(unittest.TestCase):
    def test_short_full_and_partial_batches_preserve_frames_and_shape(self):
        for count, chunk in [(0,24), (1,24), (8,24), (24,24), (25,24),
                             (47,24), (48,24), (49,24), (160,80), (161,80)]:
            with self.subTest(count=count, chunk=chunk):
                inputs = ((t, None, torch.arange(t, min(count,t+chunk)))
                          for t in range(0,count,chunk))
                covered = set()
                for start, _, batch in _full_foundation_batches(inputs, chunk):
                    self.assertEqual(len(batch), min(count,chunk))
                    self.assertTrue(torch.equal(batch, torch.arange(start,start+len(batch))))
                    covered.update(batch.tolist())
                self.assertEqual(covered, set(range(count)))


if __name__ == '__main__':
    unittest.main()
