"""The native 8-bit fast path must preserve every BF16 input bit."""
import unittest
from unittest import mock
import numpy as np
import torch
from corridorkey import video_io as candidate
from corridorkey.video_io import (
    cv2, cu, resize_hwc_lanczos, center_pad_to_shape,
    has_linear_adjust, apply_linear_adjust,
)

# Original conversion from upstream 485f5e9ab759ead6138cc990776184b59d6ea971.
def reference_preprocess(frame_bgr, meta: dict, *, need_model: bool):
    h_out, w_out = meta["shape_hw"]
    h_model, w_model = meta["model_native_hw"]
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    frame = resize_hwc_lanczos(frame, (h_model, w_model))
    frame, _ = center_pad_to_shape(frame, h_out, w_out)
    review = torch.from_numpy(frame.transpose(2, 0, 1)).to(dtype=torch.bfloat16)
    if not need_model and not has_linear_adjust(meta):
        return None, review, review
    linear = cu.srgb_to_linear(frame)
    linear_model = apply_linear_adjust(linear, meta)
    if has_linear_adjust(meta):
        foundation_np = cu.linear_to_vfm_sdr(linear_model)
        foundation = torch.from_numpy(
            foundation_np.transpose(2, 0, 1)).to(dtype=torch.bfloat16)
    else:
        foundation = review
    if not need_model:
        return None, foundation, review
    model = cu.linear_to_asinh(linear_model)
    model_t = torch.from_numpy(model.transpose(2, 0, 1)).to(dtype=torch.bfloat16)
    return model_t, foundation, review

class VideoLookupTests(unittest.TestCase):
    def compare(self, frame, meta):
        for need_model in (False, True):
            expected = reference_preprocess(frame, meta, need_model=need_model)
            actual = candidate.preprocess_video_frame_views(frame, meta, need_model=need_model)
            for a, b in zip(expected, actual):
                if a is None:
                    self.assertIsNone(b)
                else:
                    self.assertEqual(a.dtype, b.dtype)
                    self.assertEqual(a.shape, b.shape)
                    self.assertEqual(a.stride(), b.stride())
                    self.assertTrue(torch.equal(a.view(torch.int16), b.view(torch.int16)))

    def test_all_codes_layouts_sizes_and_reflection_padding(self):
        rng = np.random.default_rng(39)
        for height, width in ((16,16),(19,259),(720,1280)):
            frame = rng.integers(0,256,size=(height,width,3),dtype=np.uint8)
            frame.reshape(-1)[:256] = np.arange(256, dtype=np.uint8)
            for image in (frame,frame[:,::-1]):
                meta = dict(model_native_hw=(height,width), shape_hw=(height+5,width+7))
                self.compare(image,meta)

    def test_scaled_adjusted_and_non_u8_inputs_use_original_path(self):
        frame = np.random.default_rng(4).integers(0,256,size=(31,37,3),dtype=np.uint8)
        cases = [
            (frame,dict(model_native_hw=(27,23),shape_hw=(32,32))),
            (frame,dict(model_native_hw=(31,37),shape_hw=(32,40),linear_brightness=1.3,linear_contrast=.8)),
            (frame.astype(np.float32),dict(model_native_hw=(31,37),shape_hw=(32,40)))
        ]
        with mock.patch.object(candidate,'_u8_video_tables',side_effect=AssertionError('fast path used')):
            for image,meta in cases:
                self.compare(image,meta)

    def test_cache_is_bounded_readonly_and_stores_bf16_bits(self):
        tables = candidate._u8_video_tables()
        self.assertIs(tables,candidate._u8_video_tables())
        self.assertEqual(candidate._u8_video_tables.cache_info().maxsize,1)
        for table in tables:
            self.assertEqual(table.dtype,np.uint16)
            self.assertEqual(table.shape,(256,))
            self.assertFalse(table.flags.writeable)

if __name__ == '__main__':
    torch.set_num_threads(6)
    unittest.main()
