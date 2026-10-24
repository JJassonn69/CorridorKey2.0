"""Inspect actual Stage 1 hint tensors across overlapping windows; no neural model."""
import contextlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'Uses the existing inference environment; no Mac torch installation needed')
class HintMappingTests(unittest.TestCase):
    def exercise(self,keyframes=None,precedence='provided',dense=False):
        import numpy as np
        from corridorkey import runtime
        calls=[]
        class Stage:
            d_16=1;cradio_dim=1
            def parameters(self):yield torch.zeros(1)
            def __call__(self,rgb,hint,moge,**kwargs):
                calls.append((hint.clone(),kwargs['hint_quality'].clone()))
                t=rgb.shape[1]
                return dict(baton=torch.zeros(1,t,1,4,4),aux=torch.full((1,t,1,64,64),.25))
        class Reader:
            def __init__(self,*args):pass
            def window(self,a,b,**kwargs):return torch.zeros(b-a,3,64,64),None,None
            def close(self):pass
        class Hints:
            def window(self,a,b):return torch.full((b-a,1,64,64),.75,dtype=torch.bfloat16)
            def close(self):pass
        meta=dict(num_frames=6,shape_hw=(64,64),native_hw=(64,64),model_native_hw=(64,64),pad=(0,0,0,0))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);paths={}
            for name in ('moge','cradio'):
                p=root/(name+'.f32');np.zeros((6,1,1,1),np.float32).tofile(p)
                paths[name]=str(p);paths[name+'_shape']=[6,1,1,1]
            with patch.object(runtime,'make_hint_reader',return_value=Hints() if dense else None), \
                 patch.object(runtime,'load_initial_hint_frame',side_effect=lambda p,m:torch.full((1,64,64),.75 if p=='two' else .5,dtype=torch.bfloat16)), \
                 patch.object(torch,'autocast',side_effect=lambda *a,**k:contextlib.nullcontext()):
                runtime.hann_stage1_stream(SimpleNamespace(stage1=Stage()),'unused',meta,reader_cls=Reader,
                    reader_kind='frame_dir',chunk=4,stride=2,baton_path=root/'baton.f16',
                    initial_hint=None,initial_hint_quality=.95,full_hint_video_path='dense' if dense else None,
                    full_hint_quality=.8,rvm_path=None,foundation_paths=paths,carry_hint_from_aux=True,
                    carry_hint_quality=.6,on_baton_frames=None,hint_frames=keyframes,hint_precedence=precedence)
        return calls

    def test_missing_frames_are_absent_and_keyframes_override_carry(self):
        calls=self.exercise({2:dict(path='two',quality=.7),5:dict(path='five',quality=.8)})
        expected=[[-1,-1,.7,-1],[.7,.6,-1,.8]]
        for (_,quality),values in zip(calls,expected):
            torch.testing.assert_close(quality,torch.tensor([values],dtype=torch.bfloat16),rtol=0,atol=0)
        self.assertEqual(float(calls[1][0][0,0,0,0,0]),.75)
        self.assertEqual(float(calls[1][0][0,1,0,0,0]),.25)
        self.assertEqual(float(calls[1][0][0,2,0,0,0]),0)

    def test_dense_precedence_is_explicit_and_legacy_is_preserved(self):
        provided=self.exercise(dense=True,precedence='provided')
        carried=self.exercise(dense=True,precedence='carried')
        self.assertTrue(torch.all(provided[1][0]==.75))
        self.assertTrue(torch.all(carried[1][0][:,:2]==.25))
        self.assertTrue(torch.all(carried[1][0][:,2:]==.75))


if __name__=='__main__':unittest.main()
