"""Exact arithmetic, dispatch boundaries and training fallback for the kernel."""
import unittest
from unittest import mock
import torch
import torch.nn.functional as F
from corridorkey import depthwise as module

@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class NativeDepthwiseTests(unittest.TestCase):
    def setUp(self): torch.manual_seed(714)

    @unittest.skipUnless(module._SUPPORTED_STACK, 'PyTorch 2.9.1 / Triton 3.5.1 required')
    def test_exact_native_shapes_strides_bias_and_autocast(self):
        self.assertTrue(module._SUPPORTED_STACK)
        with torch.no_grad():
            for n,c,h,w in ((1,16,15,17),(2,32,38,54),(3,64,11,19)):
                for mult in (1,2):
                    for dtype in (torch.bfloat16,torch.float32):
                        x=torch.randn(n,c,h,w*2,device='cuda',dtype=dtype)[:,:,:,::2]
                        weight=torch.randn(c*mult,1,7,7,device='cuda',dtype=dtype)
                        for bias in (None,torch.randn(c*mult,device='cuda',dtype=dtype)):
                            with torch.autocast('cuda',dtype=torch.bfloat16):
                                ref=F.conv2d(x,weight,bias,groups=c)
                                actual=module.depthwise_conv2d(x,weight,bias,groups=c)
                            self.assertTrue(torch.equal(ref.view(torch.int16),actual.view(torch.int16)))
                            self.assertEqual(ref.stride(),actual.stride())

    @unittest.skipUnless(module._SUPPORTED_STACK, 'PyTorch 2.9.1 / Triton 3.5.1 required')
    def test_tiny_signed_values_and_zero(self):
        with torch.no_grad():
            for scale in (0.,2.**-40,2.**-70):
                x=(torch.randn(1,16,13,17,device='cuda')*scale).bfloat16()
                w=(torch.randn(32,1,7,7,device='cuda')*scale).bfloat16()
                ref=F.conv2d(x,w,groups=16)
                actual=module.depthwise_conv2d(x,w,groups=16)
                self.assertTrue(torch.equal(ref.view(torch.int16),actual.view(torch.int16)))

    def test_cudnn_layout_and_other_precision_fall_back(self):
        with torch.no_grad(),mock.patch.object(module,'_valid7',object(),create=True):
            for dtype,layout in ((torch.bfloat16,torch.channels_last),(torch.float32,torch.contiguous_format),(torch.float64,torch.contiguous_format)):
                x=torch.randn(1,16,13,17,device='cuda',dtype=dtype).contiguous(memory_format=layout)
                w=torch.randn(32,1,7,7,device='cuda',dtype=dtype)
                self.assertTrue(torch.equal(F.conv2d(x,w,groups=16),module.depthwise_conv2d(x,w,groups=16)))

    def test_gradients_and_unsupported_stack_fall_back(self):
        x=torch.randn(1,16,13,17,device='cuda',dtype=torch.bfloat16,requires_grad=True)
        w=torch.randn(32,1,7,7,device='cuda',dtype=torch.bfloat16,requires_grad=True)
        with mock.patch.object(module,'_valid7',object(),create=True):
            actual=module.depthwise_conv2d(x,w,groups=16)
            self.assertIsNotNone(actual.grad_fn)
            self.assertTrue(torch.equal(actual,F.conv2d(x,w,groups=16)))
            with torch.no_grad(),mock.patch.object(module,'_SUPPORTED_STACK',False):
                self.assertTrue(torch.equal(module.depthwise_conv2d(x,w,groups=16),F.conv2d(x,w,groups=16)))

    def test_other_kernel_stride_padding_and_batch_fall_back(self):
        with torch.no_grad(),mock.patch.object(module,'_valid7',object(),create=True):
            for kernel,stride,padding,n in ((3,1,0,1),(7,2,0,1),(7,1,2,1),(7,1,0,0)):
                x=torch.randn(n,16,17,19,device='cuda',dtype=torch.bfloat16)
                w=torch.randn(16,1,kernel,kernel,device='cuda',dtype=torch.bfloat16)
                expected=F.conv2d(x,w,stride=stride,padding=padding,groups=16)
                actual=module.depthwise_conv2d(x,w,stride=stride,padding=padding,groups=16)
                self.assertTrue(torch.equal(expected,actual))


class DepthwiseModuleTests(unittest.TestCase):
    def test_module_state_and_cpu_training_gradients(self):
        original=torch.nn.Conv2d(4,8,7,groups=4,padding=2)
        candidate=module.DepthwiseConv2d(4,8,7,groups=4,padding=2)
        candidate.load_state_dict(original.state_dict(),strict=True)
        self.assertEqual(list(candidate.state_dict()),list(original.state_dict()))
        x=torch.randn(2,4,13,15,requires_grad=True);y=x.detach().clone().requires_grad_()
        a=original(x);b=candidate(y)
        self.assertTrue(torch.equal(a,b))
        a.sum().backward();b.sum().backward()
        self.assertTrue(torch.equal(x.grad,y.grad))
        self.assertTrue(torch.equal(original.weight.grad,candidate.weight.grad))

if __name__=='__main__': unittest.main()
