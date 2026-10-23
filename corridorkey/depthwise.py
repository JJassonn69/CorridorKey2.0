"""Exact ordered-FMA fast path for native CUDA BF16 depthwise convolution.

The implementation preserves the bias-first, row-major FP32 accumulation in
PyTorch 2.9.1's native 7x7 depthwise kernel. Backend selection remains PyTorch's;
cuDNN and unsupported precision, shape, stack, or gradient cases use conv2d.
No parameters or persistent activation buffers are added.
"""
import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

_SUPPORTED_STACK = (triton is not None
                    and torch.__version__.split('+')[0] == '2.9.1'
                    and triton.__version__ == '3.5.1'
                    and hasattr(torch._C, '_select_conv_backend'))

if triton is not None:
    @triton.jit
    def _valid7(X,W,B,Y,N:tl.constexpr,CI:tl.constexpr,CO:tl.constexpr,
                HI:tl.constexpr,WI:tl.constexpr,HO:tl.constexpr,WO:tl.constexpr,
                SX0:tl.constexpr,SX1:tl.constexpr,SX2:tl.constexpr,SX3:tl.constexpr,
                HAS_BIAS:tl.constexpr,BLOCK:tl.constexpr):
        i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
        valid=i<N*CO*HO*WO
        ox=i%WO;oy=i//WO%HO;oc=i//(WO*HO)%CO;batch=i//(WO*HO*CO)
        ic=oc//(CO//CI)
        base=batch*SX0+ic*SX1+oy*SX2+ox*SX3
        acc=tl.full((BLOCK,),0,tl.float32)
        if HAS_BIAS: acc=tl.load(B+oc,valid,0).to(tl.float32)
        for kh in tl.static_range(7):
            for kw in tl.static_range(7):
                x=tl.load(X+base+kh*SX2+kw*SX3,valid,0).to(tl.float32)
                w=tl.load(W+oc*49+kh*7+kw,valid,0).to(tl.float32)
                acc=tl.fma(w,x,acc)
        tl.store(Y+i,acc,valid)

def _pair(value):
    return (value, value) if isinstance(value, int) else tuple(value)


def depthwise_conv2d(input, weight, bias=None, stride=1, padding=0,
                     dilation=1, groups=1):
    """Use the exact native-kernel replacement only for the qualified case."""
    enabled = (_SUPPORTED_STACK and not torch.is_grad_enabled()
               and input.is_cuda and input.ndim == 4 and weight.ndim == 4
               and input.dtype in (torch.float32, torch.bfloat16)
               and weight.dtype in (torch.float32, torch.bfloat16)
               and weight.device == input.device
               and groups == input.shape[1] and groups > 1
               and weight.shape[1:] == (1, 7, 7)
               and weight.shape[0] % groups == 0
               and _pair(stride) == (1, 1) and _pair(padding) == (0, 0)
               and _pair(dilation) == (1, 1)
               and input.shape[0] > 0 and min(input.shape[2:]) >= 7
               and input.numel() < 2**31
               and (bias is None or (bias.device == input.device
                    and bias.dtype in (torch.float32, torch.bfloat16)
                    and bias.ndim == 1 and bias.numel() == weight.shape[0])))
    if enabled:
        autocast = torch.is_autocast_enabled('cuda')
        dtype = torch.get_autocast_dtype('cuda') if autocast else input.dtype
        enabled = (dtype == torch.bfloat16 and
                   (autocast or (weight.dtype == input.dtype and
                    (bias is None or bias.dtype == input.dtype))))
    if enabled:
        x, w = input.to(dtype), weight.to(dtype)
        b = bias.to(dtype) if bias is not None else None
        backend = torch._C._select_conv_backend(
            x, w, b, [1, 1], [0, 0], [1, 1], False, [0, 0], groups)
        enabled = backend == torch._C._ConvBackend.CudaDepthwise2d
    if not enabled:
        return F.conv2d(input, weight, bias, stride, padding, dilation, groups)
    n, ci, hi, wi = x.shape
    co, ho, wo = w.shape[0], hi - 6, wi - 6
    if n * co * ho * wo >= 2**31:
        return F.conv2d(input, weight, bias, stride, padding, dilation, groups)
    out = torch.empty((n, co, ho, wo), device=x.device, dtype=x.dtype)
    w = w.contiguous()
    b = b.contiguous() if b is not None else None
    with torch.cuda.device(x.device):
        _valid7[(triton.cdiv(out.numel(), 256),)](
            x, w, b, out, n, ci, co, hi, wi, ho, wo, *x.stride(),
            b is not None, 256, num_warps=4, enable_fp_fusion=False)
    return out


class DepthwiseConv2d(torch.nn.Conv2d):
    """Conv2d with identical parameters and gradient-enabled fallback."""
    def forward(self, input):
        if self.padding_mode != 'zeros':
            return super().forward(input)
        return depthwise_conv2d(input, self.weight, self.bias, self.stride,
                                self.padding, self.dilation, self.groups)
