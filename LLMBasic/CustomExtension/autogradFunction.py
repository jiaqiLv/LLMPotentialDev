"""
利用torch.autograd.Function实现算子(包括前向和反向传播过程)
"""

from typing import Any
import torch


class MultiplyAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, x, b) -> Any:
        ctx.save_for_backward(w,x)
        output = w * x + b
        return output
    
    @staticmethod
    def backward(ctx: Any, grad_outputs) -> Any:
        w,x = ctx.saved_tensors
        print('grad_outputs:', grad_outputs)
        grad_w = grad_outputs * x
        grad_x = grad_outputs * w
        grad_b = grad_outputs * 1
        return grad_w,grad_x,grad_b
    
def custom_op_realization():
    x = torch.ones((2,1),requires_grad=True)
    w = torch.rand((2,2),requires_grad=True)
    b = torch.rand((2,1),requires_grad=True)
    print('x|w|b:',x,w,b)
    print('forward propagation...')
    z = MultiplyAdd.apply(w,x,b)
    print('x|w|b|z:',x,w,b,z)
    print('backward propagation...')
    z.sum().backward()
    print('x.grad|w.grad|b.grad:',x.grad,w.grad,b.grad)
    
def tensor_cutting():
    t = torch.rand((3,6,12))
    t1, t2 = torch.chunk(t,2,dim=1)
    print(t.shape,t1.shape,t2.shape)
    
if __name__ == '__main__':
    tensor_cutting()