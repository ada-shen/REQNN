import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import gradcheck
import copy

class our_mpool1d(Function):
    @staticmethod
    def forward(ctx, x, kernel_size = torch.IntTensor([2]), stride = torch.IntTensor([2])):

        x_detach = x.detach()
        inSize = x_detach.size()
        batch_size = inSize[0]//3

        inSize = torch.IntTensor([inSize[0], inSize[1], inSize[2], inSize[3]])

        x_detach = x_detach.view(3,batch_size,inSize[1],inSize[2],inSize[3]).permute(1,0,2,3,4).contiguous()
        mod = torch.sqrt(torch.sum(x_detach*x_detach, dim = 1))

        mod = mod.cuda()
        mod, indices = F.max_pool2d(mod, kernel_size, stride, return_indices = True)

        modSize = mod.size()
        ctx.save_for_backward(x, indices)
        x = x.view(inSize[0], inSize[1], -1)
        indices = indices.view(modSize[0], modSize[1], -1)
        indices = torch.cat((indices, indices,indices), 0)
        y = torch.gather(x, 2, indices)
        y = y.view(modSize[0] * 3, modSize[1], modSize[2])
        return y
    @staticmethod
    def backward(ctx, grad_outputs):
        grad_outputs = grad_outputs.cuda()
        x, indices = ctx.saved_variables
        inSize = x.size()

        grad_inputs = copy.deepcopy(x.detach())
        grad_inputs.zero_()
        grad_inputs = grad_inputs.cuda()
        grad_inputs = grad_inputs.view(inSize[0], inSize[1], -1)
        grad_outputs = grad_outputs.view(inSize[0], inSize[1], -1)
        indices = indices.view(inSize[0] // 3, inSize[1], -1)
        indices = torch.cat((indices, indices, indices), 0)
        grad_inputs = grad_inputs.scatter_(2, indices, grad_outputs)
        grad_inputs = grad_inputs.view(inSize[0],inSize[1],inSize[2],inSize[3])
        grad_inputs = grad_inputs.cuda()
        # print(grad_inputs)
        return grad_inputs, None, None

class our_mpool2d(Function):
    @staticmethod
    def forward(ctx, x, kernel_size = torch.IntTensor([2]), stride = torch.IntTensor([2])):
        x_detach = x.detach()
        inSize = x_detach.size()
        batch_size = inSize[0]//3

        inSize = torch.IntTensor([inSize[0], inSize[1], inSize[2], inSize[3]])

        x_detach = x_detach.view(batch_size,3,inSize[1],inSize[2],inSize[3])
        mod = torch.sqrt(torch.sum(x_detach*x_detach, dim = 1))

        mod = mod.cuda()
        mod, indices = F.max_pool2d(mod, kernel_size, stride, return_indices = True)

        modSize = mod.size()
        ctx.save_for_backward(x, indices)
        x = x.view(inSize[0], inSize[1], -1)
        indices = indices.view(modSize[0],1,modSize[1],-1).repeat(1,3,1,1).view(inSize[0],modSize[1],-1)
        y = torch.gather(x, 2, indices)
        y = y.view(modSize[0] * 3, modSize[1], modSize[2])
        return y

    @staticmethod
    def backward(ctx, grad_outputs):
        grad_outputs = grad_outputs.cuda()
        x, indices = ctx.saved_variables
        inSize = x.size()

        grad_inputs = copy.deepcopy(x.detach())
        grad_inputs.zero_()
        grad_inputs = grad_inputs.cuda()
        grad_inputs = grad_inputs.view(inSize[0], inSize[1], -1)
        grad_outputs = grad_outputs.view(inSize[0], inSize[1], -1)
        indices = indices.view(inSize[0]//3,1,inSize[1],-1).repeat(1,3,1,1).view(inSize[0],inSize[1],-1)
        grad_inputs = grad_inputs.scatter_(2, indices, grad_outputs)
        grad_inputs = grad_inputs.view(inSize[0],inSize[1],inSize[2],inSize[3])
        grad_inputs = grad_inputs.cuda()
        # print(grad_inputs)
        return grad_inputs, None, None

class our_mpool(nn.Module):
    def __init__(self, kernel_size = 2, stride = 2):
        super(our_mpool, self).__init__()
        self.kernel_size = torch.IntTensor([kernel_size])
        self.stride = torch.IntTensor([stride])

    def forward(self, x, isEncrypt = True):
        if not isEncrypt:
            x = F.max_pool2d(x, self.kernel_size.item(), self.stride.item())
            return x
        else:
            x = our_mpool2d.apply(x, self.kernel_size, self.stride)
            return x

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = str(2)
    device = torch.device(2)
    x = torch.randn(2, 2, 5, 5, requires_grad = True).cuda()
    pool = our_mpool(3, 2).cuda()
    y = pool(x)
    print(x)
    print(y)
    # print(x.size())
    # print(y.size())
    z = y.mean()
    z.backward()
