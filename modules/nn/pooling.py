import torch
import torch.nn as nn
import torch.nn.functional as F

from ..enums import eXplainMethod

class MaxPool2d(nn.Module):
    def __init__(self, kernel_size=2, stride=None, padding=0, dilation=1, ceil_mode=False):
        super(MaxPool2d, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, ceil_mode=ceil_mode, return_indices=True)
        self.unpool = nn.MaxUnpool2d(kernel_size=kernel_size, stride=stride, padding=padding)

        self.indices = None
        self.x = None

    def forward(self, x):
        self.x = x

        x, self.indices = self.pool(x)

        return x

    def backward(self, method, R):
        if method == eXplainMethod.guided_backprop:
            return self._guided_backprop_backward(R)

    def _guided_backprop_backward(self, R):
        B, C, H, W = self.x.shape
        H //= 2
        W //= 2

        if R.shape != torch.Size([B, C, H, W]):
            R = R.view(B, C, H, W)

        return self.unpool(R, self.indices)
