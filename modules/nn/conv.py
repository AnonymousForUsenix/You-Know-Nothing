import torch
import torch.nn as nn
import torch.nn.functional as F

from ..enums import eXplainMethod

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, bn=True, activation_fn=None):
        super(Conv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

        self.activation_fn = activation_fn

        if bn:
            self.bn = nn.BatchNorm2d(num_features=out_channels)
        else:
            self.bn = None

        self.x = 0
        self.pre_activation = 0

        # nn.init.xavier_uniform_(self.conv.weight.data)
        # self.conv.bias.data.fill_(0)

    def forward(self, x):
        self.x = x

        x = self.conv(x)

        if self.bn is not None:
            x = self.bn(x)

        self.pre_activation = x

        if self.activation_fn is not None:
            x = self.activation_fn(x)

        return x

    def backward(self, method, R):
        B, C, H, W = self.pre_activation.shape

        if R.shape != torch.Size([B, self.conv.out_channels, H, W]):
            R = R.view(B, self.conv.out_channels, H, W)

        if method == eXplainMethod.guided_backprop:
            return self._guided_backprop_backward(R)

    def _guided_backprop_backward(self, R):
        if self.activation_fn is not None:
            if hasattr(self.activation_fn, 'beta'):
                R = torch.nn.functional.softplus(R, self.activation_fn.beta) * torch.sigmoid(self.activation_fn.beta * self.pre_activation)
            else:
                R = torch.nn.functional.relu(R) * (self.pre_activation >= 0).float()

        return self.deconv(R, self.conv.weight)


    def deconv(self, y, weights):
        _, _, H_x, W_x = self.x.shape

        padding = self.conv.padding
        stride = self.conv.stride

        _, _, H_f, W_f = weights.shape

        output_padding = ((H_x + 2 * padding[0] - H_f) % stride[0],
                            (W_x + 2 * padding[1] - W_f) % stride[1])

        return F.conv_transpose2d(input=y, weight=weights, bias=None, padding=self.conv.padding, stride=self.conv.stride, groups=self.conv.groups, dilation=self.conv.dilation, output_padding=output_padding)
