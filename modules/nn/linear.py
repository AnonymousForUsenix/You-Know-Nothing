import torch
import torch.nn as nn

from ..enums import eXplainMethod

class Linear(nn.Module):
    def __init__(self, in_dim, out_dim, activation_fn=None, BN=False):
        super(Linear, self).__init__()

        self.linear = nn.Linear(in_dim, out_dim)

        self.BN = None
        if BN:
            self.BN = nn.BatchNorm1d(out_dim)

        self.activation_fn = activation_fn

        self.x = 0
        self.pre_activation = 0

        # nn.init.xavier_uniform_(self.linear.weight.data)
        # self.linear.bias.data.fill_(0)

    def forward(self, x):
        x = x.view(x.shape[0], -1)

        self.x = x

        x = self.linear(x)

        if self.BN is not None:
            x = self.BN(x)

        self.pre_activation = x

        if self.activation_fn is not None:
            x = self.activation_fn(x)

        return x

    def backward(self, method, R):
        if method == eXplainMethod.guided_backprop:
            return self._guided_backprop_backward(R)

    def _guided_backprop_backward(self, R):
        if self.activation_fn is not None:
            if hasattr(self.activation_fn, 'beta'):
                R = torch.nn.functional.softplus(R, self.activation_fn.beta) * torch.sigmoid(self.activation_fn.beta * self.pre_activation)
            else:
                R = torch.nn.functional.relu(R) * (self.pre_activation >= 0).float()

        weight = self.linear.weight
        if self.BN is not None:
            weight = weight * self.BN.weight.unsqueeze(1) / torch.sqrt(self.BN.running_var.unsqueeze(1) + self.BN.eps)

        return torch.matmul(R, weight)
