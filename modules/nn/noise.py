import numpy as np

import torch
import torch.nn as nn

class AdditiveGaussianMechanism(nn.Module):
    def __init__(self, epslion=0.1, delta=0, sensitivity=1):
        super(AdditiveGaussianMechanism, self).__init__()

        self.std = (2 * np.log(1.25 / delta) * (sensitivity ** 2)) / (epslion ** 2)

    def forward(self, x):
        return torch.clip(x + torch.randn_like(x) * self.std, min=-1.0, max=1.0)

class GaussianNoiseAdder(nn.Module):
    def __init__(self, std):
        super(GaussianNoiseAdder, self).__init__()

        self.std = std

    def forward(self, x):
        return torch.clip(x + torch.randn_like(x) * self.std, min=-1.0, max=1.0)