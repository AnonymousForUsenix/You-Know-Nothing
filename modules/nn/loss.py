import torch
import torch.nn as nn
import torch.nn.functional as F

class KLDLoss(nn.Module):
    def __init__(self):
        super(KLDLoss, self).__init__()

    def forward(self, mu, log_sig):

        assert len(mu) == len(log_sig)

        l = len(mu)

        for i in range(l):
            sum = -torch.mean(0.5 * torch.sum(1 + log_sig[i] - mu[i] ** 2 - log_sig[i].exp(), dim=1), dim=0) if i == 0 else sum + -torch.mean(0.5 * torch.sum(1 + log_sig[i] - mu[i] ** 2 - log_sig[i].exp(), dim=1), dim=0)

        # print(torch.sum(-log_sig[0].exp(), dim=1))

        return sum.mean()
