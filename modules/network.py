import torch
import torch.nn as nn
import torch.nn.utils as U
import torch.nn.functional as F

import torchvision.models as models

import numpy as np

from .enums import eXplainMethod
from . import nn as NN

class VGG16_w_XAI(nn.Module):
    def __init__(self, num_classes, input_size=224):
        super(VGG16_w_XAI, self).__init__()

        self.num_classes = num_classes

        classifier_in_dim = 512 * ((input_size // 32) ** 2)

        layers = [
            NN.Conv2d(3, 64, 3, padding=1, activation_fn=F.relu),
            NN.Conv2d(64, 64, 3, padding=1, activation_fn=F.relu),
            NN.MaxPool2d(stride=2),
            NN.Conv2d(64, 128, 3, padding=1, activation_fn=F.relu),
            NN.Conv2d(128, 128, 3, padding=1, activation_fn=F.relu),
            NN.MaxPool2d(stride=2),
            NN.Conv2d(128, 256, 3, padding=1, activation_fn=F.relu),
            NN.Conv2d(256, 256, 3, padding=1, activation_fn=F.relu),
            NN.Conv2d(256, 256, 3, padding=1, activation_fn=F.relu),
            NN.MaxPool2d(stride=2),
            NN.Conv2d(256, 512, 3, padding=1, activation_fn=F.relu),
            NN.Conv2d(512, 512, 3, padding=1, activation_fn=F.relu),
            NN.Conv2d(512, 512, 3, padding=1, activation_fn=F.relu),
            NN.MaxPool2d(stride=2),
            NN.Conv2d(512, 512, 3, padding=1, activation_fn=F.relu),
            NN.Conv2d(512, 512, 3, padding=1, activation_fn=F.relu),
            NN.Conv2d(512, 512, 3, padding=1, activation_fn=F.relu),
            NN.MaxPool2d(stride=2),
            NN.Linear(classifier_in_dim, 4096, activation_fn=F.relu),
            nn.Dropout(0.5),
            NN.Linear(4096, 4096, activation_fn=F.relu),
            nn.Dropout(0.5),
            NN.Linear(4096, num_classes)
        ]

        self.layers = nn.ModuleList(layers)

        self.R = 0

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        self.R = x

        return x

    def backward(self, method=eXplainMethod.guided_backprop, R=None, target=None):
        if R is None:
            R = self.R

        if target is not None:
            R = self.R.clone()

            R = F.softmax(R, dim=1)

        for layer in reversed(self.layers):
            if type(layer) == nn.Dropout:
                continue

            R = layer.backward(method, R)

        return R

    def explain(self, x, method, target=None):
        y = self.forward(x)

        confidence = F.softmax(y, dim=1)
        prediction = torch.max(y, 1)[1]

        if target is None:
            target = prediction

        ex_map = self.backward(method=method, R=None, target=target)

        map_max = torch.max(torch.max(torch.max(torch.abs(ex_map), dim=3)[0], dim=2)[0], dim=1)[0]
        normalized_map = ex_map * torch.pow(10, -torch.floor(torch.log10(map_max))).view(-1, 1, 1, 1)

        normalized_map = torch.clip(normalized_map, min=-1.0, max=1.0)

        return normalized_map, confidence, prediction

class VGG16_w_XAI_n_DP(nn.Module):
    def __init__(self, num_classes, epsilon, delta, input_size=224):
        super(VGG16_w_XAI_n_DP, self).__init__()

        self.num_classes = num_classes
        self.std = (2 * np.log(1.25 / delta) / (epsilon ** 2))

        classifier_in_dim = 512 * ((input_size // 32) ** 2)

        layers = [
            NN.Conv2d(3, 64, 3, padding=1, activation_fn=F.relu),
            NN.Conv2d(64, 64, 3, padding=1, activation_fn=F.relu),
            NN.MaxPool2d(stride=2),
            NN.Conv2d(64, 128, 3, padding=1, activation_fn=F.relu),
            NN.Conv2d(128, 128, 3, padding=1, activation_fn=F.relu),
            NN.MaxPool2d(stride=2),
            NN.Conv2d(128, 256, 3, padding=1, activation_fn=F.relu),
            NN.Conv2d(256, 256, 3, padding=1, activation_fn=F.relu),
            NN.Conv2d(256, 256, 3, padding=1, activation_fn=F.relu),
            NN.MaxPool2d(stride=2),
            NN.Conv2d(256, 512, 3, padding=1, activation_fn=F.relu),
            NN.Conv2d(512, 512, 3, padding=1, activation_fn=F.relu),
            NN.Conv2d(512, 512, 3, padding=1, activation_fn=F.relu),
            NN.MaxPool2d(stride=2),
            NN.Conv2d(512, 512, 3, padding=1, activation_fn=F.relu),
            NN.Conv2d(512, 512, 3, padding=1, activation_fn=F.relu),
            NN.Conv2d(512, 512, 3, padding=1, activation_fn=F.relu),
            NN.MaxPool2d(stride=2),
            NN.Linear(classifier_in_dim, 4096, activation_fn=F.relu),
            nn.Dropout(0.5),
            NN.Linear(4096, 4096, activation_fn=F.relu),
            nn.Dropout(0.5),
            NN.Linear(4096, num_classes)
        ]

        self.layers = nn.ModuleList(layers)

        self.R = 0

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        self.R = x

        return x

    def backward(self, method=eXplainMethod.guided_backprop, R=None, target=None):
        if R is None:
            R = self.R

        if target is not None:
            R = self.R.clone()

            R = F.softmax(R, dim=1)
            R = R + torch.randn_like(R) * self.std
            # R = F.softmax(R, dim=1)

        for layer in reversed(self.layers):
            if type(layer) == nn.Dropout:
                continue

            R = layer.backward(method, R)

        return R

    def explain(self, x, method, target=None):
        y = self.forward(x)

        confidence = F.softmax(y, dim=1)
        prediction = torch.max(y, 1)[1]

        if target is None:
            target = prediction

        ex_map = self.backward(method=method, R=None, target=target)

        map_max = torch.max(torch.max(torch.max(torch.abs(ex_map), dim=3)[0], dim=2)[0], dim=1)[0]
        normalized_map = ex_map * torch.pow(10, -torch.floor(torch.log10(map_max))).view(-1, 1, 1, 1)

        normalized_map = torch.clip(normalized_map, min=-1.0, max=1.0)

        return normalized_map, confidence, prediction

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        ]

        self.block = nn.Sequential(*block)

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, n_channels=3, n_residual_blocks=9):
        super(Generator, self).__init__()

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(n_channels, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]

            in_features = out_features
            out_features = in_features * 2

        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]

            in_features = out_features
            out_features = in_features // 2

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, n_channels, 7)
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return (x + self.model(x)).tanh()

class Discriminator(nn.Module):
    def __init__(self, n_channels=3):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(n_channels, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512),
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0]) #global avg pool

class SqueezeExcitationLinear(nn.Module):
    def __init__(self, feature_in, feature_out, activation_squeeze=nn.ReLU, activation_excitation=nn.Sigmoid):
        super(SqueezeExcitationLinear, self).__init__()

        n_hidden = max(feature_out // 16, 4)

        ops = []

        ops.append(nn.Linear(feature_in, n_hidden))
        ops.append(activation_squeeze())
        ops.append(nn.Linear(n_hidden, feature_out))
        ops.append(activation_excitation())

        self.ops = nn.Sequential(*ops)

    def forward(self, x):
        x_hat = torch.mean(x, dim=(2, 3))
        x_hat = x_hat.view(x_hat.size(0), -1)
        x_hat = self.ops(x_hat)
        x_hat = x_hat.view(x_hat.size(0), -1, 1, 1)

        return x * x_hat


class Conv2D(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, weight_norm=True):
        super(Conv2D, self).__init__()

        if weight_norm:
            self.ops = U.spectral_norm(nn.Conv2d(channel_in, channel_out, kernel_size, stride, padding, dilation, groups, bias))

        else:
            self.ops = nn.Conv2d(channel_in, channel_out, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        return self.ops(x)


class ConvTranspose2D(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, weight_norm=False):
        super(ConvTranspose2D, self).__init__()

        if weight_norm:
            self.ops = U.spectral_norm(nn.ConvTranspose2d(channel_in, channel_out, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))

        else:
            self.ops = nn.ConvTranspose2d(channel_in, channel_out, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        return self.ops(x)


class ResidualCell(nn.Module):
    def __init__(self, channel_in, channel_out, hidden_ratio=1.0):
        super(ResidualCell, self).__init__()

        channel_hidden = int(round(channel_in * hidden_ratio))

        ops = [
            nn.BatchNorm2d(channel_in),
            nn.SiLU(),
            Conv2D(channel_in, channel_hidden, kernel_size=1),
            nn.BatchNorm2d(channel_hidden),
            nn.SiLU(),
            Conv2D(channel_hidden, channel_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_hidden),
            nn.SiLU(),
            Conv2D(channel_hidden, channel_in, kernel_size=1),
            nn.BatchNorm2d(channel_in),
            SqueezeExcitationLinear(channel_in, channel_in),
        ]

        self.ops = nn.Sequential(*ops)

    def forward(self, x):
        # print(x[0])
        x_hat = self.ops(x)
        # print(x_hat[0])

        return x + x_hat


class EncCombiner(nn.Module):
    def __init__(self, channel_in_1, channel_in_2, channel_out):
        super(EncCombiner, self).__init__()

        self.ops = Conv2D(channel_in_2, channel_out, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x_1, x_2):
        x_2 = self.ops(x_2)

        return x_1 + x_2


class DecCombiner(nn.Module):
    def __init__(self, channel_in_1, channel_in_2, channel_out):
        super(DecCombiner, self).__init__()

        self.ops = Conv2D(channel_in_1 + channel_in_2, channel_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x_1, x_2):
        x = torch.cat([x_1, x_2], dim=1)

        x = self.ops(x)
        # print("=" * 10)
        # print(x[0])
        return x


class VAE_Encoder(nn.Module):
    def __init__(self, channel_in=3, n_process=3, n_residue=3, channel_hidden=32):
        super(VAE_Encoder, self).__init__()

        channels = channel_hidden

        process_block = []

        for i in range(n_process):
            channels = channels * 2 if i != 0 else channels

            process_block += [
                Conv2D(channel_in if i == 0 else channels // 2, channels, 3, stride=2, padding=1),
                ResidualCell(channels, channels, hidden_ratio=1.0)
            ]

        self.process_block = nn.Sequential(*process_block)

        residual_block = []

        for i in range(n_residue):
            residual_block += [ResidualCell(channels, channels, hidden_ratio=1.0)]

        self.residual_block = nn.ModuleList(residual_block)


    def forward(self, x):
        z = []
        x = self.process_block(x)

        for cell in self.residual_block:
            x = cell(x)
            z.append(x)

        z.reverse()

        return z


class VAE_Generator(nn.Module):
    def __init__(self, channel_out=3, n_process=3, n_residue=3, channel_hidden=32):
        super(VAE_Generator, self).__init__()

        self.n_residue = n_residue

        channels = channel_hidden * (2 ** (n_process-1))

        sampler_block = []
        residual_block = []

        for i in range(n_residue):
            sampler_block += [Conv2D(channels, 2 * channels, 1)]

            decode_block = [
                DecCombiner(channels, channels, channels),
                ResidualCell(channels, channels, hidden_ratio=1.0)
            ] if i != 0 else [
                ResidualCell(channels, channels, hidden_ratio=1.0)
            ]

            residual_block += [nn.ModuleList(decode_block)]

        self.sampler_block = nn.ModuleList(sampler_block)
        self.residual_block = nn.ModuleList(residual_block)

        process_block = []

        for i in range(n_process):
            if i != list(range(n_process))[-1]:
                process_block += [
                    ConvTranspose2D(channels, channels // 2, 2, stride=2),
                    ResidualCell(channels // 2, channels // 2, hidden_ratio=1.0)
                ]

                channels //= 2
            else:
                process_block += [
                    ConvTranspose2D(channels, channels, 2, stride=2),
                    ResidualCell(channels, channels, hidden_ratio=1.0),
                    nn.Conv2d(channels, channel_out, 1)
                ]

        self.process_block = nn.Sequential(*process_block)

    def forward(self, z):

        assert len(z) == self.n_residue

        mu, log_sig = [], []

        for i in range(self.n_residue):
            z_hat = self.sampler_block[i](z[i])
            mu_t, log_sig_t = torch.chunk(z_hat, 2, dim=1)

            mu.append(mu_t)
            log_sig.append(log_sig_t)

            std = torch.exp(0.5 * log_sig_t)

            s = torch.randn_like(std) * std + mu_t

            if i == 0:
                x = self.residual_block[i][0](s)
            else:
                x = self.residual_block[i][0](x, s)
                x = self.residual_block[i][1](x)



        x = self.process_block(x)

        return x.tanh(), mu, log_sig
