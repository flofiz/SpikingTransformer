import math
import torch
from torch import nn
from einops import rearrange
from .Lif import LIF


class FusedInvertedBottleneck(nn.Module):
    def __init__(self, n_steps=1, threshold=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 512, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.lif_1 = LIF(n_steps=n_steps, beta=0.5)
        self.conv2 = nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        self.lif_2 = LIF(n_steps=n_steps, beta=0.5)


    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out, _ = self.lif_1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out, _ = self.lif_2(out)
        out = out + identity
        return out


class ReduceConvBlock(nn.Module):
    def __init__(self, in_channels=1, out_channels=512, n_steps=1, threshold=0.5):
        super().__init__()
        self.conv = nn.Conv2d(64 * 8, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.lif = LIF(n_steps=n_steps, beta=0.5)

    def forward(self, x):
        x = rearrange(x, "b c h w -> b (c h) w").unsqueeze(-2)
        out = self.conv(x)
        out = self.bn(out)
        out = out.squeeze(-2)
        out, _ = self.lif(out)
        return out


class RPE2D(nn.Module):
    def __init__(self, d_model, stride=1, n_steps=1, threshold=0.5):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, stride=stride, padding=1)
        self.lif = LIF(n_steps=n_steps, beta=0.5)
        self.bn = nn.BatchNorm1d(d_model)

    def get_mem(self):
        return {"mem": getattr(self.lif, "mem", None)}

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x, _ = self.lif(x)
        return x


class CNNBackbone(nn.Module):
    def __init__(self, d_model=512, nb_layers=11, patch_size=4, n_steps=1, threshold=0.5, in_channels=1):
        super().__init__()
        self.nb_layers = nb_layers
        self.d_model = d_model
        self.layers = nn.ModuleList([FusedInvertedBottleneck(n_steps=n_steps, threshold=threshold) for _ in range(nb_layers)])
        self.space_to_depth = nn.PixelUnshuffle(patch_size)
        
        # Input: [B, C, H, W] -> PixelUnshuffle -> [B, C * patch_size^2, H/p, W/p]
        pixel_unshuffle_dim = in_channels * (patch_size ** 2)
        
        self.conv1 = nn.Conv2d(pixel_unshuffle_dim, 128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.lif_1 = LIF(n_steps=n_steps, beta=0.5)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        self.lif_2 = LIF(n_steps=n_steps, beta=0.5)
        self.reduce = ReduceConvBlock(64, d_model, n_steps=n_steps, threshold=threshold)
        self.rpe = RPE2D(d_model, n_steps=n_steps, threshold=threshold)


    def forward(self, x):
        record = {}
        x = self.space_to_depth(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x, _ = self.lif_1(x)
        record["conv1"] = x
        x = self.conv2(x)
        x = self.bn2(x)
        x, _ = self.lif_2(x)
        record["conv2"] = x
        record["FusedInvertedBottleneck"] = []
        for layer in self.layers:
            x = layer(x)
            record["FusedInvertedBottleneck"].append(x)
        x = self.reduce(x)

        rpe_value = self.rpe(x)
        x = x + rpe_value
        x = x.transpose(-1, -2)
        record["RPE2D"] = rpe_value
        record["output"] = x
        return x, record