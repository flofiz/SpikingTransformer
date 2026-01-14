import math
import torch
from torch import nn
from einops import rearrange
from .lif_auto import LIF  # Auto-selects Triton or PyTorch fallback


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
    """
    Reduces spatial dimensions and channel count.
    
    Args:
        in_channels: Number of channels before rearrange (typically 64)
        height_after_unshuffle: Height dimension after PixelUnshuffle (e.g., H/patch_size)
        out_channels: Output channel dimension (d_model)
        n_steps: SNN timesteps
        threshold: LIF threshold
    """
    def __init__(self, in_channels=64, height_after_unshuffle=8, out_channels=512, n_steps=1, threshold=0.5):
        super().__init__()
        # Input channels after rearrange is: in_channels * height_after_unshuffle
        rearranged_channels = in_channels * height_after_unshuffle
        self.conv = nn.Conv2d(rearranged_channels, out_channels, kernel_size=1)
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
    """
    CNN Backbone for image feature extraction using Spiking Neural Networks.
    
    Args:
        d_model: Output model dimension
        nb_layers: Number of FusedInvertedBottleneck layers
        patch_size: Patch size for PixelUnshuffle (space-to-depth)
        n_steps: Number of SNN timesteps
        threshold: LIF neuron threshold
        in_channels: Number of input channels (1 for grayscale, 3 for RGB)
        img_height: Height of input image (needed to compute ReduceConvBlock channels)
    """
    def __init__(
        self,
        d_model: int = 512,
        nb_layers: int = 11,
        patch_size: int = 4,
        n_steps: int = 1,
        threshold: float = 0.5,
        in_channels: int = 1,
        img_height: int = 32
    ):
        super().__init__()
        self.nb_layers = nb_layers
        self.d_model = d_model
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.img_height = img_height
        
        # Calculate channels after PixelUnshuffle
        # For grayscale (1 channel): 1 * 16 = 16
        # For RGB (3 channels): 3 * 16 = 48
        unshuffle_channels = in_channels * (patch_size ** 2)
        
        # Height after PixelUnshuffle
        height_after_unshuffle = img_height // patch_size
        
        self.layers = nn.ModuleList([
            FusedInvertedBottleneck(n_steps=n_steps, threshold=threshold) 
            for _ in range(nb_layers)
        ])
        self.space_to_depth = nn.PixelUnshuffle(patch_size)
        
        # First conv adapts to input channels after unshuffle
        self.conv1 = nn.Conv2d(unshuffle_channels, 128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.lif_1 = LIF(n_steps=n_steps, beta=0.5)
        
        self.conv2 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        self.lif_2 = LIF(n_steps=n_steps, beta=0.5)
        
        # ReduceConvBlock now uses computed height
        self.reduce = ReduceConvBlock(
            in_channels=64, 
            height_after_unshuffle=height_after_unshuffle,
            out_channels=d_model, 
            n_steps=n_steps, 
            threshold=threshold
        )
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