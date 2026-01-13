import torch
import torch.nn as nn
from .SSA import SSAMultiHeadAttention, MultiScaleXNORAttention
from .SpikingMLP import SpikingMLP
from typing import Optional, Literal

class EncoderLayer(nn.Module):
    """
    Encoder layer with Spiking Self-Attention and Spiking MLP.
    Accepts inputs of shape [T, B, N, D].
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        ff_dim: Feed-forward hidden dimension
        dropout: Dropout rate
        alpha: Alpha parameter for attention
        n_steps: Number of SNN timesteps
        mask_mode: "multiply" or "additive" for causal masking
        use_mssa: If True, use Multi-Scale Spiking Attention instead of standard SSA
        mssa_scales: Scales for MSSA if use_mssa is True
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        alpha: float = 0.125,
        n_steps: int = 10,
        mask_mode: Literal["multiply", "additive"] = "multiply",
        use_mssa: bool = False,
        mssa_scales: list = [1, 2, 4]
    ):
        super().__init__()
        
        if use_mssa:
            self.self_attns = MultiScaleXNORAttention(
                d_model=d_model,
                n_heads=n_heads,
                scales=mssa_scales,
                n_steps=n_steps,
                dropout=dropout,
                mask_mode=mask_mode
            )
        else:
            self.self_attns = SSAMultiHeadAttention(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                alpha=alpha,
                n_steps=n_steps,
                mask_mode=mask_mode
            )
        
        self.mlp = SpikingMLP(d_model=d_model, ff_dim=ff_dim, n_steps=n_steps)

    def forward(self, x, mask=None):
        # x: [T, B, N, D]
        attn_output = self.self_attns(x, x, x, attention_mask=mask)
        x = x + attn_output

        mlp_output = self.mlp(x)
        x = x + mlp_output

        return x

class Encoder(nn.Module):
    """
    Encoder consisting of multiple EncoderLayers.
    Accepts inputs of shape [T, B, N, D].
    
    Args:
        num_layers: Number of encoder layers
        d_model: Model dimension
        n_heads: Number of attention heads
        ff_dim: Feed-forward hidden dimension
        dropout: Dropout rate
        alpha: Alpha parameter for attention
        n_steps: Number of SNN timesteps
        mask_mode: "multiply" or "additive" for causal masking
        use_mssa: If True, use Multi-Scale Spiking Attention (only in first 2 layers as per MSViT)
        mssa_scales: Scales for MSSA
    """
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        n_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        alpha: float = 0.125,
        n_steps: int = 10,
        mask_mode: Literal["multiply", "additive"] = "multiply",
        use_mssa: bool = False,
        mssa_scales: list = [1, 2, 4]
    ):
        super().__init__()
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # Use MSSA only in first 2 layers (as per MSViT paper)
            layer_use_mssa = use_mssa and i < 2
            self.layers.append(
                EncoderLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                    alpha=alpha,
                    n_steps=n_steps,
                    mask_mode=mask_mode,
                    use_mssa=layer_use_mssa,
                    mssa_scales=mssa_scales
                )
            )

    def forward(self, x, mask=None):
        # x: [T, B, N, D]
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x