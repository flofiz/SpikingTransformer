from .SSA import SSAMultiHeadAttention, MultiScaleXNORAttention
from .SpikingMLP import SpikingMLP
import torch.nn as nn
from typing import Literal
import torch

class DecoderLayer(nn.Module):
    """
    Decoder layer with Spiking Self-Attention, Cross-Attention and Spiking MLP.
    Accepts inputs of shape [T, B, N, D].
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        ff_dim: Feed-forward hidden dimension
        dropout: Dropout rate
        alpha: Alpha parameter for attention
        n_steps: Number of SNN timesteps
        mask_mode: "multiply" or "additive" for causal masking
        use_mssa: If True, use Multi-Scale Spiking Attention for self-attention
        mssa_scales: Scales for MSSA
        use_fused: If True, use fused Linear-LayerNorm-LIF kernels
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
        mssa_scales: list = [1, 2, 4],
        use_fused: bool = False
    ):
        super().__init__()
        
        self.self_attns = SSAMultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            alpha=alpha,
            n_steps=n_steps,
            mask_mode=mask_mode,
            use_fused=use_fused
        )
        
        # Cross-attention always uses standard SSA (not MSSA)
        self.cross_attns = SSAMultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            alpha=alpha,
            n_steps=n_steps,
            mask_mode=mask_mode,
            use_fused=use_fused
        )
        
        self.mlp = SpikingMLP(d_model=d_model, ff_dim=ff_dim, n_steps=n_steps, use_fused=use_fused)

    def forward(self, x, enc_output, mask=None):
        # x: [T, B, N, D]
        attn_output = self.self_attns(x, x, x, attention_mask=mask)
        x = x + attn_output

        cross_attn_output = self.cross_attns(x, enc_output, enc_output)
        x = x + cross_attn_output

        mlp_output = self.mlp(x)
        x = x + mlp_output

        return x

class Decoder(nn.Module):
    """
    Decoder consisting of multiple DecoderLayers.
    Accepts inputs of shape [T, B, N, D].
    
    Args:
        num_layers: Number of decoder layers
        d_model: Model dimension
        n_heads: Number of attention heads
        ff_dim: Feed-forward hidden dimension
        dropout: Dropout rate
        alpha: Alpha parameter for attention
        n_steps: Number of SNN timesteps
        mask_mode: "multiply" or "additive" for causal masking
        use_mssa: If True, use Multi-Scale Spiking Attention (only in first 2 layers)
        mssa_scales: Scales for MSSA
        use_fused: If True, use fused Linear-LayerNorm-LIF kernels
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
        mssa_scales: list = [1, 2, 4],
        gradient_checkpointing: bool = False,
        use_fused: bool = False
    ):
        super().__init__()
        
        self.gradient_checkpointing = gradient_checkpointing
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # Use MSSA only in first 2 layers (as per MSViT paper)
            layer_use_mssa = use_mssa and i < 2
            self.layers.append(
                DecoderLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                    alpha=alpha,
                    n_steps=n_steps,
                    mask_mode=mask_mode,
                    use_mssa=layer_use_mssa,
                    mssa_scales=mssa_scales,
                    use_fused=use_fused
                )
            )

    def forward(self, x, enc_output, mask=None):
        # x: [T, B, N, D]
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, enc_output, mask, use_reentrant=False)
            else:
                x = layer(x, enc_output, mask=mask)
        return x