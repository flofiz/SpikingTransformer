import torch
import torch.nn as nn
from .SSA import SSAMultiHeadAttention
from .SpikingMLP import SpikingMLP

class EncoderLayer(nn.Module):
    """
    Encoder layer with Spiking Self-Attention and Spiking MLP.
    Accepts inputs of shape [T, B, N, D].
    """
    def __init__(self, d_model, n_heads, ff_dim, dropout=0.1, alpha=0.125, n_steps=10):
        super().__init__()
        self.self_attns = SSAMultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout, alpha=alpha, n_steps=n_steps)
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
    """
    def __init__(self, num_layers, d_model, n_heads, ff_dim, dropout=0.1, alpha=0.125, n_steps=10):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model=d_model, n_heads=n_heads, ff_dim=ff_dim, dropout=dropout, alpha=alpha, n_steps=n_steps)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        # x: [T, B, N, D]
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x