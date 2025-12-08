import torch.nn as nn
from .Lif import LIF

class SpikingMLP(nn.Module):
    """
    MLP converted to use Spiking Triton (Linear + LIF).
    Accepts inputs of shape [T, B, N, D] and applies pointwise FFN with residual.
    """
    def __init__(self, d_model, ff_dim, n_steps=1):
        super().__init__()
        self.n_steps = n_steps

        # dense layers that support [T*B, ..., D]
        self.expand = nn.Linear(d_model, ff_dim)
        self.compress = nn.Linear(ff_dim, d_model)

        # LIF layers from Triton
        self.lifexpand = LIF(n_steps=n_steps, learn_beta=True)
        self.lifcompress = LIF(n_steps=n_steps, learn_beta=True)

        # keep LayerNorm for stable activations (works on last dim)
        self.bn_expand = nn.LayerNorm(ff_dim)
        self.bn_compress = nn.LayerNorm(d_model)


    def forward(self, x):
        # x: [T, B, N, D]
        identity = x

        x = self.expand(x)         # pointwise linear on last dim
        x = self.bn_expand(x)      # LayerNorm on last dim
        x, _ = self.lifexpand(x)   # spiking non-linearity

        x = self.compress(x)
        x = self.bn_compress(x)
        x, _ = self.lifcompress(x)

        return x