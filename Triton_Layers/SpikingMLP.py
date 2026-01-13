import torch.nn as nn
from .Lif import LIF

# Try to import fused kernel (may fail on non-Linux or without Triton)
try:
    from .FusedLinearLayerNormLIF import FusedLinearLayerNormLIF
    FUSED_AVAILABLE = True
except ImportError:
    FUSED_AVAILABLE = False


class SpikingMLP(nn.Module):
    """
    MLP converted to use Spiking Triton (Linear + LIF).
    Accepts inputs of shape [T, B, N, D] and applies pointwise FFN with residual.
    
    Args:
        use_fused: If True, use fused Linear-LayerNorm-LIF kernel
    """
    def __init__(self, d_model, ff_dim, n_steps=1, use_fused=False):
        super().__init__()
        self.n_steps = n_steps
        self.use_fused = use_fused and FUSED_AVAILABLE
        self.d_model = d_model
        self.ff_dim = ff_dim

        if self.use_fused:
            # Fused: Linear + LayerNorm + LIF in one kernel
            self.expand_fused = FusedLinearLayerNormLIF(
                in_features=d_model,
                out_features=ff_dim,
                n_steps=n_steps,
                learn_beta=True
            )
            self.compress_fused = FusedLinearLayerNormLIF(
                in_features=ff_dim,
                out_features=d_model,
                n_steps=n_steps,
                learn_beta=True
            )
        else:
            # Separate layers (original implementation)
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
        original_shape = x.shape
        
        if self.use_fused:
            # Flatten for fused kernel: [T, B, N, D] -> [T*B*N, D]
            # Use reshape instead of view for non-contiguous tensors
            x_flat = x.reshape(-1, self.d_model)
            
            # Expand: d_model -> ff_dim
            x_flat, _ = self.expand_fused(x_flat)
            
            # Compress: ff_dim -> d_model
            x_flat, _ = self.compress_fused(x_flat)
            
            # Reshape back
            x = x_flat.reshape(original_shape)
        else:
            # Original: separate Linear -> LayerNorm -> LIF
            x = self.expand(x)         # pointwise linear on last dim
            x = self.bn_expand(x)      # LayerNorm on last dim
            x, _ = self.lifexpand(x)   # spiking non-linearity

            x = self.compress(x)
            x = self.bn_compress(x)
            x, _ = self.lifcompress(x)

        return x
    
    def extra_repr(self) -> str:
        return (
            f'd_model={self.d_model}, '
            f'ff_dim={self.ff_dim}, '
            f'n_steps={self.n_steps}, '
            f'use_fused={self.use_fused}'
        )