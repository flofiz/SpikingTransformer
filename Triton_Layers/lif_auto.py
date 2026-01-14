"""
LIF (Leaky Integrate-and-Fire) Module with Triton/PyTorch Backend Selection

This module provides a unified interface for LIF neurons that automatically
selects between Triton-accelerated and pure PyTorch implementations based
on availability.

Usage:
    from Triton_Layers.lif_auto import LIF
    
    lif = LIF(beta=0.9, v_th=1.0, n_steps=8)
"""

import warnings
import torch
import torch.nn as nn
from typing import Tuple

# ============================================================================
# Check Triton availability
# ============================================================================
TRITON_AVAILABLE = False
_triton_import_error = None

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError as e:
    _triton_import_error = str(e)
    TRITON_AVAILABLE = False


def is_triton_available() -> bool:
    """Returns True if Triton is available and functional."""
    return TRITON_AVAILABLE


# ============================================================================
# PyTorch Fallback Implementation
# ============================================================================

class LIFLayerPyTorch(torch.autograd.Function):
    """
    Pure PyTorch implementation of LIF neuron with surrogate gradient.
    Used as fallback when Triton is not available.
    """
    
    @staticmethod
    def forward(ctx, input_current, beta, v_th, v_reset, k_superspike):
        """
        Forward pass of LIF neuron.
        
        Args:
            input_current: (T, B, N) input currents
            beta: leak factor (scalar tensor)
            v_th: threshold voltage (scalar tensor)
            v_reset: reset voltage (scalar tensor)
            k_superspike: surrogate gradient sharpness
            
        Returns:
            output_spikes: (T, B, N) binary spikes
            v_mem_final: (B, N) final membrane potential
        """
        T, B, N = input_current.shape
        device = input_current.device
        dtype = input_current.dtype
        
        # Initialize outputs
        output_spikes = torch.zeros_like(input_current)
        v_mem = torch.zeros(B, N, device=device, dtype=dtype)
        v_mem_history = torch.zeros(T, B, N, device=device, dtype=dtype)
        
        # Get scalar values
        beta_val = beta.item() if hasattr(beta, 'item') else beta
        v_th_val = v_th.item() if hasattr(v_th, 'item') else v_th
        v_reset_val = v_reset.item() if hasattr(v_reset, 'item') else v_reset
        
        # Forward through time
        for t in range(T):
            # Leak and integrate
            v_mem = v_mem * beta_val + input_current[t]
            v_mem_history[t] = v_mem
            
            # Spike generation (hard threshold)
            spike = (v_mem > v_th_val).float()
            output_spikes[t] = spike
            
            # Reset on spike
            v_mem = torch.where(spike > 0.5, 
                               torch.full_like(v_mem, v_reset_val), 
                               v_mem)
        
        # Save for backward
        ctx.save_for_backward(input_current, output_spikes, v_mem_history, 
                              beta, v_th, v_reset)
        ctx.k_superspike = k_superspike
        ctx.T = T
        
        return output_spikes, v_mem
    
    @staticmethod
    def backward(ctx, grad_output_spikes, grad_v_mem_final):
        """
        Backward pass with surrogate gradient (SuperSpike).
        """
        (input_current, output_spikes, v_mem_history, 
         beta, v_th, v_reset) = ctx.saved_tensors
        k = ctx.k_superspike
        T = ctx.T
        
        # Get values
        beta_val = beta.item() if hasattr(beta, 'item') else beta
        v_th_val = v_th.item() if hasattr(v_th, 'item') else v_th
        
        grad_input = torch.zeros_like(input_current)
        grad_beta_total = torch.tensor(0.0, device=input_current.device)
        
        # Backward through time
        grad_v_mem = grad_v_mem_final if grad_v_mem_final is not None else torch.zeros_like(v_mem_history[0])
        
        for t in range(T - 1, -1, -1):
            v_mem = v_mem_history[t]
            spike = output_spikes[t]
            
            # Surrogate gradient: d(spike)/d(v_mem) = 1 / (1 + k|v - v_th|)^2
            v_over_th = v_mem - v_th_val
            surrogate = 1.0 / (1.0 + k * torch.abs(v_over_th)) ** 2
            
            # Chain rule through spike
            grad_from_spike = grad_output_spikes[t] * surrogate
            grad_v_mem = grad_v_mem + grad_from_spike
            
            # Gradient w.r.t. input
            grad_input[t] = grad_v_mem
            
            # Gradient w.r.t. beta (leak contribution)
            if t > 0:
                grad_beta_total = grad_beta_total + (grad_v_mem * v_mem_history[t-1]).sum()
            
            # Propagate gradient through leak (multiply by beta for previous timestep)
            grad_v_mem = grad_v_mem * beta_val * (1.0 - spike)  # No gradient through reset
        
        # Return gradients in order: input, beta, v_th, v_reset, k_superspike
        return grad_input, grad_beta_total, None, None, None


class LIFPyTorch(nn.Module):
    """
    Pure PyTorch LIF neuron module.
    
    This is a fallback implementation when Triton is not available.
    It provides the same interface as the Triton-optimized version.
    """
    
    def __init__(
        self,
        beta: float = 0.9,
        v_th: float = 1.0,
        v_reset: float = 0.0,
        k_superspike: float = 4.0,
        n_steps: int = 4,
        learn_beta: bool = False,
        learn_v_th: bool = False,
        learn_v_reset: bool = False
    ):
        super().__init__()
        
        self.n_steps = n_steps
        self.learn_beta = learn_beta
        self.learn_v_th = learn_v_th
        self.learn_v_reset = learn_v_reset
        
        # Beta parameter
        if learn_beta:
            self.beta_raw = nn.Parameter(torch.tensor(self._inverse_sigmoid(beta)))
        else:
            self.register_buffer('beta', torch.tensor(beta))
        
        # v_th parameter
        if learn_v_th:
            self.v_th = nn.Parameter(torch.tensor(v_th))
        else:
            self.register_buffer('v_th', torch.tensor(v_th))
        
        # v_reset parameter
        if learn_v_reset:
            self.v_reset = nn.Parameter(torch.tensor(v_reset))
        else:
            self.register_buffer('v_reset', torch.tensor(v_reset))
        
        self.register_buffer('k_superspike', torch.tensor(k_superspike))
    
    @staticmethod
    def _inverse_sigmoid(x: float) -> float:
        x = max(min(x, 0.9999), 0.0001)
        return -torch.log(torch.tensor(1.0 / x - 1.0)).item()
    
    def get_beta(self) -> float:
        if self.learn_beta:
            return torch.sigmoid(self.beta_raw).item()
        return self.beta.item()
    
    def forward(self, input_current: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_current: Input tensor of any shape, will be reshaped for temporal processing
            
        Returns:
            spikes: Output spikes (same shape as input)
            v_mem_final: Final membrane potential
        """
        original_shape = input_current.shape
        
        # Reshape: [B, ...] -> [T, B_new, Feat_flat]
        try:
            input_current = input_current.reshape(
                self.n_steps, 
                original_shape[0] // self.n_steps, 
                *original_shape[1:]
            )
            input_current = input_current.reshape(
                self.n_steps, 
                original_shape[0] // self.n_steps, 
                -1
            )
        except RuntimeError as e:
            raise RuntimeError(
                f"LIF reshape error. Input shape {original_shape}, n_steps={self.n_steps}. "
                f"Batch dim ({original_shape[0]}) must be divisible by n_steps. Error: {e}"
            )
        
        # Get parameters
        beta = torch.sigmoid(self.beta_raw) if self.learn_beta else self.beta
        v_th = self.v_th
        v_reset = self.v_reset
        
        # Forward pass
        spikes, v_mem_final = LIFLayerPyTorch.apply(
            input_current,
            beta,
            v_th,
            v_reset,
            self.k_superspike.item()
        )
        
        # Reshape back to original
        spikes = spikes.view(original_shape[0], -1).view(original_shape)
        v_mem_final = v_mem_final.view(
            original_shape[0] // self.n_steps, 
            *original_shape[1:]
        )
        
        return spikes, v_mem_final
    
    def extra_repr(self) -> str:
        return (
            f'beta={self.get_beta():.3f} (learnable={self.learn_beta}), '
            f'v_th={self.v_th.item():.3f}, v_reset={self.v_reset.item():.3f}, '
            f'k_superspike={self.k_superspike.item():.1f}, backend=PyTorch'
        )


# ============================================================================
# Unified LIF Module with Auto Backend Selection
# ============================================================================

_TRITON_WARNING_SHOWN = False

def get_lif_class():
    """
    Returns the appropriate LIF class based on Triton availability.
    Shows a warning once if using PyTorch fallback.
    """
    global _TRITON_WARNING_SHOWN
    
    if TRITON_AVAILABLE:
        # Import Triton version
        from Triton_Layers.Lif import LIF as LIFTriton
        return LIFTriton
    else:
        if not _TRITON_WARNING_SHOWN:
            warnings.warn(
                f"\n{'='*60}\n"
                f"⚠️  TRITON NOT AVAILABLE - Using PyTorch fallback for LIF neurons\n"
                f"    This may be slower than the Triton-optimized version.\n"
                f"    Reason: {_triton_import_error}\n"
                f"    \n"
                f"    To enable Triton acceleration:\n"
                f"    - Use Linux (Triton doesn't support Windows natively)\n"
                f"    - Install triton: pip install triton\n"
                f"{'='*60}",
                RuntimeWarning,
                stacklevel=2
            )
            _TRITON_WARNING_SHOWN = True
        return LIFPyTorch

# Create unified LIF class that auto-selects backend
class LIF(nn.Module):
    """
    Unified LIF (Leaky Integrate-and-Fire) neuron module.
    
    Supports two modes:
    - SPIKE mode (frequency_mode=False): Temporal simulation over n_steps
    - FREQUENCY mode (frequency_mode=True): Quantized activation, no temporal expansion
    
    IMPORTANT: When learn_beta=True, the frequency mode dynamically uses the learned
    beta from the spike LIF to ensure equivalence during training.
    
    Args:
        beta: Leak factor (0 < beta <= 1)
        v_th: Threshold voltage
        v_reset: Reset voltage
        k_superspike: Surrogate gradient sharpness
        n_steps: Number of time steps
        learn_beta: If True, beta becomes learnable
        learn_v_th: If True, v_th becomes learnable
        learn_v_reset: If True, v_reset becomes learnable
    """
    
    def __init__(
        self,
        beta: float = 0.9,
        v_th: float = 1.0,
        v_reset: float = 0.0,
        k_superspike: float = 4.0,
        n_steps: int = 4,
        learn_beta: bool = False,
        learn_v_th: bool = False,
        learn_v_reset: bool = False
    ):
        super().__init__()
        
        self.n_steps = n_steps
        self.init_v_th = v_th
        self.learn_beta = learn_beta
        
        # Get appropriate spike-based LIF class
        LIFClass = get_lif_class()
        
        # Create inner temporal LIF module (spike mode)
        # This is the authoritative source for learned parameters
        self._lif_spike = LIFClass(
            beta=beta,
            v_th=v_th,
            v_reset=v_reset,
            k_superspike=k_superspike,
            n_steps=n_steps,
            learn_beta=learn_beta,
            learn_v_th=learn_v_th,
            learn_v_reset=learn_v_reset
        )
        
        self.backend = "triton" if TRITON_AVAILABLE else "pytorch"
        
        # Mode: False = spike (default), True = frequency
        self.frequency_mode = False
    
    def _compute_frequency_output(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute LIF frequency output with ISI-based approximation.
        
        For a discrete LIF neuron with leak (beta) and hard reset, the average firing
        rate for constant input I over T timesteps can be approximated using the
        Inter-Spike Interval (ISI) theory.
        
        **Theory:**
        For constant current I > v_th, a leaky integrate-and-fire neuron with:
        - Leak factor beta (membrane voltage decay)
        - Threshold v_th
        - Hard reset to v_reset (typically 0)
        
        The membrane voltage evolves as:
            v[t] = beta * v[t-1] + I    (when v < v_th)
            v[t] = v_reset              (when v >= v_th, spike and reset)
        
        The time to first spike (ISI) starting from v_reset can be derived:
        For beta < 1 (leaky neuron):
            The steady-state input required to reach v_th is:
            v_th = I * (1 - beta^ISI) / (1 - beta)  when no spike occurs
            
        For constant input with periodic spiking (reset after each spike):
            ISI = log(1 - v_th*(1-beta)/I) / log(beta)  (continuous approximation)
            
        For discrete timesteps, a simplified approximation that works well:
            - If I >> v_th: neuron spikes every timestep (rate ≈ 1)
            - If I ≈ v_th: ISI depends on beta (more leak = longer ISI = lower rate)
            - If I < v_th: neuron rarely/never spikes
            
        **Practical approximation:**
        We use a beta-adjusted effective input:
            I_eff = I / v_th * gain_factor
        where gain_factor accounts for the accumulation efficiency with leak:
            gain_factor = (1 - beta^T) / (1 - beta) / T  for beta ≠ 1
            gain_factor = 1                               for beta = 1 (no leak)
        
        This gives a normalized rate estimate that:
        1. Approaches 0 when I << v_th (sub-threshold)
        2. Approaches 1 when I >> v_th (saturates to spike every step)
        3. Depends on beta (lower beta = more leak = lower rate for same input)
        4. Is quantized to n_steps levels for STE gradient training
        
        **Numerical stability:**
        - Use clamp to keep values in [0, 1] range
        - Add small epsilon (1e-7) to prevent division by zero
        - Compatible with AMP (automatic mixed precision)
        
        Args:
            x: Input tensor (constant current over T steps)
            
        Returns:
            Quantized firing rate in {0, 1/T, 2/T, ..., 1}
        """
        # Get current parameters from spike LIF (may be learned)
        v_th = self._lif_spike.v_th
        if not isinstance(v_th, torch.Tensor):
            v_th = torch.tensor(v_th, device=x.device, dtype=x.dtype)
        
        # Get beta value
        beta_val = self.get_beta()
        beta = torch.tensor(beta_val, device=x.device, dtype=x.dtype)
        
        # Compute gain factor for accumulation efficiency with leak
        # gain = (1 - beta^T) / ((1 - beta) * T)
        # This represents the effective integration efficiency per timestep
        eps = 1e-7
        if abs(beta_val - 1.0) < eps:
            # No leak case: beta ≈ 1, full accumulation
            gain = torch.tensor(1.0, device=x.device, dtype=x.dtype)
        else:
            # With leak: reduced accumulation efficiency
            beta_T = torch.pow(beta, self.n_steps)
            gain = (1.0 - beta_T) / ((1.0 - beta + eps) * self.n_steps)
        
        # Normalize input by threshold and adjust for leak
        # This gives an estimate of firing rate per timestep
        x_normalized = torch.relu(x) / (v_th + eps) * gain
        
        # Clamp to [0, 1]: rate cannot be negative or exceed 1 spike/step
        rate = torch.clamp(x_normalized, 0.0, 1.0)
        
        # Quantize with STE to n_steps levels for training
        from .Lif_Frequency import STEQuantize
        output = STEQuantize.apply(rate, self.n_steps)
        
        return output
    
    def forward(self, input_current: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.frequency_mode:
            # Mode FREQUENCY: calcul dynamique avec beta actuel
            # Input: [B, ...] -> Output: [B, ...]
            output = self._compute_frequency_output(input_current)
            return output, None
        else:
            # Mode SPIKE: simulation temporelle
            # Input attend [T*B, ...] (pré-répété) -> Output: [T*B, ...]
            return self._lif_spike(input_current)
    
    def get_beta(self) -> float:
        return self._lif_spike.get_beta()
    
    @property
    def beta(self):
        """Get current beta value (handles both learned and fixed cases)."""
        # Use get_beta() which handles learn_beta properly
        return self._lif_spike.get_beta()
    
    @property
    def v_th(self):
        """Get current v_th value."""
        return self._lif_spike.v_th.item() if hasattr(self._lif_spike.v_th, 'item') else self._lif_spike.v_th
    
    def extra_repr(self) -> str:
        mode = "frequency" if self.frequency_mode else "spike"
        return f'backend={self.backend}, mode={mode}, learn_beta={self.learn_beta}'


# Export check function
__all__ = ['LIF', 'LIFPyTorch', 'is_triton_available', 'TRITON_AVAILABLE']
