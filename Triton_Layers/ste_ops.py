"""
Straight-Through Estimator (STE) operations for binarization and quantization.

This module provides differentiable binarization and quantization operations
that use STE gradients for training spiking neural networks in frequency mode.

The STE allows gradients to flow through non-differentiable operations like
binarization and quantization by using a differentiable approximation in the
backward pass.
"""

import torch
from typing import Tuple


class STEBinarize(torch.autograd.Function):
    """
    Binarization with Straight-Through Estimator (STE).
    
    Forward pass: Binary output (0 or 1) based on threshold
    Backward pass: Gradient passes through as if the function were identity
                   in the range [0, 1], zero outside
    
    This is useful for training networks that need to produce binary outputs
    (like spike patterns) but still need gradients for learning.
    
    Example:
        >>> x = torch.tensor([0.2, 0.6, 0.8], requires_grad=True)
        >>> binary_x = STEBinarize.apply(x, 0.5)
        >>> # binary_x = tensor([0., 1., 1.])
        >>> loss = binary_x.sum()
        >>> loss.backward()
        >>> # x.grad will be non-zero for values in [0, 1]
    """
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Binarize input tensor based on threshold.
        
        Args:
            x: Input tensor (typically in range [0, 1] for meaningful gradients)
            threshold: Threshold value for binarization (default: 0.5)
            
        Returns:
            Binary tensor with values in {0.0, 1.0}
            - 0.0 if x < threshold
            - 1.0 if x >= threshold
        """
        # Save input for backward pass
        ctx.save_for_backward(x)
        
        # Binary threshold: x >= threshold -> 1, else 0
        binary_output = (x >= threshold).float()
        
        return binary_output
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """
        STE backward: gradient passes through in the dynamic range [0, 1].
        
        The gradient is passed through unchanged for values in [0, 1],
        and zeroed for values outside this range. This encourages the
        network to keep activations in the learnable region.
        
        Args:
            grad_output: Gradient from the next layer
            
        Returns:
            Tuple of (grad_input, None) where:
            - grad_input: Gradient w.r.t. input x
            - None: No gradient for threshold parameter
        """
        x, = ctx.saved_tensors
        
        # Gradient passes through only in the range [0, 1]
        # This prevents saturation and keeps learning active
        grad_mask = (x >= 0.0) & (x <= 1.0)
        grad_input = grad_output * grad_mask.float()
        
        return grad_input, None  # No gradient for threshold


class STEBinarizeClip(torch.autograd.Function):
    """
    Alternative binarization with clipped gradient (clip-based STE).
    
    Forward: Binary threshold
    Backward: Gradient uses clipped sigmoid approximation
    
    This variant uses a sigmoid-like approximation in the backward pass,
    which can provide smoother gradients near the threshold.
    """
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, threshold: float = 0.5, temperature: float = 1.0) -> torch.Tensor:
        """
        Binarize with sigmoid gradient approximation.
        
        Args:
            x: Input tensor
            threshold: Binarization threshold
            temperature: Controls gradient steepness (lower = sharper)
        """
        ctx.save_for_backward(x)
        ctx.threshold = threshold
        ctx.temperature = temperature
        
        binary_output = (x >= threshold).float()
        return binary_output
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        """
        Backward with sigmoid approximation of the step function.
        """
        x, = ctx.saved_tensors
        threshold = ctx.threshold
        temperature = ctx.temperature
        
        # Sigmoid approximation: d(step(x))/dx ≈ sigmoid'(x)
        # sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        z = (x - threshold) / temperature
        sigmoid_z = torch.sigmoid(z)
        grad_approx = sigmoid_z * (1 - sigmoid_z) / temperature
        
        # Clip to [0, 1] range to prevent extreme gradients
        grad_approx = torch.clamp(grad_approx, 0.0, 1.0)
        
        grad_input = grad_output * grad_approx
        
        return grad_input, None, None


def binarize_ste(x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Convenience function for binarization with STE.
    
    Args:
        x: Input tensor (should be in [0, 1] for best gradient behavior)
        threshold: Binarization threshold (default: 0.5)
        
    Returns:
        Binary tensor with values {0.0, 1.0}
        
    Example:
        >>> Q = torch.randn(8, 64, requires_grad=True)
        >>> Q_norm = torch.sigmoid(Q)  # Normalize to [0, 1]
        >>> Q_binary = binarize_ste(Q_norm, threshold=0.5)
    """
    return STEBinarize.apply(x, threshold)


def binarize_ste_sigmoid(x: torch.Tensor, threshold: float = 0.5, temperature: float = 1.0) -> torch.Tensor:
    """
    Convenience function for binarization with sigmoid-based STE.
    
    Args:
        x: Input tensor
        threshold: Binarization threshold (default: 0.5)
        temperature: Gradient temperature (default: 1.0, lower = sharper)
        
    Returns:
        Binary tensor with values {0.0, 1.0}
    """
    return STEBinarizeClip.apply(x, threshold, temperature)


# Export main components
__all__ = [
    'STEBinarize',
    'STEBinarizeClip', 
    'binarize_ste',
    'binarize_ste_sigmoid'
]
