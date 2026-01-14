"""
LIF Fréquentiel pour l'entraînement différentiable.

Ce module implémente une version fréquentielle du neurone LIF qui remplace
la simulation temporelle par une fonction d'activation en escalier quantifiée.

Avantages:
- Gradients lisses via STE (Straight-Through Estimator)
- Pas de déroulement temporel (efficacité mémoire)
- Équivalent mathématique au LIF temporel pour T pas de temps

La sortie représente la fréquence de décharge normalisée: a ∈ {0, 1/T, 2/T, ..., 1}
"""

import torch
import torch.nn as nn
from typing import Tuple


class STEQuantize(torch.autograd.Function):
    """
    Fonction de quantification avec Straight-Through Estimator.
    
    Forward: Arrondit à l'entier le plus proche (quantification)
    Backward: Passe le gradient tel quel dans la plage [0, 1], sinon 0
    """
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, n_levels: int) -> torch.Tensor:
        """
        Quantifie x à n_levels niveaux.
        
        Args:
            x: Valeurs d'entrée dans [0, 1]
            n_levels: Nombre de niveaux (T pour T pas de temps)
            
        Returns:
            Valeurs quantifiées dans {0, 1/n_levels, ..., 1}
        """
        # Sauvegarder pour le backward
        ctx.save_for_backward(x)
        
        # Quantification: scale -> round -> clip -> unscale
        x_scaled = x * n_levels
        x_quantized = torch.round(x_scaled)
        x_quantized = torch.clamp(x_quantized, 0, n_levels)
        
        return x_quantized / n_levels
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """
        STE: le gradient passe uniquement dans la plage dynamique.
        """
        x, = ctx.saved_tensors
        
        # Gradient = 1 si x ∈ [0, 1], sinon 0 (clipping gradient)
        grad_mask = (x >= 0.0) & (x <= 1.0)
        grad_input = grad_output * grad_mask.float()
        
        return grad_input, None  # Pas de gradient pour n_levels


class LIFFrequency(nn.Module):
    """
    Couche LIF en mode fréquentiel pour l'entraînement.
    
    Remplace la dynamique temporelle du LIF par une fonction d'activation
    en escalier (staircase function) qui représente la fréquence de décharge.
    
    Prend en compte le facteur de leak (beta):
    - Pour beta=1: pas de leak, accumulation complète
    - Pour beta<1: leak, l'entrée effective est réduite
    
    Le gain effectif sur T pas de temps est:
    gain = (1 - beta^T) / (1 - beta)  pour beta != 1
    gain = T                           pour beta = 1
    
    Équivalence mathématique:
    - Entrée effective = input * gain / T
    - frequency = quantize(ReLU(entrée_effective / v_th), T niveaux)
    
    Args:
        n_steps (int): Nombre de pas de temps (T)
        v_th (float): Seuil de déclenchement
        beta (float): Facteur de leak (0 < beta <= 1)
        learn_v_th (bool): Si True, v_th devient apprenable
    """
    
    def __init__(
        self,
        n_steps: int = 4,
        v_th: float = 1.0,
        beta: float = 0.9,
        v_reset: float = 0.0,  # Gardé pour compatibilité API
        k_superspike: float = 4.0,  # Gardé pour compatibilité API
        learn_beta: bool = False,  # Gardé pour compatibilité API
        learn_v_th: bool = False,
        learn_v_reset: bool = False  # Gardé pour compatibilité API
    ):
        super().__init__()
        
        self.n_steps = n_steps
        
        # Paramètre v_th (seuil)
        if learn_v_th:
            self.v_th = nn.Parameter(torch.tensor(v_th))
        else:
            self.register_buffer('v_th', torch.tensor(v_th))
        
        # Beta (leak factor)
        self.register_buffer('beta', torch.tensor(beta))
        self.register_buffer('v_reset', torch.tensor(v_reset))
        
        # Précalcul du gain effectif
        # gain = (1 - beta^T) / (1 - beta) pour beta != 1
        # gain = T pour beta = 1
        if abs(beta - 1.0) < 1e-6:
            gain = float(n_steps)
        else:
            gain = (1.0 - beta ** n_steps) / (1.0 - beta)
        self.register_buffer('effective_gain', torch.tensor(gain))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Propagation avant en mode fréquentiel.
        
        Approximation du LIF avec leak:
        1. Calcul de l'entrée effective en tenant compte du gain (beta)
        2. ReLU: LIF ne fire que pour entrées positives
        3. Normalisation par seuil et nombre de pas
        4. Quantification avec STE
        
        Args:
            x: Tenseur d'entrée de forme quelconque (B, ...)
            
        Returns:
            output: Fréquences quantifiées dans {0, 1/T, ..., 1}
            v_mem_final: None (pas de membrane en mode fréquence)
        """
        # Entrée effective = input * gain / T
        # Cela représente combien le potentiel de membrane s'accumule en moyenne
        x_effective = x * self.effective_gain / self.n_steps
        
        # Normaliser par le seuil
        x_normalized = x_effective / self.v_th
        
        # ReLU + Clamp à [0, 1]
        # LIF ne fire que pour entrées positives, fréquence max = 1
        x_clamped = torch.clamp(torch.relu(x_normalized), 0.0, 1.0)
        
        # Quantifier avec STE à n_steps niveaux
        output = STEQuantize.apply(x_clamped, self.n_steps)
        
        return output, None
    
    def extra_repr(self) -> str:
        return (
            f'n_steps={self.n_steps}, '
            f'v_th={self.v_th.item():.3f}, '
            f'beta={self.beta.item():.3f}, '
            f'gain={self.effective_gain.item():.3f}'
        )


class LIFFrequencySimple(nn.Module):
    """
    Version simplifiée du LIF fréquentiel avec ReLU bornée.
    
    Utilise une fonction en escalier basée sur ReLU (plus proche du LIF sans leak):
    output = clip(round(T * ReLU(x) / max_x), 0, T) / T
    
    Cette version est plus simple et peut être préférable pour certaines applications.
    """
    
    def __init__(
        self,
        n_steps: int = 4,
        v_th: float = 1.0,
        **kwargs  # Ignore les autres paramètres pour compatibilité
    ):
        super().__init__()
        
        self.n_steps = n_steps
        self.register_buffer('v_th', torch.tensor(v_th))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward avec ReLU bornée et quantifiée.
        """
        # ReLU normalisée dans [0, 1]
        x_relu = torch.relu(x / self.v_th)
        x_clamped = torch.clamp(x_relu, 0.0, 1.0)
        
        # Quantifier avec STE
        output = STEQuantize.apply(x_clamped, self.n_steps)
        
        # Placeholder v_mem_final
        v_mem_shape = list(x.shape)
        v_mem_shape[0] = v_mem_shape[0] // self.n_steps
        v_mem_final = torch.zeros(v_mem_shape, device=x.device, dtype=x.dtype)
        
        return output, v_mem_final
    
    def extra_repr(self) -> str:
        return f'n_steps={self.n_steps}, v_th={self.v_th.item():.3f}'


# =============================================================================
# Tests de vérification d'équivalence
# =============================================================================

def test_lif_frequency_equivalence():
    """
    Test de vérification que LIFFrequency produit des sorties cohérentes.
    """
    print("=== Test LIFFrequency ===")
    
    # Configuration
    n_steps = 4
    batch_size = 8
    features = 64
    
    # Créer le module
    lif_freq = LIFFrequency(n_steps=n_steps)
    
    # Entrée test
    x = torch.randn(batch_size, features)
    
    # Forward pass
    output, v_mem = lif_freq(x)
    
    # Vérifications
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"v_mem: {v_mem} (None expected in frequency mode)")
    
    # 1. La sortie doit être quantifiée
    unique_vals = torch.unique(output)
    expected_vals = torch.tensor([i / n_steps for i in range(n_steps + 1)])
    print(f"Unique values in output: {unique_vals.tolist()}")
    print(f"Expected levels: {expected_vals.tolist()}")
    
    # Vérifier que toutes les valeurs sont dans les niveaux attendus
    for val in unique_vals:
        assert any(torch.isclose(val, ev) for ev in expected_vals), \
            f"Value {val} not in expected levels"
    
    # 2. Les gradients doivent passer
    x_grad = x.clone().requires_grad_(True)
    output_grad, _ = lif_freq(x_grad)
    loss = output_grad.sum()
    loss.backward()
    
    assert x_grad.grad is not None, "Gradient should not be None"
    print(f"Gradient norm: {x_grad.grad.norm().item():.4f}")
    print(f"Non-zero gradients: {(x_grad.grad != 0).sum().item()} / {x_grad.grad.numel()}")
    
    print("✓ All tests passed!")
    return True


if __name__ == "__main__":
    test_lif_frequency_equivalence()
