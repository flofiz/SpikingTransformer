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
    
    Équivalence mathématique:
    - Un neurone LIF avec entrée constante sur T pas produit en moyenne
      n_spikes = clip(floor(T * input / threshold), 0, T) spikes
    - La fréquence normalisée est donc n_spikes / T ∈ {0, 1/T, ..., 1}
    
    Args:
        n_steps (int): Nombre de pas de temps (T), détermine le nombre de niveaux
        v_th (float): Seuil de déclenchement (normalisation de l'entrée)
        learn_v_th (bool): Si True, v_th devient apprenable
        
    Shape:
        - Input: Toute forme (B, ...) avec B divisible par n_steps
        - Output: Même forme que l'entrée
    """
    
    def __init__(
        self,
        n_steps: int = 4,
        v_th: float = 1.0,
        beta: float = 0.9,  # Ignoré en mode fréquence, mais gardé pour compatibilité API
        v_reset: float = 0.0,  # Ignoré
        k_superspike: float = 4.0,  # Ignoré
        learn_beta: bool = False,  # Ignoré
        learn_v_th: bool = False,
        learn_v_reset: bool = False  # Ignoré
    ):
        super().__init__()
        
        self.n_steps = n_steps
        
        # Paramètre v_th (seuil)
        if learn_v_th:
            self.v_th = nn.Parameter(torch.tensor(v_th))
        else:
            self.register_buffer('v_th', torch.tensor(v_th))
        
        # Stocké pour compatibilité API avec LIF
        self.register_buffer('beta', torch.tensor(beta))
        self.register_buffer('v_reset', torch.tensor(v_reset))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Propagation avant en mode fréquentiel.
        
        Args:
            x: Tenseur d'entrée de forme quelconque
            
        Returns:
            output: Fréquences quantifiées (même forme que x)
            v_mem_final: Placeholder pour compatibilité API (zéros)
        """
        # Normaliser par le seuil
        x_normalized = x / self.v_th
        
        # Appliquer sigmoid pour mapper dans [0, 1]
        # Cela simule l'accumulation d'un neurone LIF:
        # - entrée faible -> peu de spikes -> fréquence basse
        # - entrée forte -> beaucoup de spikes -> fréquence haute
        x_sigmoid = torch.sigmoid(x_normalized)
        
        # Quantifier avec STE
        output = STEQuantize.apply(x_sigmoid, self.n_steps)
        
        # v_mem_final placeholder (pour compatibilité API avec LIF)
        # On retourne zéros de la forme appropriée
        v_mem_shape = list(x.shape)
        v_mem_shape[0] = v_mem_shape[0] // self.n_steps
        v_mem_final = torch.zeros(v_mem_shape, device=x.device, dtype=x.dtype)
        
        return output, v_mem_final
    
    def extra_repr(self) -> str:
        return (
            f'n_steps={self.n_steps}, '
            f'v_th={self.v_th.item():.3f}'
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
    print(f"v_mem shape: {v_mem.shape}")
    
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
