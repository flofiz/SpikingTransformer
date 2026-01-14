import torch
from .lif_auto import LIF  # Auto-selects Triton or PyTorch fallback
from .Lif_Frequency import LIFFrequency
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Literal


class SSAMultiHeadAttention_(nn.Module):
    """
    [Documentation inchangée...]
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        alpha: Optional[float] = None,
        causal: bool = False,
        learnable_alpha: bool = False,
        dropout: float = 0.0,
        n_steps: int = 1,
        bias: bool = True,
        mask_mode: Literal["multiply", "additive"] = "multiply"
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.mask_mode = mask_mode
        
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)

        self.lnq = nn.LayerNorm(d_model)
        self.lnk = nn.LayerNorm(d_model)
        self.lnv = nn.LayerNorm(d_model)
        self.lno = nn.LayerNorm(d_model)

        self.lifq = LIF(n_steps=n_steps)
        self.lifk = LIF(n_steps=n_steps)
        self.lifv = LIF(n_steps=n_steps)
        self.lifs = LIF(n_steps=n_steps, v_th=0.5)
        self.lifo = LIF(n_steps=n_steps)

        
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if key is None:
            key = query
        if value is None:
            value = query
        
        B, N, D = query.shape
        B_k, N_k, D_k = key.shape
        
        Q = self.q_proj(query)
        Q = self.lnq(Q)
        Q, _ = self.lifq(Q)

        K = self.k_proj(key)
        K = self.lnk(K)
        K, _ = self.lifk(K)

        V = self.v_proj(value)
        V = self.lnv(V)
        V, _ = self.lifv(V)

        Q = Q.view(B, N, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, N, Dh)
        K = K.view(B, N_k, self.n_heads, self.d_head).transpose(1, 2) # (B, H, N_k, Dh)
        V = V.view(B, N_k, self.n_heads, self.d_head).transpose(1, 2) # (B, H, N_k, Dh)

        attn_output = (Q @ K.transpose(-2, -1))
        if attention_mask is not None:
            if self.mask_mode == "multiply":
                attn_output = attn_output * attention_mask
            elif self.mask_mode == "additive":
                attn_output = attn_output.masked_fill(attention_mask == 0, float('-inf'))
        attn_output = (attn_output @ V)* 0.125

        attn_output, _ = self.lifs(attn_output)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, D)
        
        output = self.out_proj(attn_output)
        output = self.lno(output)
        output, _ = self.lifo(output)
        
        return output
    
    def extra_repr(self) -> str:
        return (
            f'd_model={self.d_model}, '
            f'n_heads={self.n_heads}, '
            f'd_head={self.d_head}, '
            f'mask_mode={self.mask_mode}'
        )


class SSAMultiHeadAttention(nn.Module):
    """
    Spiking Self-Attention with XNOR attention and Log Positional Encoding.
    Supports configurable mask mode (multiply or additive).
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        alpha: Optional[float] = None,
        causal: bool = False,
        learnable_alpha: bool = False,
        dropout: float = 0.0,
        n_steps: int = 1,
        bias: bool = True,
        mask_mode: Literal["multiply", "additive"] = "multiply"
    ):
        super().__init__()

        # Scale initialized to smaller value as per paper recommendation
        self.scale = nn.Parameter(torch.tensor(0.05), requires_grad=True)
        self.mask_mode = mask_mode
        self.n_steps = n_steps
        
        # Mode: False = spike (default), True = frequency
        self.frequency_mode = False
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)

        self.lnq = nn.LayerNorm(d_model)
        self.lnk = nn.LayerNorm(d_model)
        self.lnv = nn.LayerNorm(d_model)
        self.lno = nn.LayerNorm(d_model)

        # Spike-mode LIF layers
        self.lifq = LIF(n_steps=n_steps)
        self.lifk = LIF(n_steps=n_steps)
        self.lifv = LIF(n_steps=n_steps)
        self.lifs = LIF(n_steps=n_steps, v_th=0.5)
        self.lifo = LIF(n_steps=n_steps)
        
        # Frequency-mode LIF layers
        self.lifq_freq = LIFFrequency(n_steps=n_steps)
        self.lifk_freq = LIFFrequency(n_steps=n_steps)
        self.lifv_freq = LIFFrequency(n_steps=n_steps)
        self.lifs_freq = LIFFrequency(n_steps=n_steps, v_th=0.5)
        self.lifo_freq = LIFFrequency(n_steps=n_steps)

        
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def xnor_attention(self, Q, K):
        """XNOR attention pour spikes binaires (mode spike)"""
        B, H, L, Dh = Q.shape
        
        # Travailler directement en 4D
        # Q: (B, H, L, D), K: (B, H, L, D)
        
        # Produit: einsum est toujours différentiable
        qk_product = torch.einsum('bhld,bhmd->bhlm', Q, K)
        
        # Sommes
        q_sum = Q.sum(dim=-1, keepdim=True)  # (B, H, L, 1)
        k_sum = K.sum(dim=-1, keepdim=True)  # (B, H, L, 1)
        
        # Hamming distance
        hamming_dist = q_sum + k_sum.transpose(2, 3) - 2 * qk_product
        
        # XNOR count
        attn_map = Dh - hamming_dist
        
        return attn_map
    
    def xnor_attention_frequency(self, Q, K):
        """
        Équivalent probabiliste de l'attention XNOR pour l'entraînement fréquentiel.
        
        Formule: P(XNOR=1) = 2*q*k - q - k + 1
        
        Cette formule donne la probabilité que deux bits q et k (représentés
        comme probabilités de spike) soient identiques (XNOR = 1).
        
        Avantages vs XNOR binaire:
        - Gradients informatifs même quand k=0 (contrairement au produit scalaire)
        - Surface d'optimisation lisse
        - Équivalent exact de XNOR aux coins {0,1}
        
        Args:
            Q: Tensor (B, H, L, D) avec valeurs dans [0, 1]
            K: Tensor (B, H, M, D) avec valeurs dans [0, 1]
            
        Returns:
            attn_map: Tensor (B, H, L, M) représentant la similarité XNOR
        """
        B, H, L, Dh = Q.shape
        _, _, M, _ = K.shape
        
        # Produit Q*K: term 2qk
        qk_prod = torch.einsum('bhld,bhmd->bhlm', Q, K)  # (B, H, L, M)
        
        # Sommes pour les termes -q et -k
        q_sum = Q.sum(dim=-1, keepdim=True)  # (B, H, L, 1)
        k_sum = K.sum(dim=-1, keepdim=True)  # (B, H, M, 1)
        
        # Formule XNOR probabiliste: 2qk - q - k + 1 (sommée sur D dimension)
        # = 2 * sum(qi*ki) - sum(qi) - sum(ki) + D
        attn_map = 2 * qk_prod - q_sum - k_sum.transpose(2, 3) + Dh
        
        return attn_map
    
    def get_log_pe_bias(self, seq_len, device):
        """Calcule le biais Log-PE: R[i,j] = ceil(log2((L-1)/(|i-j|+1)))"""
        pos = torch.arange(seq_len, device=device).unsqueeze(1)
        distance = torch.abs(pos - pos.t()).float()
        
        bias = torch.ceil(torch.log2((seq_len - 1) / (distance + 1)))
        bias = torch.clamp(bias, min=0)
        
        return bias

    def get_log_pe_bias_cross(self, seq_len_q, seq_len_k, device):
        """
        Calcule le biais Log-PE pour cross-attention: R[i,j] = ceil(log2((L-1)/(|i-j|+1)))
        
        Args:
            seq_len_q: Longueur de la séquence query
            seq_len_k: Longueur de la séquence key
            device: Device PyTorch
        
        Returns:
            bias: Tensor de shape [seq_len_q, seq_len_k]
        """
        # Positions pour queries et keys
        pos_q = torch.arange(seq_len_q, device=device).unsqueeze(1)  # [seq_len_q, 1]
        pos_k = torch.arange(seq_len_k, device=device).unsqueeze(0)  # [1, seq_len_k]
        
        # Distance absolue entre toutes les paires (q, k)
        distance = torch.abs(pos_q - pos_k).float()  # [seq_len_q, seq_len_k]
        
        # Normalisation: utiliser la longueur maximale
        max_len = max(seq_len_q, seq_len_k)
        bias = torch.ceil(torch.log2((max_len - 1) / (distance + 1)))
        bias = torch.clamp(bias, min=0)
        
        return bias

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if key is None:
            key = query
        if value is None:
            value = query
        
        B, N, D = query.shape
        B_k, N_k, D_k = key.shape
        
        if self.frequency_mode:
            # === Mode fréquentiel pour l'entraînement ===
            Q = self.q_proj(query)
            Q = self.lnq(Q)
            Q, _ = self.lifq_freq(Q)

            K = self.k_proj(key)
            K = self.lnk(K)
            K, _ = self.lifk_freq(K)

            V = self.v_proj(value)
            V = self.lnv(V)
            V, _ = self.lifv_freq(V)

            Q = Q.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
            K = K.view(B, N_k, self.n_heads, self.d_head).transpose(1, 2)
            V = V.view(B, N_k, self.n_heads, self.d_head).transpose(1, 2)

            # XNOR probabiliste
            attn_output = self.xnor_attention_frequency(Q, K)
        else:
            # === Mode spike pour l'inférence (comportement existant) ===
            Q = self.q_proj(query)
            Q = self.lnq(Q)
            Q, _ = self.lifq(Q)

            K = self.k_proj(key)
            K = self.lnk(K)
            K, _ = self.lifk(K)

            V = self.v_proj(value)
            V = self.lnv(V)
            V, _ = self.lifv(V)

            Q = Q.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
            K = K.view(B, N_k, self.n_heads, self.d_head).transpose(1, 2)
            V = V.view(B, N_k, self.n_heads, self.d_head).transpose(1, 2)

            # XNOR binaire
            attn_output = self.xnor_attention(Q, K)
        
        # Log-PE (commun aux deux modes)
        log_bias = self.get_log_pe_bias_cross(N, N_k, Q.device)
        log_bias = log_bias.unsqueeze(0).unsqueeze(0)
        attn_output = attn_output + log_bias
        
        # Masque d'attention
        if attention_mask is not None:
            if self.mask_mode == "multiply":
                attn_output = attn_output * attention_mask
            elif self.mask_mode == "additive":
                attn_output = attn_output.masked_fill(attention_mask == 0, float('-inf'))
        
        # Agrégation avec V
        attn_output = (attn_output * self.scale) @ V

        # LIF de sortie (adapté au mode)
        if self.frequency_mode:
            attn_output, _ = self.lifs_freq(attn_output)
        else:
            attn_output, _ = self.lifs(attn_output)
            
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, D)
        
        output = self.out_proj(attn_output)
        output = self.lno(output)
        
        if self.frequency_mode:
            output, _ = self.lifo_freq(output)
        else:
            output, _ = self.lifo(output)
        
        return output
    
    def extra_repr(self) -> str:
        return (
            f'd_model={self.d_model}, '
            f'n_heads={self.n_heads}, '
            f'd_head={self.d_head}, '
            f'mask_mode={self.mask_mode}'
        )


class MultiScaleXNORAttention(nn.Module):
    """
    Multi-Scale Spiking Self-Attention (MSSA) adapted with XNOR attention and LogPE.
    Inspired by MSViT paper: each head operates at a different scale via pooling/upsampling.
    This enriches the receptive field by capturing both global and local features.
    
    Args:
        d_model: Model dimension
        n_heads: Total number of attention heads
        scales: List of scales to use (e.g., [1, 2, 4] means downsample by 1x, 2x, 4x)
        n_steps: Number of SNN timesteps
        mask_mode: "multiply" or "additive" for causal masking
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        scales: List[int] = [1, 2, 4],
        n_steps: int = 1,
        dropout: float = 0.0,
        alpha: Optional[float] = None,
        mask_mode: Literal["multiply", "additive"] = "multiply"
    ):
        super().__init__()
        self.scales = scales
        self.n_heads = n_heads
        self.d_model = d_model
        self.mask_mode = mask_mode
        
        # Distribute heads across scales
        self.heads_per_scale = n_heads // len(scales)
        assert n_heads % len(scales) == 0, f"n_heads ({n_heads}) must be divisible by number of scales ({len(scales)})"
        
        # One XNOR attention per scale
        self.attention_heads = nn.ModuleList([
            SSAMultiHeadAttention(
                d_model=d_model,
                n_heads=self.heads_per_scale,
                n_steps=n_steps,
                dropout=dropout,
                mask_mode=mask_mode
            )
            for _ in scales
        ])
        
        # Fusion projection
        self.fusion = nn.Linear(d_model * len(scales), d_model)
        self.fusion_ln = nn.LayerNorm(d_model)
        self.fusion_lif = LIF(n_steps=n_steps)
    
    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if key is None:
            key = query
        if value is None:
            value = query
            
        B, N, D = query.shape
        outputs = []
        
        for scale, attn in zip(self.scales, self.attention_heads):
            if scale > 1 and N >= scale:
                # Downsample -> Attention -> Upsample
                # Use avg pooling on sequence dimension
                q_scaled = F.avg_pool1d(query.transpose(1, 2), scale, stride=scale).transpose(1, 2)
                k_scaled = F.avg_pool1d(key.transpose(1, 2), scale, stride=scale).transpose(1, 2)
                v_scaled = F.avg_pool1d(value.transpose(1, 2), scale, stride=scale).transpose(1, 2)
                
                # No mask for scaled attention (global context)
                out = attn(q_scaled, k_scaled, v_scaled, attention_mask=None)
                
                # Upsample back to original sequence length
                out = F.interpolate(out.transpose(1, 2), size=N, mode='linear', align_corners=False).transpose(1, 2)
            else:
                # Scale 1 OR sequence too short: standard attention
                # Note: For greedy decoding (N=1), we can't downsample, so we use full resolution
                out = attn(query, key, value, attention_mask=attention_mask)
            
            outputs.append(out)
        
        # Concatenate outputs from all scales
        fused = torch.cat(outputs, dim=-1)  # (B, N, D * num_scales)
        
        # Fusion projection
        fused = self.fusion(fused)
        fused = self.fusion_ln(fused)
        fused, _ = self.fusion_lif(fused)
        
        return fused
    
    def extra_repr(self) -> str:
        return (
            f'd_model={self.d_model}, '
            f'n_heads={self.n_heads}, '
            f'scales={self.scales}, '
            f'heads_per_scale={self.heads_per_scale}, '
            f'mask_mode={self.mask_mode}'
        )