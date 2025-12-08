import torch
import triton
import triton.language as tl
from .Lif import LIF
import torch.nn as nn
import math
from typing import Optional


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
        bias: bool = True
    ):
        super().__init__()
        
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
            attn_output = attn_output * attention_mask
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
            f'd_head={self.d_head}'
        )


class SSAMultiHeadAttention(nn.Module):
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
        bias: bool = True
    ):
        super().__init__()

        self.scale = nn.Parameter(torch.sqrt(torch.tensor(1.0 / (d_model // n_heads))), requires_grad=True)
        
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

        self.lifq = LIF(n_steps=n_steps)
        self.lifk = LIF(n_steps=n_steps)
        self.lifv = LIF(n_steps=n_steps)
        self.lifs = LIF(n_steps=n_steps, v_th=0.5)
        self.lifo = LIF(n_steps=n_steps)

        
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def xnor_attention(self, Q, K):
        """Version sans reshape pour éviter les problèmes de vue"""
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

        attn_output = self.xnor_attention(Q, K)
        log_bias = self.get_log_pe_bias_cross(N, N_k, Q.device)  # (L, L)
        
        # Broadcast sur batch et heads: (L, L) -> (1, 1, L, L)
        log_bias = log_bias.unsqueeze(0).unsqueeze(0)
        attn_output = attn_output + log_bias  # (B, n_heads, L, L)
        if attention_mask is not None:
            attn_output = attn_output * attention_mask
        attn_output = (attn_output *self.scale)@ V

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
            f'd_head={self.d_head}'
        )