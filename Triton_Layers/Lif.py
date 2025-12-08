import torch
import triton
import triton.language as tl
import torch.nn as nn
from typing import Tuple


class LIFLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, 
                input_current,
                beta,  # Tenseur (scalaire)
                v_th,  # Tenseur (scalaire)
                v_reset, # Tenseur (scalaire)
                k_superspike):
        
        T, N_BATCH, N_NEURONS = input_current.shape
        input_current = input_current.contiguous()
        
        output_spikes = torch.empty_like(input_current, dtype=torch.float32)
        v_mem_init = torch.zeros_like(input_current[0], dtype=torch.float32)   
        v_mem_final = torch.empty_like(v_mem_init, dtype=torch.float32)
        
        BLOCK_SIZE_N_MIN = 128
        n_blocks = (N_NEURONS + BLOCK_SIZE_N_MIN - 1) // BLOCK_SIZE_N_MIN
        grid = (N_BATCH, n_blocks)
        
        lif_forward_kernel[grid](
            input_current, output_spikes, v_mem_final,
            v_mem_init, 
            beta, v_th, v_reset,  # <-- MODIFIÉ: On passe les tenseurs
            T, N_BATCH, N_NEURONS,
            input_current.stride(0), input_current.stride(1), input_current.stride(2),
            output_spikes.stride(0), output_spikes.stride(1), output_spikes.stride(2),
            v_mem_init.stride(0), v_mem_init.stride(1),
            BLOCK_SIZE_T=T
        )
        
        # <-- MODIFIÉ: Sauvegarder tous les tenseurs apprenables
        ctx.save_for_backward(input_current, output_spikes, beta, v_th, v_reset)
        ctx.k_superspike = k_superspike
        ctx.T = T
        
        return output_spikes, v_mem_final

    @staticmethod
    def backward(ctx, grad_output_spikes, grad_v_mem_final):
        # <-- MODIFIÉ: Récupérer tous les tenseurs
        input_current, output_spikes, beta, v_th, v_reset = ctx.saved_tensors
        T = ctx.T
        k_superspike = ctx.k_superspike
        N_BATCH, N_NEURONS = input_current.shape[1], input_current.shape[2]

        grad_output_spikes = grad_output_spikes.contiguous()
        grad_input = torch.empty_like(input_current)
        grad_beta_per_neuron = torch.empty((N_BATCH, N_NEURONS),
                                           dtype=torch.float32,
                                           device=input_current.device)
        
        v_mem_init = torch.zeros((N_BATCH, N_NEURONS), 
                                 dtype=torch.float32, 
                                 device=input_current.device)
        
        v_mem_history = torch.zeros((T, N_BATCH, N_NEURONS), 
                                    dtype=torch.float32, 
                                    device=input_current.device)
        
        if grad_v_mem_final is None:
            grad_v_mem_final = torch.zeros((N_BATCH, N_NEURONS), 
                                           device=input_current.device,
                                           dtype=torch.float32)
        else:
            grad_v_mem_final = grad_v_mem_final.contiguous().to(input_current.device)
        
        BLOCK_SIZE_N_MIN = 128
        n_blocks = (N_NEURONS + BLOCK_SIZE_N_MIN - 1) // BLOCK_SIZE_N_MIN
        grid = (N_BATCH, n_blocks)
        
        lif_backward_kernel[grid](
            grad_output_spikes, grad_input, grad_v_mem_final,
            grad_beta_per_neuron,
            input_current, output_spikes, v_mem_init,
            v_mem_history,
            beta, v_th, v_reset, # <-- MODIFIÉ: On passe les tenseurs
            k_superspike, T, N_BATCH, N_NEURONS,
            grad_output_spikes.stride(0), grad_output_spikes.stride(1), grad_output_spikes.stride(2),
            grad_input.stride(0), grad_input.stride(1), grad_input.stride(2),
            grad_v_mem_final.stride(0), grad_v_mem_final.stride(1),
            grad_beta_per_neuron.stride(0), grad_beta_per_neuron.stride(1),
            input_current.stride(0), input_current.stride(1), input_current.stride(2),
            output_spikes.stride(0), output_spikes.stride(1), output_spikes.stride(2),
            v_mem_init.stride(0), v_mem_init.stride(1),
            v_mem_history.stride(0), v_mem_history.stride(1), v_mem_history.stride(2),
            BLOCK_SIZE_T=T
        )
        
        grad_beta = torch.sum(grad_beta_per_neuron)
        
        # Renvoie les gradients dans l'ordre: input_current, beta, v_th, v_reset, k_superspike
        # NOTE: Si v_th/v_reset ne sont pas apprenables, leur grad (None) sera ignoré par PyTorch
        grad_v_th = None # Non implémenté
        grad_v_reset = None # Non implémenté
        
        return grad_input, grad_beta, grad_v_th, grad_v_reset, None


# ============================================================================
# Kernel Forward (MODIFIÉ)
# ============================================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_N': 128}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SIZE_N': 256}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SIZE_N': 512}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE_N': 1024}, num_warps=16, num_stages=3),
    ],
    key=['N_NEURONS']
)
@triton.jit
def lif_forward_kernel(
    INPUT_PTR, OUTPUT_SPIKES_PTR, V_MEM_FINAL_PTR,
    V_MEM_INIT_PTR, 
    BETA_PTR,         # <-- MODIFIÉ: Pointeur
    V_TH_PTR,         # <-- MODIFIÉ: Pointeur
    V_RESET_PTR,      # <-- MODIFIÉ: Pointeur
    T: tl.constexpr, 
    N_BATCH: tl.constexpr, 
    N_NEURONS: tl.constexpr,
    stride_in_t, stride_in_b, stride_in_n,
    stride_out_t, stride_out_b, stride_out_n,
    stride_v_b, stride_v_n,
    BLOCK_SIZE_T: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    pid_batch = tl.program_id(axis=0)
    pid_block = tl.program_id(axis=1)
    
    neuron_start = pid_block * BLOCK_SIZE_N
    neuron_offsets = neuron_start + tl.arange(0, BLOCK_SIZE_N)
    neuron_mask = neuron_offsets < N_NEURONS
    
    in_base = INPUT_PTR + pid_batch * stride_in_b
    out_base = OUTPUT_SPIKES_PTR + pid_batch * stride_out_b
    v_init_base = V_MEM_INIT_PTR + pid_batch * stride_v_b
    v_final_base = V_MEM_FINAL_PTR + pid_batch * stride_v_b
    
    # <-- MODIFIÉ: Charger les scalaires depuis les pointeurs
    beta = tl.load(BETA_PTR)
    v_th = tl.load(V_TH_PTR)
    v_reset = tl.load(V_RESET_PTR)
    
    v_mem = tl.load(v_init_base + neuron_offsets * stride_v_n, mask=neuron_mask, other=0.0)
    
    for t in range(0, BLOCK_SIZE_T):
        current_in = tl.load(in_base + t * stride_in_t + neuron_offsets * stride_in_n, 
                             mask=neuron_mask, other=0.0)
        
        # <-- MODIFIÉ: Utiliser les variables lues
        v_mem = v_mem * beta + current_in
        spike = tl.where(v_mem > v_th, 1.0, 0.0)
        
        tl.store(out_base + t * stride_out_t + neuron_offsets * stride_out_n, 
                 spike, mask=neuron_mask)
        
        # <-- MODIFIÉ: Utiliser les variables lues
        v_mem = tl.where(spike > 0.0, v_reset, v_mem)
    
    tl.store(v_final_base + neuron_offsets * stride_v_n, v_mem, mask=neuron_mask)


# ============================================================================
# Kernel Backward (MODIFIÉ)
# ============================================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_N': 128}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SIZE_N': 256}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SIZE_N': 512}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE_N': 1024}, num_warps=16, num_stages=3),
    ],
    key=['N_NEURONS']
)
@triton.jit
def lif_backward_kernel(
    GRAD_OUT_PTR, GRAD_IN_PTR, GRAD_V_FINAL_PTR,
    GRAD_BETA_NEURON_PTR,
    INPUT_PTR, OUTPUT_SPIKES_PTR, V_MEM_INIT_PTR,
    V_MEM_HISTORY_PTR,
    BETA_PTR,         # <-- MODIFIÉ: Pointeur
    V_TH_PTR,         # <-- MODIFIÉ: Pointeur
    V_RESET_PTR,      # <-- MODIFIÉ: Pointeur
    K_SUPERSPIKE: tl.constexpr, # <-- K_SUPERSPIKE reste constexpr (il est fixe)
    T: tl.constexpr, N_BATCH: tl.constexpr, N_NEURONS: tl.constexpr,
    stride_grad_out_t, stride_grad_out_b, stride_grad_out_n,
    stride_grad_in_t, stride_grad_in_b, stride_grad_in_n,
    stride_grad_v_b, stride_grad_v_n,
    stride_grad_beta_b, stride_grad_beta_n,
    stride_in_t, stride_in_b, stride_in_n,
    stride_out_t, stride_out_b, stride_out_n,
    stride_v_init_b, stride_v_init_n,
    stride_v_hist_t, stride_v_hist_b, stride_v_hist_n,
    BLOCK_SIZE_T: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    pid_batch = tl.program_id(axis=0)
    pid_block = tl.program_id(axis=1)
    
    neuron_start = pid_block * BLOCK_SIZE_N
    neuron_offsets = neuron_start + tl.arange(0, BLOCK_SIZE_N)
    neuron_mask = neuron_offsets < N_NEURONS
    
    input_base = INPUT_PTR + pid_batch * stride_in_b
    spike_base = OUTPUT_SPIKES_PTR + pid_batch * stride_out_b
    grad_in_base = GRAD_IN_PTR + pid_batch * stride_grad_in_b
    grad_out_base = GRAD_OUT_PTR + pid_batch * stride_grad_out_b
    v_hist_base = V_MEM_HISTORY_PTR + pid_batch * stride_v_hist_b
    v_init_base = V_MEM_INIT_PTR + pid_batch * stride_v_init_b
    grad_beta_base = GRAD_BETA_NEURON_PTR + pid_batch * stride_grad_beta_b
    
    # <-- MODIFIÉ: Charger les scalaires depuis les pointeurs
    beta = tl.load(BETA_PTR)
    v_th = tl.load(V_TH_PTR)
    v_reset = tl.load(V_RESET_PTR)
    
    # PASSE 1: RE-CALCUL
    v_mem = tl.load(v_init_base + neuron_offsets * stride_v_init_n, 
                    mask=neuron_mask, other=0.0)
    
    for t in range(0, BLOCK_SIZE_T):
        current_in = tl.load(input_base + t * stride_in_t + neuron_offsets * stride_in_n, 
                             mask=neuron_mask, other=0.0)
        v_mem = v_mem * beta + current_in # <-- MODIFIÉ
        tl.store(v_hist_base + t * stride_v_hist_t + neuron_offsets * stride_v_hist_n, 
                 v_mem, mask=neuron_mask)
        spike_t = tl.load(spike_base + t * stride_out_t + neuron_offsets * stride_out_n, 
                          mask=neuron_mask, other=0.0)
        v_mem = tl.where(spike_t > 0.0, v_reset, v_mem) # <-- MODIFIÉ
    
    # PASSE 2: GRADIENT
    grad_state = tl.load(GRAD_V_FINAL_PTR + pid_batch * stride_grad_v_b + neuron_offsets * stride_grad_v_n, 
                         mask=neuron_mask, other=0.0)
    
    grad_beta_accumulator = tl.zeros(neuron_offsets.shape, dtype=tl.float32)

    for t in range(BLOCK_SIZE_T - 1, -1, -1):
        v_mem_t = tl.load(v_hist_base + t * stride_v_hist_t + neuron_offsets * stride_v_hist_n, 
                          mask=neuron_mask, other=0.0)
        spike_t = tl.load(spike_base + t * stride_out_t + neuron_offsets * stride_out_n, 
                          mask=neuron_mask, other=0.0)
        grad_spike = tl.load(grad_out_base + t * stride_grad_out_t + neuron_offsets * stride_grad_out_n, 
                             mask=neuron_mask, other=0.0)
        
        v_over_th = v_mem_t - v_th # <-- MODIFIÉ
        grad_surrogate = superspike_surrogate_grad(v_over_th, K_SUPERSPIKE)
        
        grad_from_state = tl.where(spike_t > 0.0, 0.0, grad_state)
        grad_v = (grad_spike * grad_surrogate) + grad_from_state
        
        tl.store(grad_in_base + t * stride_grad_in_t + neuron_offsets * stride_grad_in_n, 
                 grad_v, mask=neuron_mask)
        
        # --- Calcul du gradient de Beta ---
        v_prev_post_spike = tl.zeros(neuron_offsets.shape, dtype=tl.float32)
        if t > 0:
            v_prev_pre_spike = tl.load(v_hist_base + (t - 1) * stride_v_hist_t + neuron_offsets * stride_v_hist_n, 
                                       mask=neuron_mask, other=0.0)
            spike_prev = tl.load(spike_base + (t - 1) * stride_out_t + neuron_offsets * stride_out_n, 
                                 mask=neuron_mask, other=0.0)
            v_prev_post_spike = tl.where(spike_prev > 0.0, v_reset, v_prev_pre_spike) # <-- MODIFIÉ
        else:
            v_prev_post_spike = tl.load(v_init_base + neuron_offsets * stride_v_init_n, 
                                        mask=neuron_mask, other=0.0)
            
        grad_beta_accumulator += grad_v * v_prev_post_spike
        # --- Fin calcul grad_beta ---

        grad_state = grad_v * beta # <-- MODIFIÉ

    tl.store(grad_beta_base + neuron_offsets * stride_grad_beta_n, 
             grad_beta_accumulator, mask=neuron_mask)


@triton.jit
def superspike_surrogate_grad(v_over_th, K: tl.constexpr):
    abs_v_over_th = tl.abs(v_over_th)
    return 1.0 / (1.0 + K * abs_v_over_th) / (1.0 + K * abs_v_over_th)

# =============================================================================
# Module PyTorch wrapper (MODIFIÉ)
# =============================================================================

class LIF(nn.Module):
    """
    Couche Leaky Integrate-and-Fire (LIF) optimisée avec Triton.
    
    Permet maintenant l'apprentissage granulaire de beta, v_th, v_reset.
    
    Args:
        beta (float): Facteur de fuite (0 < beta <= 1)
        v_th (float): Seuil de déclenchement
        v_reset (float): Potentiel de reset
        k_superspike (float): Pente du surrogate gradient
        n_steps (int): Nombre de pas de temps
        learn_beta (bool): Si True, beta devient apprenable
        learn_v_th (bool): Si True, v_th devient apprenable
        learn_v_reset (bool): Si True, v_reset devient apprenable
        
    Shape:
        - Input: (T, N_BATCH, N_INPUT)
        - Output: (spikes, v_mem_final)
            * spikes: (T, N_BATCH, N_NEURONS)
            * v_mem_final: (N_BATCH, N_NEURONS)
    """
    
    def __init__(
        self,
        beta: float = 0.9,
        v_th: float = 1.0,
        v_reset: float = 0.0,
        k_superspike: float = 4.0,
        n_steps: int = 4,
        learn_beta: bool = False,   # Remplacement de learnable_params
        learn_v_th: bool = False,   # Nouveau
        learn_v_reset: bool = False # Nouveau
    ):
        super().__init__()

        self.n_steps = n_steps
        self.learn_beta = learn_beta
        self.learn_v_th = learn_v_th
        self.learn_v_reset = learn_v_reset
        
        # --- Paramètre Beta ---
        if learn_beta:
            # Paramètre apprenable (stocké en log-space pour stabilité)
            self.beta_raw = nn.Parameter(torch.tensor(self._inverse_sigmoid(beta)))
        else:
            # Paramètre fixe
            self.register_buffer('beta', torch.tensor(beta))
        
        # --- Paramètre v_th ---
        if learn_v_th:
            self.v_th = nn.Parameter(torch.tensor(v_th))
        else:
            self.register_buffer('v_th', torch.tensor(v_th))
            
        # --- Paramètre v_reset ---
        if learn_v_reset:
            self.v_reset = nn.Parameter(torch.tensor(v_reset))
        else:
            self.register_buffer('v_reset', torch.tensor(v_reset))
        
        # k_superspike reste toujours fixe (hyperparam du gradient)
        self.register_buffer('k_superspike', torch.tensor(k_superspike))

    
    @staticmethod
    def _inverse_sigmoid(x: float) -> float:
        """Inverse de la sigmoïde pour initialiser beta_raw"""
        x = max(min(x, 0.9999), 0.0001)  # Clamp
        return -torch.log(torch.tensor(1.0 / x - 1.0)).item()
    
    def get_beta(self) -> float:
        """Retourne la valeur actuelle de beta (entre 0 et 1)"""
        if self.learn_beta: # Modifié
            return torch.sigmoid(self.beta_raw).item()
        else:
            return self.beta.item()
    
    # =============================================================================
    # Module PyTorch wrapper (MODIFIÉ)
    # =============================================================================
    
    # ... (gardez __init__, _inverse_sigmoid, get_beta) ...

    def forward(
        self, 
        input_current: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Propagation avant de la couche LIF.
        """
        
        # --- DÉBUT DE LA LOGIQUE CORRIGÉE ---
        
        # 1. Sauvegarder la forme d'origine (ex: [B, C, H, W])
        original_shape = input_current.shape
        
        # 2. Utiliser VOTRE logique de reshape originale.
        # Elle transforme l'entrée (ex: [B, ...]) en [T, B_new, Feat_flat]
        # où T = n_steps et B_new = B // n_steps
        try:
            input_current = input_current.reshape(self.n_steps, original_shape[0] // self.n_steps, *original_shape[1:])
            input_current = input_current.reshape(self.n_steps, original_shape[0] // self.n_steps, -1)
        except RuntimeError as e:
            # Fournir un message d'erreur plus clair si le reshape échoue
            raise RuntimeError(
                f"Erreur dans LIF.forward reshape. La forme d'entrée était {original_shape} "
                f"et n_steps={self.n_steps}. "
                f"La dimension 0 ({original_shape[0]}) n'est peut-être pas divisible par {self.n_steps}. "
                f"Erreur PyTorch: {e}"
            )
        
        # 3. Récupérer les paramètres (en tant que tenseurs)
        beta = torch.sigmoid(self.beta_raw) if self.learn_beta else self.beta
        v_th = self.v_th if self.learn_v_th else self.v_th
        v_reset = self.v_reset if self.learn_v_reset else self.v_reset
        
        # 4. Appel de la fonction autograd (en passant les TENSEURS)
        # input_current est maintenant [T, B_new, Feat_flat], ce qui est correct (3D)
        spikes, v_mem_final = LIFLayer.apply(
            input_current,
            beta, 
            v_th, 
            v_reset,
            self.k_superspike.item() # k_superspike n'est jamais apprenable
        )
        
        # 5. Remettre les spikes à la forme d'origine (ex: [B, C, H, W])
        spikes = spikes.view(original_shape[0],-1)
        spikes = spikes.view(original_shape)
        
        # 6. Remettre v_mem_final à la forme [B_new, ...Feat...]
        # v_mem_final est [B_new, Feat_flat]
        # original_shape[1:] est [C, H, W]
        # On reshape donc en [B_new, C, H, W]
        v_mem_final = v_mem_final.view(original_shape[0] // self.n_steps, *original_shape[1:])
        
        # --- FIN DE LA LOGIQUE CORRIGÉE ---
        
        return spikes, v_mem_final
    
    def extra_repr(self) -> str:
        """Représentation string pour print(model)"""
        beta_val = self.get_beta()
        v_th_val = self.v_th.item()
        v_reset_val = self.v_reset.item()
        
        s = (
            f'beta={beta_val:.3f} (learnable={self.learn_beta}), '
            f'v_th={v_th_val:.3f} (learnable={self.learn_v_th}), '
            f'v_reset={v_reset_val:.3f} (learnable={self.learn_v_reset}), '
            f'k_superspike={self.k_superspike.item():.1f}'
        )
        return s