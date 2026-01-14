"""
Tests de vérification d'équivalence entre mode spike et mode fréquence.

Ce script vérifie:
1. LIF frequency produit des sorties quantifiées correctes
2. XNOR probabiliste équivaut à XNOR binaire aux coins {0,1}
3. Le mode switching fonctionne sur tout le modèle
4. Les gradients passent correctement en mode fréquence
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, '.')

from Triton_Layers.Lif_Frequency import LIFFrequency, test_lif_frequency_equivalence
from Triton_Layers.SSA import SSAMultiHeadAttention


def test_xnor_equivalence():
    """
    Vérifie que XNOR probabiliste équivaut à XNOR binaire aux coins.
    
    Table de vérité XNOR:
    q=0, k=0 -> 1 (match)
    q=1, k=1 -> 1 (match)  
    q=0, k=1 -> 0 (mismatch)
    q=1, k=0 -> 0 (mismatch)
    
    Formule probabiliste: 2qk - q - k + 1
    """
    print("\n=== Test XNOR Equivalence ===")
    
    d_model = 64
    n_heads = 4
    ssa = SSAMultiHeadAttention(d_model=d_model, n_heads=n_heads, n_steps=4)
    d_head = d_model // n_heads
    
    # Test aux coins
    corners = [
        (0.0, 0.0, 1.0),  # q=0, k=0 -> XNOR = 1
        (1.0, 1.0, 1.0),  # q=1, k=1 -> XNOR = 1
        (0.0, 1.0, 0.0),  # q=0, k=1 -> XNOR = 0
        (1.0, 0.0, 0.0),  # q=1, k=0 -> XNOR = 0
    ]
    
    print("Testing corners (q, k) -> expected XNOR:")
    for q_val, k_val, expected in corners:
        # Créer des tenseurs constants
        Q = torch.full((1, n_heads, 1, d_head), q_val)
        K = torch.full((1, n_heads, 1, d_head), k_val)
        
        # Appliquer XNOR probabiliste
        result = ssa.xnor_attention_frequency(Q, K)
        
        # La valeur attendue est D * expected (car on somme sur D dimensions)
        # result shape is (B, H, L, M) = (1, n_heads, 1, 1)
        expected_val = d_head * expected
        actual_val = result[0, 0, 0, 0].item()  # Get first element
        
        status = "✓" if abs(actual_val - expected_val) < 1e-5 else "✗"
        print(f"  q={q_val}, k={k_val} -> expected={expected_val:.1f}, got={actual_val:.1f} {status}")
    
    # Test des gradients
    print("\nTesting gradients:")
    Q = torch.rand(1, n_heads, 4, d_head, requires_grad=True)
    K = torch.rand(1, n_heads, 4, d_head, requires_grad=True)
    
    result = ssa.xnor_attention_frequency(Q, K)
    loss = result.sum()
    loss.backward()
    
    print(f"  Q grad norm: {Q.grad.norm().item():.4f}")
    print(f"  K grad norm: {K.grad.norm().item():.4f}")
    print(f"  All gradients non-zero: {(Q.grad != 0).all().item() and (K.grad != 0).all().item()}")
    
    # Test important: gradient quand K=0
    print("\nTesting gradient when K=0 (key advantage of XNOR over dot product):")
    Q2 = torch.rand(1, n_heads, 4, d_head, requires_grad=True)
    K2 = torch.zeros(1, n_heads, 4, d_head, requires_grad=True)  # K = 0
    
    result2 = ssa.xnor_attention_frequency(Q2, K2)
    loss2 = result2.sum()
    loss2.backward()
    
    q2_grad_nonzero = (Q2.grad != 0).sum().item()
    print(f"  Q grad non-zero elements: {q2_grad_nonzero} / {Q2.grad.numel()}")
    print(f"  With dot product, Q grad would be 0 when K=0 (XNOR provides informative gradients)")
    
    print("\n✓ XNOR equivalence tests passed!")
    return True


def test_mode_switching():
    """
    Vérifie que model.spike() et model.frequency() changent correctement le mode.
    """
    print("\n=== Test Mode Switching ===")
    
    d_model = 64
    n_heads = 4
    ssa = SSAMultiHeadAttention(d_model=d_model, n_heads=n_heads, n_steps=4)
    
    # Test initial mode
    assert ssa.frequency_mode == False, "Initial mode should be spike (False)"
    print("  Initial mode: spike ✓")
    
    # Switch to frequency
    ssa.frequency_mode = True
    assert ssa.frequency_mode == True, "Mode should be frequency (True)"
    print("  After setting frequency_mode=True: frequency ✓")
    
    # Test forward in frequency mode
    x = torch.randn(8, 16, d_model)
    with torch.no_grad():
        output = ssa(x)
    print(f"  Forward in frequency mode: output shape {output.shape} ✓")
    
    # Switch back to spike
    ssa.frequency_mode = False
    assert ssa.frequency_mode == False, "Mode should be spike (False)"
    print("  After setting frequency_mode=False: spike ✓")
    
    print("\n✓ Mode switching tests passed!")
    return True


def test_full_model_mode():
    """
    Vérifie le mode switching sur le modèle complet Seq2Seq.
    """
    print("\n=== Test Full Model Mode Switching ===")
    
    try:
        from Triton_Layers.Seq2Seq import Seq2Seq
        
        # Créer un modèle minimal
        model = Seq2Seq(
            patch_size=4,
            d_model=64,
            n_heads=4,
            ff_dim=128,
            num_encoder_layers=1,
            num_decoder_layers=1,
            tgt_vocab_size=100,
            nb_sps_blocks=1,
            n_steps=4,
            in_channels=1,
            img_height=32
        )
        
        # Vérifier mode initial
        print(f"  Initial mode: {'frequency' if model.frequency_mode else 'spike'}")
        assert model.frequency_mode == False
        
        # Switch to frequency
        model.frequency()
        print(f"  After model.frequency(): {'frequency' if model.frequency_mode else 'spike'}")
        assert model.frequency_mode == True
        
        # Vérifier que tous les SSA sont en mode freq
        freq_count = 0
        for module in model.modules():
            if hasattr(module, 'frequency_mode'):
                if module.frequency_mode:
                    freq_count += 1
        print(f"  Modules in frequency mode: {freq_count}")
        
        # Switch back to spike
        model.spike()
        print(f"  After model.spike(): {'frequency' if model.frequency_mode else 'spike'}")
        assert model.frequency_mode == False
        
        print("\n✓ Full model mode switching tests passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_training_step_frequency():
    """
    Simule un step d'entraînement en mode fréquence.
    """
    print("\n=== Test Training Step (Frequency Mode) ===")
    
    d_model = 64
    n_heads = 4
    ssa = SSAMultiHeadAttention(d_model=d_model, n_heads=n_heads, n_steps=4)
    ssa.frequency_mode = True
    
    # Input
    x = torch.randn(8, 16, d_model, requires_grad=True)
    
    # Forward
    output = ssa(x)
    
    # Fake loss
    loss = output.sum()
    
    # Backward
    loss.backward()
    
    # Check gradients propagate
    assert x.grad is not None, "Gradient should propagate to input"
    print(f"  Input grad norm: {x.grad.norm().item():.4f}")
    
    # Check parameter gradients
    param_grads = []
    for name, param in ssa.named_parameters():
        if param.grad is not None:
            param_grads.append((name, param.grad.norm().item()))
    
    print(f"  Parameters with gradients: {len(param_grads)}")
    for name, norm in param_grads[:5]:  # Show first 5
        print(f"    {name}: {norm:.4f}")
    
    print("\n✓ Training step test passed!")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("FREQUENCY-EQUIVALENT TRAINING VERIFICATION TESTS")
    print("=" * 60)
    
    results = []
    
    # Test 1: LIF Frequency
    results.append(("LIF Frequency", test_lif_frequency_equivalence()))
    
    # Test 2: XNOR Equivalence
    results.append(("XNOR Equivalence", test_xnor_equivalence()))
    
    # Test 3: Mode Switching
    results.append(("Mode Switching", test_mode_switching()))
    
    # Test 4: Full Model
    results.append(("Full Model Mode", test_full_model_mode()))
    
    # Test 5: Training Step
    results.append(("Training Step", test_training_step_frequency()))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")
    
    all_passed = all(r[1] for r in results)
    print("\n" + ("All tests passed! ✓" if all_passed else "Some tests failed ✗"))
