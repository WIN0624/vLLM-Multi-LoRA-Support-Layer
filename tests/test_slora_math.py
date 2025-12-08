import torch
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tools.slora.utils import create_delta_lora_weights

def test_delta_lora_math():
    print("Running SLoRA Math Verification...")
    torch.manual_seed(42)
    
    # Dimensions
    d_in = 64
    d_out = 128
    
    # Hot LoRA
    rank_hot = 8
    alpha_hot = 16.0
    A_hot = torch.randn(rank_hot, d_in)
    B_hot = torch.randn(d_out, rank_hot)
    
    # Cold LoRA
    rank_cold = 4
    alpha_cold = 8.0
    A_cold = torch.randn(rank_cold, d_in)
    B_cold = torch.randn(d_out, rank_cold)
    
    # Base Model Weight (Dummy)
    W_base = torch.randn(d_out, d_in)
    
    # 1. Calculate Expected Target (Base + Cold)
    scale_cold = alpha_cold / rank_cold
    W_cold_target = W_base + scale_cold * (B_cold @ A_cold)
    
    # 2. Calculate Fused Base (Base + Hot)
    scale_hot = alpha_hot / rank_hot
    W_fused = W_base + scale_hot * (B_hot @ A_hot)
    
    # 3. Calculate Delta LoRA
    A_delta, B_delta, rank_delta, alpha_delta = create_delta_lora_weights(
        A_hot, B_hot, rank_hot, alpha_hot,
        A_cold, B_cold, rank_cold, alpha_cold
    )
    
    # 4. Apply Delta to Fused Base
    scale_delta = alpha_delta / rank_delta # Should be 1.0
    W_reconstructed = W_fused + scale_delta * (B_delta @ A_delta)
    
    # 5. Verify
    diff = (W_reconstructed - W_cold_target).abs().max()
    print(f"Max difference: {diff.item()}")
    
    if diff < 1e-5:
        print("SUCCESS: Delta LoRA correctly transforms Fused Base to Cold Target.")
    else:
        print(f"FAILURE: Math verification failed! Diff: {diff}")
        sys.exit(1)

if __name__ == "__main__":
    test_delta_lora_math()
