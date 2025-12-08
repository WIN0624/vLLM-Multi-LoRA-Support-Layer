import torch
from typing import Tuple

def absorb_scaling(
    A: torch.Tensor,
    B: torch.Tensor,
    rank: int,
    alpha: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Absorb the LoRA scaling factor (alpha / rank) into the weights.
    W_lora = (alpha / rank) * B @ A
    We want W_lora = B_new @ A_new
    So we can scale B by (alpha / rank) or split it.
    Usually, it's safer to scale B, as A is often initialized with specific variance.
    """
    scaling = alpha / rank
    return A, B * scaling

def create_delta_lora_weights(
    A_hot: torch.Tensor,
    B_hot: torch.Tensor,
    rank_hot: int,
    alpha_hot: float,
    A_cold: torch.Tensor,
    B_cold: torch.Tensor,
    rank_cold: int,
    alpha_cold: float
) -> Tuple[torch.Tensor, torch.Tensor, int, float]:
    """
    Construct Delta LoRA weights (A_delta, B_delta) such that:
    (alpha_new / rank_new) * B_delta @ A_delta = (alpha_cold/rank_cold) * B_cold @ A_cold - (alpha_hot/rank_hot) * B_hot @ A_hot
    
    Strategy:
    1. Absorb scalings into B matrices.
    2. Construct A_delta = concat([A_cold, A_hot], dim=0)
    3. Construct B_delta = concat([B_cold_scaled, -B_hot_scaled], dim=1)
    4. Set new rank = rank_cold + rank_hot
    5. Set new alpha = new rank (so scaling factor is 1)
    """
    
    # 1. Absorb scalings
    # Note: A shape is (rank, in_dim), B shape is (out_dim, rank)
    scaling_hot = alpha_hot / rank_hot
    scaling_cold = alpha_cold / rank_cold
    
    B_hot_scaled = B_hot * scaling_hot
    B_cold_scaled = B_cold * scaling_cold
    
    # 2. Construct A_delta
    # A_hot: (r_h, d_in), A_cold: (r_c, d_in)
    # A_delta: (r_c + r_h, d_in)
    A_delta = torch.cat([A_cold, A_hot], dim=0)
    
    # 3. Construct B_delta
    # B_hot_scaled: (d_out, r_h), B_cold_scaled: (d_out, r_c)
    # We want B_delta @ A_delta = B_c' @ A_c - B_h' @ A_h
    # [B_c', -B_h'] @ [A_c; A_h] = B_c' @ A_c + (-B_h') @ A_h
    # B_delta: (d_out, r_c + r_h)
    B_delta = torch.cat([B_cold_scaled, -B_hot_scaled], dim=1)
    
    new_rank = rank_cold + rank_hot
    new_alpha = float(new_rank) # So that new_scaling = 1.0
    
    return A_delta, B_delta, new_rank, new_alpha
