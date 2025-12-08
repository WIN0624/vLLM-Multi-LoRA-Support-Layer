import os
import json
import torch
import shutil
import sys
from safetensors.torch import save_file, load_file

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tools.slora.builder import SLoRABuilder

def test_builder_delta_generation():
    print("Running SLoRA Builder Integration Test...")
    # Setup paths
    test_dir = "tests/temp_slora_test"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir, exist_ok=True)
    
    hot_dir = os.path.join(test_dir, "hot_lora")
    cold_dir = os.path.join(test_dir, "cold_lora")
    output_dir = os.path.join(test_dir, "delta_lora")
    
    os.makedirs(hot_dir, exist_ok=True)
    os.makedirs(cold_dir, exist_ok=True)
    
    # 1. Create Dummy Hot LoRA
    hot_rank = 8
    hot_alpha = 16
    hot_config = {
        "r": hot_rank,
        "lora_alpha": hot_alpha,
        "target_modules": ["q_proj"],
        "peft_type": "LORA"
    }
    with open(os.path.join(hot_dir, "adapter_config.json"), 'w') as f:
        json.dump(hot_config, f)
        
    hot_tensors = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.randn(hot_rank, 32),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.randn(32, hot_rank)
    }
    save_file(hot_tensors, os.path.join(hot_dir, "adapter_model.safetensors"))
    
    # 2. Create Dummy Cold LoRA
    cold_rank = 4
    cold_alpha = 8
    cold_config = {
        "r": cold_rank,
        "lora_alpha": cold_alpha,
        "target_modules": ["q_proj"],
        "peft_type": "LORA"
    }
    with open(os.path.join(cold_dir, "adapter_config.json"), 'w') as f:
        json.dump(cold_config, f)
        
    cold_tensors = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.randn(cold_rank, 32),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.randn(32, cold_rank)
    }
    save_file(cold_tensors, os.path.join(cold_dir, "adapter_model.safetensors"))
    
    # 3. Run Builder
    # We don't need a real base model path for build_delta_lora
    builder = SLoRABuilder(base_model_path="dummy")
    builder.build_delta_lora(hot_dir, cold_dir, output_dir)
    
    # 4. Verify
    if not os.path.exists(os.path.join(output_dir, "adapter_config.json")):
        print("FAILURE: adapter_config.json not found")
        sys.exit(1)
    if not os.path.exists(os.path.join(output_dir, "adapter_model.safetensors")):
        print("FAILURE: adapter_model.safetensors not found")
        sys.exit(1)
    
    with open(os.path.join(output_dir, "adapter_config.json"), 'r') as f:
        delta_config = json.load(f)
        
    expected_rank = hot_rank + cold_rank
    print(f"Expected Rank: {expected_rank}, Actual Rank: {delta_config['r']}")
    
    if delta_config['r'] != expected_rank:
        print(f"FAILURE: Rank mismatch. Expected {expected_rank}, got {delta_config['r']}")
        sys.exit(1)
        
    if delta_config['lora_alpha'] != expected_rank:
        print(f"FAILURE: Alpha mismatch. Expected {expected_rank}, got {delta_config['lora_alpha']}")
        sys.exit(1)
    
    delta_tensors = load_file(os.path.join(output_dir, "adapter_model.safetensors"))
    delta_A = delta_tensors["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"]
    delta_B = delta_tensors["base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"]
    
    if delta_A.shape != (expected_rank, 32):
        print(f"FAILURE: A shape mismatch. Expected {(expected_rank, 32)}, got {delta_A.shape}")
        sys.exit(1)
        
    if delta_B.shape != (32, expected_rank):
        print(f"FAILURE: B shape mismatch. Expected {(32, expected_rank)}, got {delta_B.shape}")
        sys.exit(1)
    
    print("SUCCESS: Delta LoRA generation verified.")
    
    # Cleanup
    shutil.rmtree(test_dir)

if __name__ == "__main__":
    test_builder_delta_generation()
