import os
import json
import torch
import numpy as np
import argparse
from safetensors.torch import save_file
from transformers import AutoConfig

def generate_dummy_lora(output_path: str, rank: int, modules: list, base_model_name: str):
    """
    Generates a dummy LoRA adapter with random weights.
    """
    os.makedirs(output_path, exist_ok=True)
    
    print(f"Loading config for {base_model_name}...")
    model_config = AutoConfig.from_pretrained(base_model_name, trust_remote_code=True)
    hidden_size = model_config.hidden_size
    
    # Calculate dimensions
    num_heads = model_config.num_attention_heads
    num_kv_heads = getattr(model_config, "num_key_value_heads", num_heads)
    head_dim = hidden_size // num_heads
    kv_dim = num_kv_heads * head_dim
    
    print(f"Model Config: Hidden={hidden_size}, Heads={num_heads}, KV_Heads={num_kv_heads}, Head_Dim={head_dim}, KV_Dim={kv_dim}")

    config = {
        "base_model_name_or_path": base_model_name,
        "r": rank,
        "lora_alpha": rank * 2,
        "lora_dropout": 0.05,
        "bias": "none",
        "target_modules": modules,
        "peft_type": "LORA"
    }
    
    with open(os.path.join(output_path, "adapter_config.json"), 'w') as f:
        json.dump(config, f, indent=2)
        
    tensors = {}
    for module in modules:
        # Assuming module names like "q_proj", "v_proj"
        # In a real model, keys are full paths like "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
        # For simplicity in this dummy generator, we'll generate for a single layer (layer 0)
        # to keep file size manageable for testing.
        
        key_A = f"base_model.model.model.layers.0.self_attn.{module}.lora_A.weight"
        key_B = f"base_model.model.model.layers.0.self_attn.{module}.lora_B.weight"
        
        # A: (rank, in_dim)
        # B: (out_dim, rank)
        
        in_dim = hidden_size
        out_dim = hidden_size
        
        if module == "v_proj":
            out_dim = kv_dim
        elif module == "k_proj":
            out_dim = kv_dim
        
        tensors[key_A] = torch.randn(rank, in_dim)
        tensors[key_B] = torch.randn(out_dim, rank)
        
    save_file(tensors, os.path.join(output_path, "adapter_model.safetensors"))
    print(f"Generated dummy LoRA at {output_path} with rank {rank}")

def generate_trace(
    output_path: str,
    num_requests: int,
    hot_adapter_name: str,
    cold_adapter_names: list,
    hot_ratio: float = 0.98,
    use_real_lengths: bool = False
):
    """
    Generates a trace file.
    If use_real_lengths is True, samples lengths from a distribution mimicking real chat datasets (e.g., ShareGPT).
    """
    
    requests = []
    
    num_hot = int(num_requests * hot_ratio)
    num_cold = num_requests - num_hot
    
    # Real-world length distribution parameters (approximate)
    # Log-normal distribution for prompt length
    mean_prompt = 128
    sigma_prompt = 0.5
    # Log-normal for output length
    mean_output = 128
    sigma_output = 0.5
    
    def get_len(mean, sigma):
        if use_real_lengths:
            return int(np.random.lognormal(np.log(mean), sigma))
        return mean

    # Generate Hot Requests
    for i in range(num_hot):
        requests.append({
            "prompt_len": get_len(mean_prompt, sigma_prompt),
            "output_len": get_len(mean_output, sigma_output),
            "lora_name": hot_adapter_name,
            "req_id": i
        })
        
    # Generate Cold Requests
    for i in range(num_cold):
        # Pick a random cold adapter
        cold_name = np.random.choice(cold_adapter_names)
        requests.append({
            "prompt_len": get_len(mean_prompt, sigma_prompt),
            "output_len": get_len(mean_output, sigma_output),
            "lora_name": cold_name,
            "req_id": num_hot + i
        })
        
    # Shuffle requests to simulate real traffic
    np.random.shuffle(requests)
    
    with open(output_path, 'w') as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")
            
    print(f"Generated trace at {output_path} with {num_requests} requests.")
    print(f"Hot Ratio: {num_hot/num_requests:.2%}")

def generate_real_trace(
    output_path: str,
    num_requests: int,
    hot_adapter_name: str,
    cold_adapter_names: list,
    hot_ratio: float = 0.98
):
    """
    Generates a trace file using real dataset lengths (Alpaca, OASST1, CodeAlpaca).
    Requires 'datasets' library.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' library not found. Please install it with `pip install datasets`.")
        return

    print("Loading real datasets (Alpaca, OASST1, CodeAlpaca)...")
    
    # Load datasets (using small subsets or streaming if possible to be fast)
    # For simplicity, we'll load a few examples.
    
    # 1. Alpaca (Hot)
    ds_alpaca = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)
    alpaca_iter = iter(ds_alpaca)
    
    # 2. OASST1 (Cold 1)
    ds_oasst = load_dataset("OpenAssistant/oasst1", split="train", streaming=True)
    oasst_iter = iter(ds_oasst)
    
    # 3. CodeAlpaca (Cold 2)
    ds_code = load_dataset("sahil2801/CodeAlpaca-20k", split="train", streaming=True)
    code_iter = iter(ds_code)
    
    requests = []
    num_hot = int(num_requests * hot_ratio)
    num_cold = num_requests - num_hot
    
    # Helper to estimate token length (rough char count / 4)
    def est_len(text):
        return max(16, len(text) // 4)

    print(f"Generating {num_hot} hot requests (Alpaca)...")
    for i in range(num_hot):
        try:
            item = next(alpaca_iter)
            prompt = item['instruction'] + "\n" + item['input']
            output = item['output']
            requests.append({
                "prompt_len": est_len(prompt),
                "output_len": est_len(output),
                "lora_name": hot_adapter_name,
                "req_id": i
            })
        except StopIteration:
            alpaca_iter = iter(ds_alpaca) # Reset if exhausted
            
    print(f"Generating {num_cold} cold requests (OASST1/CodeAlpaca)...")
    for i in range(num_cold):
        cold_name = np.random.choice(cold_adapter_names)
        try:
            if "oasst" in cold_name:
                item = next(oasst_iter)
                prompt = item['text'] # OASST structure is complex, simplifying
                output = "Response" 
            else:
                item = next(code_iter)
                prompt = item['instruction'] + "\n" + item['input']
                output = item['output']
                
            requests.append({
                "prompt_len": est_len(prompt),
                "output_len": est_len(output),
                "lora_name": cold_name,
                "req_id": num_hot + i
            })
        except StopIteration:
             # Fallback if exhausted
             pass

    np.random.shuffle(requests)
    
    with open(output_path, 'w') as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")
            
    print(f"Generated REAL trace at {output_path} with {len(requests)} requests.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", type=str, required=True, choices=["lora", "trace", "real_trace"])
    parser.add_argument("--output", type=str, required=True)
    
    # Lora args
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--base_model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    # Trace args
    parser.add_argument("--num_reqs", type=int, default=1000)
    parser.add_argument("--hot_name", type=str, default="lora_hot")
    parser.add_argument("--cold_names", type=str, nargs="+", default=["lora_cold_1", "lora_cold_2"])
    parser.add_argument("--hot_ratio", type=float, default=0.98)
    parser.add_argument("--real_lengths", action="store_true", help="Use realistic length distribution (LogNormal)")
    
    args = parser.parse_args()
    
    if args.action == "lora":
        generate_dummy_lora(args.output, args.rank, ["q_proj", "v_proj"], args.base_model)
    elif args.action == "trace":
        generate_trace(args.output, args.num_reqs, args.hot_name, args.cold_names, args.hot_ratio, use_real_lengths=args.real_lengths)
    elif args.action == "real_trace":
        generate_real_trace(args.output, args.num_reqs, args.hot_name, args.cold_names, args.hot_ratio)
