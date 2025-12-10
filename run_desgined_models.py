import os
import sys
import shutil
import subprocess
import argparse
import json
import random
import pandas as pd
from huggingface_hub import snapshot_download

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tools.slora.builder import SLoRABuilder

def run_command(cmd):
    print(f"Running: {cmd}")
    subprocess.check_call(cmd, shell=True)

def download_model(repo_id, local_dir):
    if os.path.exists(local_dir):
        print(f"Model {repo_id} already exists at {local_dir}. Skipping download.")
        return local_dir
    print(f"Downloading {repo_id} to {local_dir}...")
    try:
        return snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False)
    except Exception as e:
        print(f"Error downloading {repo_id}: {e}")
        print("Please ensure you are logged in to HuggingFace (huggingface-cli login) if accessing gated models.")
        sys.exit(1)

def generate_trace(trace_file, num_reqs, hot_ratio, hot_name, cold_names):
    num_hot = int(num_reqs * hot_ratio)
    num_cold = num_reqs - num_hot
    
    reqs = []
    # Hot requests
    for i in range(num_hot):
        reqs.append({
            "req_id": i,
            "prompt_len": random.randint(100, 512),
            "output_len": random.randint(50, 200),
            "lora_name": hot_name
        })
        
    # Cold requests
    for i in range(num_hot, num_reqs):
        cold_name = random.choice(cold_names)
        reqs.append({
            "req_id": i,
            "prompt_len": random.randint(100, 512),
            "output_len": random.randint(50, 200),
            "lora_name": cold_name
        })
        
    random.shuffle(reqs)
    
    with open(trace_file, 'w') as f:
        for req in reqs:
            f.write(json.dumps(req) + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default="designed_models_workspace")
    parser.add_argument("--num_reqs", type=int, default=200) # Lower default for real models
    args = parser.parse_args()

    work_dir = os.path.abspath(args.work_dir)
    models_dir = os.path.join(work_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    print(f"=== Starting Designed Models Benchmark in {work_dir} ===")
    
    # 1. Define Models
    base_model_id = "baffo32/decapoda-research-llama-7B-hf"
    hot_lora_id = "tloen/alpaca-lora-7b"
    # Replaced broken PHI-LM with Guanaco (OASST1 based, fits "Community Chat" role)
    cold_lora_1_id = "plncmm/guanaco-lora-7b" 
    cold_lora_2_id = "winddude/wizardLM-LlaMA-LoRA-7B"
    
    hot_name = "alpaca"
    cold_name_1 = "guanaco"
    cold_name_2 = "wizard"
    
    # 2. Download Models
    print("\n--- Step 1: Preparing Models ---")
    base_model_path = download_model(base_model_id, os.path.join(models_dir, "llama-7b"))
    hot_lora_path = download_model(hot_lora_id, os.path.join(models_dir, "alpaca-lora"))
    cold_lora_1_path = download_model(cold_lora_1_id, os.path.join(models_dir, "guanaco-lora"))
    cold_lora_2_path = download_model(cold_lora_2_id, os.path.join(models_dir, "wizard-lora"))
    
    # Paths for SLoRA artifacts
    fused_model_path = os.path.join(work_dir, "fused_base_llama1")
    delta_lora_1_path = os.path.join(work_dir, "guanaco_delta")
    delta_lora_2_path = os.path.join(work_dir, "wizard_delta")
    
    # 3. Build SLoRA
    print("\n--- Step 2: Building SLoRA (Offline Weights Update) ---")
    builder = SLoRABuilder(base_model_path, device="auto")
    
    # Check if fused model is valid (has tokenizer)
    fused_model_valid = os.path.exists(fused_model_path) and os.path.exists(os.path.join(fused_model_path, "tokenizer.model"))
    
    # Force rebuild to apply tokenizer fix
    fused_model_valid = False 
    
    if not fused_model_valid:
        if os.path.exists(fused_model_path):
            print(f"Fused model at {fused_model_path} is missing tokenizer or incomplete. Rebuilding...")
            # Ensure we don't fail on permission errors if files are open
            try:
                shutil.rmtree(fused_model_path)
            except Exception as e:
                print(f"Warning: Could not delete {fused_model_path}: {e}")
            
        print(f"Merging {hot_lora_id} into Base Model...")
        builder.merge_base_model(hot_lora_path, fused_model_path)
    else:
        print("Fused model already exists and appears valid. Skipping merge.")
        
    if not os.path.exists(delta_lora_1_path):
        print(f"Creating Delta LoRA for {cold_lora_1_id}...")
        builder.build_delta_lora(hot_lora_path, cold_lora_1_path, delta_lora_1_path)
        
    if not os.path.exists(delta_lora_2_path):
        print(f"Creating Delta LoRA for {cold_lora_2_id}...")
        builder.build_delta_lora(hot_lora_path, cold_lora_2_path, delta_lora_2_path)
        
    # 4. Run Benchmark
    print("\n--- Step 3: Running Benchmark ---")
    
    ratios = [0.0, 0.5, 1.0] # Restore full set
    results = []
    
    for ratio in ratios:
        print(f"\n>>> Testing Hot Ratio: {ratio}")
        
        trace_file = os.path.join(work_dir, f"trace_{ratio}.jsonl")
        generate_trace(trace_file, args.num_reqs, ratio, hot_name, [cold_name_1, cold_name_2])
        
        output_json = os.path.join(work_dir, f"results_{ratio}.json")
        
        # Construct command
        # python benchmarks/benchmark_slora.py --trace ... --model ... --mode slora ...
        
        cmd = [
            "python", "benchmarks/benchmark_slora.py",
            "--trace", trace_file,
            "--model", fused_model_path,
            "--mode", "slora",
            "--hot_lora_path", hot_lora_path,
            "--hot_lora_name", hot_name,
            "--cold_loras", f"{cold_name_1}:{delta_lora_1_path}", f"{cold_name_2}:{delta_lora_2_path}",
            "--output_json", output_json
        ]
        
        try:
            run_command(" ".join(cmd))
            
            with open(output_json, 'r') as f:
                res = json.load(f)
                res["hot_ratio"] = ratio
                res["system"] = "SLoRA" # Mark as SLoRA
                results.append(res)
                
        except subprocess.CalledProcessError as e:
            print(f"Error running benchmark for ratio {ratio}: {e}")

    # --- Run Baseline (Standard vLLM) ---
    print("\n>>> Running Baseline (Standard vLLM) ...")
    # For baseline, we use the base model directly without LoRA switching (or just one LoRA loaded statically)
    # To make a fair comparison, we can run the same trace but treat all requests as hitting the base model 
    # (simulating a scenario where we can't switch, or just measuring raw throughput of the engine)
    # OR, if we want to simulate "Naive" serving, we would need to reload weights, which vLLM doesn't support easily in one process.
    # So usually Baseline = "Ideal Upper Bound" (No LoRA overhead) or "Naive" (High latency).
    # Let's run a "No-LoRA" baseline to see the overhead of our SLoRA kernel.
    
    baseline_trace_file = os.path.join(work_dir, "trace_baseline.jsonl")
    generate_trace(baseline_trace_file, args.num_reqs, 1.0, "base", []) # All requests to base
    
    baseline_output_json = os.path.join(work_dir, "results_baseline.json")
    
    cmd_baseline = [
        "python", "benchmarks/benchmark_slora.py",
        "--trace", baseline_trace_file,
        "--model", base_model_path, # Use original base model
        "--mode", "base", # New mode for baseline
        "--output_json", baseline_output_json
    ]
    
    try:
        run_command(" ".join(cmd_baseline))
        with open(baseline_output_json, 'r') as f:
            res = json.load(f)
            res["hot_ratio"] = 1.0
            res["system"] = "Baseline (No LoRA)"
            results.append(res)
    except subprocess.CalledProcessError as e:
        print(f"Error running baseline: {e}")

    # Save Results
    df = pd.DataFrame(results)
    print("\n=== Results ===")
    print(df)
    df.to_csv(os.path.join(work_dir, "designed_results.csv"), index=False)

if __name__ == "__main__":
    main()
