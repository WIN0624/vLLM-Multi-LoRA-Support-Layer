import os
import sys
import shutil
import subprocess
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tools.slora.builder import SLoRABuilder

def run_command(cmd):
    print(f"Running: {cmd}")
    subprocess.check_call(cmd, shell=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="HuggingFace model name or path")
    parser.add_argument("--work_dir", type=str, default="real_experiment_workspace")
    parser.add_argument("--num_reqs", type=int, default=200)
    parser.add_argument("--hot_ratio", type=float, default=0.80) # 80% hot for realistic skew
    args = parser.parse_args()

    work_dir = os.path.abspath(args.work_dir)
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir)

    print(f"=== Starting Real Workload Experiment in {work_dir} ===")
    
    # Paths
    hot_lora_path = os.path.join(work_dir, "hot_lora")
    cold_lora_path = os.path.join(work_dir, "cold_lora")
    fused_model_path = os.path.join(work_dir, "fused_base")
    delta_lora_path = os.path.join(work_dir, "delta_lora")
    trace_path = os.path.join(work_dir, "trace.jsonl")
    
    # 1. Generate Dummy LoRAs
    print("\n--- Step 1: Generating Dummy LoRAs ---")
    # Using Rank 16 for Hot and Rank 8 for Cold as a typical scenario
    run_command(f"python tools/slora/generate_data.py --action lora --output {hot_lora_path} --rank 16 --base_model {args.base_model}")
    run_command(f"python tools/slora/generate_data.py --action lora --output {cold_lora_path} --rank 8 --base_model {args.base_model}")
    
    # 2. Build SLoRA Assets
    print("\n--- Step 2: Building SLoRA Assets (Offline Phase) ---")
    builder = SLoRABuilder(base_model_path=args.base_model, device="cuda")
    
    # 2a. Merge Hot -> Fused Base
    builder.merge_base_model(hot_lora_path, fused_model_path)
    
    # 2b. Build Delta (Cold - Hot)
    builder.build_delta_lora(hot_lora_path, cold_lora_path, delta_lora_path)
    
    # 3. Generate Trace with Real Lengths
    print("\n--- Step 3: Generating Traffic Trace (Real Lengths) ---")
    # Added --real_lengths flag here
    run_command(f"python tools/slora/generate_data.py --action trace --output {trace_path} --num_reqs {args.num_reqs} --hot_name lora_hot --cold_names lora_cold --hot_ratio {args.hot_ratio} --real_lengths")
    
    # 4. Run Baseline Benchmark
    print("\n--- Step 4: Running Baseline Benchmark ---")
    
    # Copy benchmark script to work_dir
    shutil.copy("benchmarks/benchmark_slora.py", os.path.join(work_dir, "benchmark_slora.py"))
    
    cmd_baseline = (
        f"python benchmark_slora.py "
        f"--trace trace.jsonl "
        f"--model {args.base_model} "
        f"--mode baseline "
        f"--hot_lora_path hot_lora "
        f"--hot_lora_name lora_hot "
        f"--cold_loras lora_cold:cold_lora"
    )
    
    print(f"Running in {work_dir}: {cmd_baseline}")
    subprocess.check_call(cmd_baseline, shell=True, cwd=work_dir)
    
    # 5. Run SLoRA Benchmark
    print("\n--- Step 5: Running SLoRA Benchmark ---")
    
    cmd_slora = (
        f"python benchmark_slora.py "
        f"--trace trace.jsonl "
        f"--model {fused_model_path} "
        f"--mode slora "
        f"--hot_lora_path hot_lora "
        f"--hot_lora_name lora_hot "
        f"--cold_loras lora_cold:{delta_lora_path}"
    )
    
    print(f"Running in {work_dir}: {cmd_slora}")
    subprocess.check_call(cmd_slora, shell=True, cwd=work_dir)

if __name__ == "__main__":
    main()
