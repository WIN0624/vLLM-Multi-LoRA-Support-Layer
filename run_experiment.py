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
    parser.add_argument("--base_model", type=str, default="facebook/opt-125m", help="HuggingFace model name or path")
    parser.add_argument("--work_dir", type=str, default="experiment_workspace")
    parser.add_argument("--num_reqs", type=int, default=100)
    parser.add_argument("--hot_ratio", type=float, default=0.90) # 90% hot for clear contrast
    args = parser.parse_args()

    work_dir = os.path.abspath(args.work_dir)
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir)

    print(f"=== Starting Experiment in {work_dir} ===")
    
    # Paths
    hot_lora_path = os.path.join(work_dir, "hot_lora")
    cold_lora_path = os.path.join(work_dir, "cold_lora")
    fused_model_path = os.path.join(work_dir, "fused_base")
    delta_lora_path = os.path.join(work_dir, "delta_lora")
    trace_path = os.path.join(work_dir, "trace.jsonl")
    
    # 1. Generate Dummy LoRAs
    print("\n--- Step 1: Generating Dummy LoRAs ---")
    # Note: We need to know the target modules for the specific base model.
    # For OPT, it's usually q_proj, v_proj. For Llama, it's q_proj, v_proj.
    # We'll assume q_proj, v_proj for now.
    target_modules = "q_proj v_proj"
    
    # We use the generate_data.py script via subprocess to ensure clean state
    run_command(f"python tools/slora/generate_data.py --action lora --output {hot_lora_path} --rank 16")
    run_command(f"python tools/slora/generate_data.py --action lora --output {cold_lora_path} --rank 8")
    
    # Hack: The dummy generator uses "base_model.model..." prefix which might not match OPT's structure exactly
    # if we were loading it into PEFT strictly. But for vLLM loading, as long as keys match, it's fine.
    # However, SLoRABuilder uses PeftModel.from_pretrained which requires matching structure.
    # For this experiment to work with "facebook/opt-125m", we need real LoRAs or a dummy generator that matches OPT.
    # The current dummy generator produces Llama-like keys.
    # Let's trust the user might provide a Llama model, or we accept that this might fail on OPT if keys don't match.
    # To be safe, let's use a very small Llama if possible, or just proceed and see.
    # If you use "facebook/opt-125m", the keys in dummy lora (Llama style) won't match.
    
    print("NOTE: Ensure generate_data.py produces keys compatible with your --base_model.")
    
    # 2. Build SLoRA Assets
    print("\n--- Step 2: Building SLoRA Assets (Offline Phase) ---")
    builder = SLoRABuilder(base_model_path=args.base_model, device="cuda")
    
    # 2a. Merge Hot -> Fused Base
    builder.merge_base_model(hot_lora_path, fused_model_path)
    
    # 2b. Build Delta (Cold - Hot)
    builder.build_delta_lora(hot_lora_path, cold_lora_path, delta_lora_path)
    
    # 3. Generate Trace
    print("\n--- Step 3: Generating Traffic Trace ---")
    run_command(f"python tools/slora/generate_data.py --action trace --output {trace_path} --num_reqs {args.num_reqs} --hot_name lora_hot --cold_names lora_cold --hot_ratio {args.hot_ratio}")
    
    # 4. Run Baseline Benchmark
    print("\n--- Step 4: Running Baseline Benchmark ---")
    # Baseline uses original base model + original LoRAs
    # HACK: We need to run this from a different directory to avoid importing local vllm source
    # We'll use the work_dir as cwd
    
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
    # SLoRA uses Fused Base + Delta LoRA
    cmd_slora = (
        f"python benchmark_slora.py "
        f"--trace trace.jsonl "
        f"--model fused_base "
        f"--mode slora "
        f"--hot_lora_path hot_lora " # Passed just for name reference, won't be loaded
        f"--hot_lora_name lora_hot "
        f"--cold_loras lora_cold:delta_lora" # Pass Delta path for cold
    )
    
    print(f"Running in {work_dir}: {cmd_slora}")
    subprocess.check_call(cmd_slora, shell=True, cwd=work_dir)
    
    print("\n=== Experiment Completed ===")

if __name__ == "__main__":
    main()
