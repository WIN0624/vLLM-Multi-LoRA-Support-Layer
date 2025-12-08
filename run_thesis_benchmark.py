import os
import sys
import shutil
import subprocess
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt

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
    parser.add_argument("--work_dir", type=str, default="thesis_benchmark_workspace")
    parser.add_argument("--num_reqs", type=int, default=1000)
    args = parser.parse_args()

    work_dir = os.path.abspath(args.work_dir)
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir)

    print(f"=== Starting Thesis Benchmark in {work_dir} ===")
    
    # Paths
    hot_lora_path = os.path.join(work_dir, "alpaca_lora")
    cold_lora_1_path = os.path.join(work_dir, "oasst1_lora")
    cold_lora_2_path = os.path.join(work_dir, "codealpaca_lora")
    
    fused_model_path = os.path.join(work_dir, "fused_base")
    delta_lora_1_path = os.path.join(work_dir, "oasst1_delta")
    delta_lora_2_path = os.path.join(work_dir, "codealpaca_delta")
    
    # 1. Generate Dummy LoRAs
    print("\n--- Step 1: Generating Dummy LoRAs (Alpaca, OASST1, CodeAlpaca) ---")
    # Alpaca (Hot) - Rank 16
    run_command(f"python tools/slora/generate_data.py --action lora --output {hot_lora_path} --rank 16 --base_model {args.base_model}")
    # OASST1 (Cold) - Rank 8
    run_command(f"python tools/slora/generate_data.py --action lora --output {cold_lora_1_path} --rank 8 --base_model {args.base_model}")
    # CodeAlpaca (Cold) - Rank 8
    run_command(f"python tools/slora/generate_data.py --action lora --output {cold_lora_2_path} --rank 8 --base_model {args.base_model}")
    
    # 2. Build SLoRA Assets
    print("\n--- Step 2: Building SLoRA Assets (Offline Phase) ---")
    builder = SLoRABuilder(base_model_path=args.base_model, device="cuda")
    
    # 2a. Merge Hot (Alpaca) -> Fused Base
    builder.merge_base_model(hot_lora_path, fused_model_path)
    
    # 2b. Build Deltas
    builder.build_delta_lora(hot_lora_path, cold_lora_1_path, delta_lora_1_path)
    builder.build_delta_lora(hot_lora_path, cold_lora_2_path, delta_lora_2_path)
    
    # Copy benchmark script
    shutil.copy("benchmarks/benchmark_slora.py", os.path.join(work_dir, "benchmark_slora.py"))
    
    # 3. Run Experiments across different Hot Ratios
    ratios = [0.0, 0.2, 0.5, 0.8, 0.9, 0.98, 1.0]
    results = []
    
    for ratio in ratios:
        print(f"\n>>> Testing Hot Ratio: {ratio:.2%} <<<")
        trace_path = os.path.join(work_dir, f"trace_{ratio}.jsonl")
        
        # Generate Trace (Try real trace first, fallback to dummy if datasets not installed)
        try:
            run_command(f"python tools/slora/generate_data.py --action real_trace --output {trace_path} --num_reqs {args.num_reqs} --hot_name alpaca --cold_names oasst1 codealpaca --hot_ratio {ratio}")
        except subprocess.CalledProcessError:
            print("Real trace generation failed (likely missing datasets lib). Falling back to dummy trace.")
            run_command(f"python tools/slora/generate_data.py --action trace --output {trace_path} --num_reqs {args.num_reqs} --hot_name alpaca --cold_names oasst1 codealpaca --hot_ratio {ratio} --real_lengths")
        
        # Run Baseline
        baseline_json = os.path.join(work_dir, f"baseline_{ratio}.json")
        cmd_baseline = (
            f"python benchmark_slora.py "
            f"--trace {os.path.basename(trace_path)} "
            f"--model {args.base_model} "
            f"--mode baseline "
            f"--hot_lora_path alpaca_lora "
            f"--hot_lora_name alpaca "
            f"--cold_loras oasst1:oasst1_lora codealpaca:codealpaca_lora "
            f"--output_json {os.path.basename(baseline_json)}"
        )
        print(f"Running Baseline...")
        subprocess.check_call(cmd_baseline, shell=True, cwd=work_dir)
        
        with open(baseline_json, 'r') as f:
            baseline_metrics = json.load(f)
            
        # Run SLoRA
        slora_json = os.path.join(work_dir, f"slora_{ratio}.json")
        cmd_slora = (
            f"python benchmark_slora.py "
            f"--trace {os.path.basename(trace_path)} "
            f"--model fused_base "
            f"--mode slora "
            f"--hot_lora_path alpaca_lora "
            f"--hot_lora_name alpaca "
            f"--cold_loras oasst1:{delta_lora_1_path} codealpaca:{delta_lora_2_path} "
            f"--output_json {os.path.basename(slora_json)}"
        )
        print(f"Running SLoRA...")
        subprocess.check_call(cmd_slora, shell=True, cwd=work_dir)
        
        with open(slora_json, 'r') as f:
            slora_metrics = json.load(f)
            
        # Record Results
        res = {
            "ratio": ratio,
            # Requests per second
            "baseline_req_s": baseline_metrics["throughput_req_s"],
            "slora_req_s": slora_metrics["throughput_req_s"],
            "gain_req_s": slora_metrics["throughput_req_s"] / baseline_metrics["throughput_req_s"],
            
            # Tokens per second
            "baseline_tok_s": baseline_metrics["throughput_tok_s"],
            "slora_tok_s": slora_metrics["throughput_tok_s"],
            "gain_tok_s": slora_metrics["throughput_tok_s"] / baseline_metrics["throughput_tok_s"],

            # Latency Metrics
            "baseline_ttft_ms": baseline_metrics["avg_ttft_ms"],
            "slora_ttft_ms": slora_metrics["avg_ttft_ms"],
            
            "baseline_tpot_ms": baseline_metrics["avg_tpot_ms"],
            "slora_tpot_ms": slora_metrics["avg_tpot_ms"],
            
            "baseline_e2e_s": baseline_metrics["avg_e2e_s"],
            "slora_e2e_s": slora_metrics["avg_e2e_s"]
        }
        results.append(res)
        print(f"Result for {ratio}: Gain (Req/s) = {res['gain_req_s']:.2f}x")

    # 4. Save and Plot
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(work_dir, "thesis_results.csv"), index=False)
    print("\n=== Final Results ===")
    print(df)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(df["ratio"], df["gain_req_s"], marker='o', linewidth=2, label='Throughput Gain (Req/s)')
    plt.axhline(y=1.0, color='r', linestyle='--', label='Baseline (1.0x)')
    plt.xlabel("Hot Model Portion (Alpaca %)")
    plt.ylabel("Throughput Gain (SLoRA / Baseline)")
    plt.title("SLoRA Performance Gain vs. Hot Model Popularity")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(work_dir, "thesis_plot.png"))
    print(f"Plot saved to {os.path.join(work_dir, 'thesis_plot.png')}")

if __name__ == "__main__":
    main()
