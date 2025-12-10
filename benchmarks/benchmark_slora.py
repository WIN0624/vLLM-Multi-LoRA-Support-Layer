import argparse
import asyncio
import json
import time
import os
import sys
from typing import List, Optional

# HACK: Prevent importing vllm from the current directory (source repo)
# We want to use the installed vllm package
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_dir)
if repo_root in sys.path:
    sys.path.remove(repo_root)
if os.getcwd() in sys.path:
    sys.path.remove(os.getcwd())

from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from vllm.lora.request import LoRARequest

async def run_benchmark(
    trace_file: str,
    model_path: str,
    mode: str, # "baseline" or "slora"
    hot_lora_path: str,
    hot_lora_name: str,
    delta_lora_map: dict, # {cold_name: delta_path} for slora mode
    cold_lora_map: dict,  # {cold_name: cold_path} for baseline mode
):
    # 1. Initialize Engine
    print(f"Initializing Engine in {mode} mode...")
    engine_args = EngineArgs(
        model=model_path,
        enable_lora=True,
        max_loras=1, # Constrain to 1 to demonstrate SLoRA's advantage in mixed batches
        max_lora_rank=64, # Ensure enough rank for Delta LoRA
        gpu_memory_utilization=0.8,
        enforce_eager=True # Often safer for LoRA benchmarks
    )
    engine = LLMEngine.from_engine_args(engine_args)
    
    # 2. Load Trace
    with open(trace_file, 'r') as f:
        requests_data = [json.loads(line) for line in f]
        
    print(f"Loaded {len(requests_data)} requests.")
    
    # 3. Process Requests
    start_time = time.time()
    
    request_stats = {} # req_id -> {arrival, first_token, end, tokens}

    # Submit all requests
    for req in requests_data:
        prompt = "Hello " * (req['prompt_len'] // 2) # Dummy prompt
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=req['output_len'],
            ignore_eos=True
        )
        
        req_id = str(req['req_id'])
        lora_name = req['lora_name']
        
        lora_request = None
        
        if mode == "baseline":
            # Baseline: Always load the requested LoRA
            if lora_name == hot_lora_name:
                lora_request = LoRARequest(hot_lora_name, 1, hot_lora_path)
            else:
                # Cold LoRA
                path = cold_lora_map.get(lora_name)
                if path:
                    # Use a unique ID for each cold lora to force switching if needed, 
                    # or same ID if we want to reuse loaded ones. 
                    # Let's assume unique ID based on name hash to simulate distinct adapters.
                    lora_id = abs(hash(lora_name)) % 1000 + 2 
                    lora_request = LoRARequest(lora_name, lora_id, path)
                    
        elif mode == "slora":
            # SLoRA: 
            # If Hot -> No LoRA (because it's fused)
            # If Cold -> Load Delta LoRA
            
            if lora_name == hot_lora_name:
                lora_request = None # Zero Overhead!
            else:
                # Cold LoRA -> Load Delta
                delta_path = delta_lora_map.get(lora_name)
                if delta_path:
                    lora_id = abs(hash(lora_name)) % 1000 + 2
                    lora_request = LoRARequest(lora_name + "_delta", lora_id, delta_path)
        
        engine.add_request(
            req_id,
            prompt,
            sampling_params,
            lora_request=lora_request
        )
        request_stats[req_id] = {"arrival": time.time(), "first_token": None, "end": None, "tokens": 0}
        
    # Run Engine
    print("Starting inference...")
    
    num_requests = len(requests_data)
    finished_requests = 0
    
    while engine.has_unfinished_requests():
        request_outputs: List[RequestOutput] = engine.step()
        now = time.time()
        for request_output in request_outputs:
            rid = request_output.request_id
            
            # Check for first token
            if request_stats[rid]["first_token"] is None and len(request_output.outputs[0].token_ids) > 0:
                 request_stats[rid]["first_token"] = now
            
            if request_output.finished:
                finished_requests += 1
                request_stats[rid]["end"] = now
                request_stats[rid]["tokens"] = len(request_output.outputs[0].token_ids)
                
    end_time = time.time()
    total_time = end_time - start_time
    throughput = num_requests / total_time
    
    # Calculate Metrics
    total_tokens = sum(s["tokens"] for s in request_stats.values())
    token_throughput = total_tokens / total_time
    
    ttfts = []
    e2es = []
    tpots = []
    
    for rid, stats in request_stats.items():
        if stats["end"] and stats["first_token"]:
            ttft = stats["first_token"] - stats["arrival"]
            e2e = stats["end"] - stats["arrival"]
            # TPOT: Time per output token (excluding first token latency)
            # If only 1 token, TPOT is 0 or undefined, let's say e2e
            if stats["tokens"] > 1:
                tpot = (e2e - ttft) / (stats["tokens"] - 1)
            else:
                tpot = 0
            
            ttfts.append(ttft)
            e2es.append(e2e)
            tpots.append(tpot)
            
    avg_ttft = sum(ttfts) / len(ttfts) if ttfts else 0
    avg_e2e = sum(e2es) / len(e2es) if e2es else 0
    avg_tpot = sum(tpots) / len(tpots) if tpots else 0
    
    print(f"Mode: {mode}")
    print(f"Total Time: {total_time:.2f} s")
    print(f"Throughput: {throughput:.2f} req/s")
    print(f"Token Throughput: {token_throughput:.2f} tok/s")
    print(f"Avg TTFT: {avg_ttft*1000:.2f} ms")
    print(f"Avg TPOT: {avg_tpot*1000:.2f} ms")
    print(f"Avg E2E: {avg_e2e:.2f} s")
    
    metrics = {
        "throughput_req_s": throughput,
        "throughput_tok_s": token_throughput,
        "avg_ttft_ms": avg_ttft * 1000,
        "avg_tpot_ms": avg_tpot * 1000,
        "avg_e2e_s": avg_e2e,
        "total_time": total_time
    }
    
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, help="Base model path (or Fused Base for slora mode)")
    parser.add_argument("--mode", type=str, required=True, choices=["baseline", "slora"])
    parser.add_argument("--output_json", type=str, default=None, help="Path to save metrics json")
    
    # LoRA Configs
    parser.add_argument("--hot_lora_path", type=str, required=True)
    parser.add_argument("--hot_lora_name", type=str, default="lora_hot")
    
    # For simplicity, pass cold lora paths as "name:path" strings
    parser.add_argument("--cold_loras", type=str, nargs="+", help="Format: name:path")
    
    args = parser.parse_args()
    
    cold_map = {}
    delta_map = {}
    
    if args.cold_loras:
        for item in args.cold_loras:
            name, path = item.split(":")
            cold_map[name] = path
            delta_map[name] = path # In slora mode, the input path should be the delta path
            
    metrics = asyncio.run(run_benchmark(
        args.trace,
        args.model,
        args.mode,
        args.hot_lora_path,
        args.hot_lora_name,
        delta_map,
        cold_map
    ))
    
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(metrics, f, indent=2)
