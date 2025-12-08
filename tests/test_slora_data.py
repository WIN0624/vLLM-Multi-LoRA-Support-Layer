import os
import json
import shutil
import sys
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tools.slora.generate_data import generate_trace

def test_trace_generation():
    print("Running Trace Generation Verification...")
    output_path = "tests/temp_trace.jsonl"
    
    num_reqs = 1000
    hot_ratio = 0.98
    hot_name = "hot_lora"
    cold_names = ["cold_1", "cold_2", "cold_3"]
    
    generate_trace(
        output_path=output_path,
        num_requests=num_reqs,
        hot_adapter_name=hot_name,
        cold_adapter_names=cold_names,
        hot_ratio=hot_ratio
    )
    
    # Verify
    with open(output_path, 'r') as f:
        lines = f.readlines()
        
    assert len(lines) == num_reqs
    
    hot_count = 0
    cold_counts = {name: 0 for name in cold_names}
    
    for line in lines:
        req = json.loads(line)
        if req['lora_name'] == hot_name:
            hot_count += 1
        elif req['lora_name'] in cold_counts:
            cold_counts[req['lora_name']] += 1
        else:
            print(f"FAILURE: Unknown lora name {req['lora_name']}")
            sys.exit(1)
            
    actual_hot_ratio = hot_count / num_reqs
    print(f"Target Hot Ratio: {hot_ratio}, Actual: {actual_hot_ratio}")
    
    # Allow small floating point tolerance if we were doing probabilistic generation,
    # but our script is deterministic in counts, so it should be exact or off by 1 due to rounding.
    assert abs(actual_hot_ratio - hot_ratio) < 0.01
    
    print("SUCCESS: Trace distribution verified.")
    
    if os.path.exists(output_path):
        os.remove(output_path)

if __name__ == "__main__":
    test_trace_generation()
