## Quick Start

### Setup

```bash
git clone ...
cd vLLM-Multi-LoRA-Support-Layer/    # work dir
```

### 1. Deploy

Both deploy scripts accept an optional `max_loras` argument to control how many LoRA adapters can be kept in GPU memory simultaneously.

**Baseline:**
```bash
# Default: max_loras=1 (LoRA swapping enabled)
/bin/bash benchmarks/multi_lora/deploy_base_vllm.sh

# With custom max_loras (e.g., 3 to keep all adapters in memory)
/bin/bash benchmarks/multi_lora/deploy_base_vllm.sh 3
```

**Pre-merge Optimized:**
```bash
# Default: max_loras=1
/bin/bash benchmarks/multi_lora/deploy_multi_lora.sh

# With custom max_loras
/bin/bash benchmarks/multi_lora/deploy_multi_lora.sh 3
```

### 2. Run Benchmark

```bash
# For baseline vLLM (default max_loras=1)
/bin/bash benchmarks/multi_lora/run_benchmark.sh baseline

# For baseline with max_loras=3
/bin/bash benchmarks/multi_lora/run_benchmark.sh baseline 3

# For pre-merge optimized (default max_loras=1)
/bin/bash benchmarks/multi_lora/run_benchmark.sh multi_lora

# For pre-merge optimized with max_loras=2
/bin/bash benchmarks/multi_lora/run_benchmark.sh multi_lora 3
```
