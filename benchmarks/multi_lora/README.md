# Multi-LoRA Benchmark

Benchmark for **Pre-merge Hot LoRA** optimization strategy.

## Quick Start

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

## Parameters

| Script | Argument | Description | Default |
|--------|----------|-------------|---------|
| `deploy_*.sh` | `max_loras` | Max LoRA adapters in GPU memory | 1 |
| `run_benchmark.sh` | `mode` | `baseline` or `multi_lora` | `baseline` |
| `run_benchmark.sh` | `max_loras` | Max LoRA adapters (for filename) | 1 |

> **Note**: Make sure `max_loras` in deploy and benchmark scripts match!
