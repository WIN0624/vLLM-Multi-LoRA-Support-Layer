#!/bin/bash
# Deploy the base model with the hot LoRA adapter merged into it.
#
# Usage:
#   /bin/bash deploy_base_vllm.sh [max_loras]
#
# Arguments:
#   max_loras - Maximum number of LoRA adapters to keep in GPU memory (default: 1)

set -e  # Exit on error

BASE_MODEL="baffo32/decapoda-research-llama-7B-hf"
HOT_LORA="tloen/alpaca-lora-7b"
COLD_LORA1="plncmm/guanaco-lora-7b"
COLD_LORA2="winddude/wizardLM-LlaMA-LoRA-7B"

# Parse command line argument
MAX_LORAS=${1:-1}

echo "=============================================="
echo "  Deploy Baseline vLLM with LoRA"
echo "=============================================="
echo "Base Model:     ${BASE_MODEL}"
echo "Hot Adapter:    ${HOT_LORA}"
echo "Cold Adapters:  ${COLD_LORA1} ${COLD_LORA2}"
echo "Max LoRAs:      ${MAX_LORAS}"
echo "=============================================="
echo ""

vllm serve ${BASE_MODEL} \
    --enable-lora \
    --lora-modules hot-lora=${HOT_LORA} \
                   cold-lora1=${COLD_LORA1} \
                   cold-lora2=${COLD_LORA2} \
    --tokenizer huggyllama/llama-7b \
    --gpu-memory-utilization 0.8 \
    --max-loras ${MAX_LORAS} \
    --max-lora-rank 64
