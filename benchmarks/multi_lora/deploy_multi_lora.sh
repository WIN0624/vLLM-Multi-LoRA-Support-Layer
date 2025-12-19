#!/bin/bash
# Calculate the merged and delta LoRA weights, and deploy the models.
#
# Usage:
#   /bin/bash deploy_multi_lora.sh [max_loras]
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

OUTPUT_DIR="models"

BASE_MODEL_NAME=$(basename ${BASE_MODEL})
HOT_NAME=$(basename ${HOT_LORA}) 
COLD1_NAME=$(basename ${COLD_LORA1})
COLD2_NAME=$(basename ${COLD_LORA2})

# === 1. Generate the merged and delta LoRA weights ===

echo "=============================================="
echo "  Deploy Multi-LoRA (Pre-merge Optimized)"
echo "=============================================="
echo "Base Model:     ${BASE_MODEL}"
echo "Hot Adapter:    ${HOT_LORA}"
echo "Cold Adapters:  ${COLD_LORA1} ${COLD_LORA2}"
echo "Max LoRAs:      ${MAX_LORAS}"
echo "Output Dir:     ${OUTPUT_DIR}"
echo "=============================================="
echo ""

python tools/pre_merge_hot_lora/generate_weights.py \
        --base-model ${BASE_MODEL} \
        --hot-adapter ${HOT_LORA} \
        --cold-adapters ${COLD_LORA1} ${COLD_LORA2} \
        --output-dir ${OUTPUT_DIR}

# === 2. Deploy the models ===
REAL_MERGED_PATH="${OUTPUT_DIR}/${BASE_MODEL_NAME}-${HOT_NAME}-fused"
REAL_DELTA1_PATH="${OUTPUT_DIR}/${COLD1_NAME}-delta"
REAL_DELTA2_PATH="${OUTPUT_DIR}/${COLD2_NAME}-delta"

vllm serve ${REAL_MERGED_PATH} \
    --served-model-name hot-lora \
    --enable-lora \
    --lora-modules cold-lora1=${REAL_DELTA1_PATH} \
                   cold-lora2=${REAL_DELTA2_PATH} \
    --tokenizer huggyllama/llama-7b \
    --gpu-memory-utilization 0.8 \
    --max-loras ${MAX_LORAS} \
    --max-lora-rank 64
