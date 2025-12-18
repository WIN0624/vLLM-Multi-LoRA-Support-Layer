#!/bin/bash
# Multi-LoRA Benchmark Script: Test different hot_ratio values
# This script runs benchmark_lora_popularity.py with varying hot_ratio values
# to analyze how LoRA cache hit rate affects performance.

set -e  # Exit on error

# ==================== Configuration ====================
BASE_MODEL="baffo32/decapoda-research-llama-7B-hf"
TOKENIZER="huggyllama/llama-7b"
HOT_ADAPTER="apaca-lora"
COLD_ADAPTERS="guanaco-lora wizard-lora"

# Test parameters
NUM_REQUESTS=200
PROMPT_LEN=400
OUTPUT_LEN=128
REQUEST_RATE="inf"  # Use "inf" for max throughput, or set a specific rate like "10"

# Server configuration
HOST="127.0.0.1"
PORT=8000

# Hot ratio values to test (from 0.0 to 1.0)
HOT_RATIOS=(0.0 0.2 0.4 0.5 0.6 0.8 1.0)

# Result directory
RESULT_DIR="benchmarks/multi_lora/test_results"

# ==================== Script ====================

# Create result directory if not exists
mkdir -p "${RESULT_DIR}"

echo "=============================================="
echo "  Multi-LoRA Hot Ratio Benchmark"
echo "=============================================="
echo "Base Model:     ${BASE_MODEL}"
echo "Hot Adapter:    ${HOT_ADAPTER}"
echo "Cold Adapters:  ${COLD_ADAPTERS}"
echo "Num Requests:   ${NUM_REQUESTS}"
echo "Prompt Length:  ${PROMPT_LEN}"
echo "Output Length:  ${OUTPUT_LEN}"
echo "Request Rate:   ${REQUEST_RATE}"
echo "Hot Ratios:     ${HOT_RATIOS[*]}"
echo "Result Dir:     ${RESULT_DIR}"
echo "=============================================="
echo ""

# Run benchmark for each hot_ratio
for HOT_RATIO in "${HOT_RATIOS[@]}"; do
    echo "----------------------------------------------"
    echo "Running benchmark with hot_ratio = ${HOT_RATIO}"
    echo "----------------------------------------------"

    RESULT_FILENAME="baseline_prompt${PROMPT_LEN}_hot_ratio${HOT_RATIO}_out${OUTPUT_LEN}.json"
        
    python benchmarks/multi_lora/benchmark_lora_popularity.py \
        --base-model "${BASE_MODEL}" \
        --tokenizer "${TOKENIZER}" \
        --hot-adapter "${HOT_ADAPTER}" \
        --cold-adapters ${COLD_ADAPTERS} \
        --hot-ratio "${HOT_RATIO}" \
        --num-requests "${NUM_REQUESTS}" \
        --prompt-len "${PROMPT_LEN}" \
        --output-len "${OUTPUT_LEN}" \
        --request-rate "${REQUEST_RATE}" \
        --host "${HOST}" \
        --port "${PORT}" \
        --trust-remote-code \
        --save-results \
        --result-dir "${RESULT_DIR}" \
        --result-filename "${RESULT_FILENAME}"
    
    echo ""
    echo "Result saved to: ${RESULT_DIR}/${RESULT_FILENAME}"
    echo ""
done

echo "=============================================="
echo "  All benchmarks completed!"
echo "  Results saved in: ${RESULT_DIR}"
echo "=============================================="