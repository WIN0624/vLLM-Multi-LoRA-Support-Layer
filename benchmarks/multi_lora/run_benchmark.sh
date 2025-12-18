#!/bin/bash
# Multi-LoRA Benchmark Script: Test different hot_ratio values
# Usage: ./run_benchmark.sh [baseline|multi_lora] [max_loras]

set -e  # Exit on error

# ==================== Parse Arguments ====================
MODE="${1:-baseline}"  # Default to baseline if not specified
MAX_LORA="${2:-1}"     # Default max_loras to 1

if [[ "$MODE" != "baseline" && "$MODE" != "multi_lora" ]]; then
    echo "Usage: $0 [baseline|multi_lora] [max_loras]"
    echo "  baseline   - Run benchmark with original vLLM"
    echo "  multi_lora - Run benchmark with pre-merge optimized model"
    echo "  max_loras  - Max LoRA adapters in GPU memory (default: 1)"
    exit 1
fi

PREFIX="$MODE"

# ==================== Configuration ====================
BASE_MODEL="baffo32/decapoda-research-llama-7B-hf"
TOKENIZER="huggyllama/llama-7b"
HOT_ADAPTER="hot-lora"
COLD_ADAPTERS="cold-lora1 cold-lora2"

# Test parameters
NUM_REQUESTS=200
PROMPT_LEN=400
OUTPUT_LEN=128
REQUEST_RATE="10"  # Use "inf" for max throughput, or set a specific rate like "10"
# Server configuration
HOST="127.0.0.1"
PORT=8000

# Hot ratio values to test (from 0.0 to 1.0)
HOT_RATIOS=(0.0 0.2 0.4 0.5 0.6 0.8 1.0)

# Result directory
RESULT_DIR="benchmarks/multi_lora/test_results" 
LOG_DIR="benchmarks/multi_lora/test_logs"

# ==================== Script ====================

# Create result directory if not exists
mkdir -p "${RESULT_DIR}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/${PREFIX}_max_lora${MAX_LORA}.log"

# Function to log and print
log() {
    echo "$@" | tee -a "${LOG_FILE}"
}

# Function to run command, show full output in terminal, but filter tqdm in log
run_and_log() {
    # Use process substitution: terminal sees everything, log filters out tqdm progress bars
    "$@" 2>&1 | while IFS= read -r line; do
        echo "$line"
        # Only write to log if not a tqdm progress bar (contains %| or it/s])
        if [[ ! "$line" =~ (%\||\[.*it/s\]) ]]; then
            echo "$line" >> "${LOG_FILE}"
        fi
    done
}

log "=============================================="
log "  Multi-LoRA Hot Ratio Benchmark"
log "=============================================="
log "Mode:           ${MODE}"
log "Max LoRAs:      ${MAX_LORA}"
log "Base Model:     ${BASE_MODEL}"
log "Hot Adapter:    ${HOT_ADAPTER}"
log "Cold Adapters:  ${COLD_ADAPTERS}"
log "Num Requests:   ${NUM_REQUESTS}"
log "Prompt Length:  ${PROMPT_LEN}"
log "Output Length:  ${OUTPUT_LEN}"
log "Request Rate:   ${REQUEST_RATE}"
log "Hot Ratios:     ${HOT_RATIOS[*]}"
log "Result Dir:     ${RESULT_DIR}"
log "Log File:       ${LOG_FILE}"
log "=============================================="
log ""

# Run benchmark for each hot_ratio
for HOT_RATIO in "${HOT_RATIOS[@]}"; do
    log "----------------------------------------------"
    log "Running benchmark with hot_ratio = ${HOT_RATIO}"
    log "----------------------------------------------"

    RESULT_FILENAME="${PREFIX}_prompt${PROMPT_LEN}_hot_ratio${HOT_RATIO}_out${OUTPUT_LEN}_max_lora${MAX_LORA}.json"
        
    run_and_log python benchmarks/multi_lora/benchmark_lora_popularity.py \
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
        --ignore-eos \
        --save-results \
        --result-dir "${RESULT_DIR}" \
        --result-filename "${RESULT_FILENAME}"
    
    log ""
    log "Result saved to: ${RESULT_DIR}/${RESULT_FILENAME}"
    log ""
done

log "=============================================="
log "  All benchmarks completed!"
log "  Results saved in: ${RESULT_DIR}"
log "  Log saved to: ${LOG_FILE}"
log "=============================================="
