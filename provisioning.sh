#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

# --- 1. Install Dependencies and vLLM ---
# Use pip to install vLLM
# We use the /venv/main/ environment provided by the base Vast.ai image
/usr/bin/pip3 install vllm transformers accelerate torch --upgrade

# --- 2. Configuration Variables ---
# The model to download and serve
HF_MODEL="deepseek-ai/deepseek-coder-33b-instruct-v1.5"
# The quantization method (AWQ is often fast for inference)
QUANTIZATION="awq"
# Set tensor parallel size to the number of GPUs you rented
GPU_COUNT=4
# Set VRAM utilization conservative to leave room for the OS and cache
GPU_UTILIZATION=0.95 

echo "Starting vLLM server for $HF_MODEL across $GPU_COUNT GPUs..."

# --- 3. Start the vLLM Server ---
# We use nohup to run the server in the background and & to detach it
# The server is run on port 8000, which you will need to map externally
nohup python3 -m vllm.entrypoints.api_server \
    --model $HF_MODEL \
    --tensor-parallel-size $GPU_COUNT \
    --quantization $QUANTIZATION \
    --dtype bfloat16 \
    --gpu-memory-utilization $GPU_UTILIZATION \
    --host 0.0.0.0 \
    --port 8000 > /workspace/vllm_server.log 2>&1 &

echo "vLLM server started in the background. Check logs at /workspace/vllm_server.log."
echo "Wait a few minutes for the model to download and load onto the 4 GPUs."