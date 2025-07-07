#!/bin/bash

# Ensure necessary directories exist
mkdir -p /home/user/.triton/autotune
mkdir -p /home/user/.cache/matplotlib
mkdir -p /home/user/.cache/huggingface

# Set proper permissions
chown -R user:user /home/user/.triton /home/user/.cache 2>/dev/null || true

# Set environment variables
export MPLBACKEND=Agg
export TRITON_CACHE_DIR=/home/user/.triton
export HF_HOME=/home/user/.cache/huggingface

# Start the main application
exec python3 -u /app/rp_handler.py --model-dir="${WORKER_MODEL_DIR}" 