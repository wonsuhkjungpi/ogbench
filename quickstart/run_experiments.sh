#!/bin/bash
# OGBench Quickstart Training Script
# This script trains a GCBC agent on the antmaze-large-navigate-v0 environment
set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OGBENCH_DIR="$(dirname "$SCRIPT_DIR")"
IMPLS_DIR="$OGBENCH_DIR/impls"

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ogbench

# Change to impls directory
cd "$IMPLS_DIR"

echo "=========================================="
echo "OGBench Quickstart Training"
echo "=========================================="
echo "Environment: antmaze-large-navigate-v0"
echo "Agent: GCBC"
echo "Training steps: 1,000,000"
echo "=========================================="

# Run training with GCBC on antmaze-large-navigate-v0
# This is one of the simplest experiments to verify the setup
python main.py \
    --env_name=antmaze-large-navigate-v0 \
    --agent=agents/gcbc.py \
    --train_steps=1000000 \
    --eval_interval=100000 \
    --log_interval=5000 \
    --save_interval=1000000 \
    --eval_episodes=20 \
    --video_episodes=1 \
    --run_group=Quickstart \
    --seed=0

echo "=========================================="
echo "Training completed successfully!"
echo "=========================================="
