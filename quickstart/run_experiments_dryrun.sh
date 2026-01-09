#!/bin/bash
# OGBench Quickstart Dryrun Script
# This script runs a quick validation with minimal training steps
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
echo "OGBench Quickstart DRYRUN"
echo "=========================================="
echo "Environment: antmaze-large-navigate-v0"
echo "Agent: GCBC"
echo "Training steps: 1,000 (reduced for quick validation)"
echo "=========================================="

# Run training with minimal steps for quick validation
# Wandb logging is set to offline mode to avoid permission issues
python main.py \
    --env_name=antmaze-large-navigate-v0 \
    --agent=agents/gcbc.py \
    --train_steps=1000 \
    --eval_interval=1000 \
    --log_interval=100 \
    --save_interval=10000 \
    --eval_episodes=2 \
    --eval_tasks=2 \
    --video_episodes=0 \
    --run_group=DryrunTest \
    --seed=0

echo "=========================================="
echo "Dryrun completed successfully!"
echo "Environment is properly configured."
echo "=========================================="
