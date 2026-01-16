# OGBench Integration for RobOMimic

This document describes the workflow for training OGBench goal-conditioned RL algorithms on RobOMimic datasets and evaluating them on the FAIL-Detect benchmark.

## Overview

OGBench provides JAX-based implementations of goal-conditioned RL algorithms (GCBC, GCIVL, CRL, etc.) that can be trained on RobOMimic manipulation datasets and evaluated using the FAIL-Detect benchmark infrastructure.

**Workflow Summary:**
```
HDF5 (RobOMimic) → NPZ (OGBench) → Train (JAX) → Evaluate (PyTorch)
```

## Prerequisites

### Two Conda Environments Required

1. **`faildetect`** (PyTorch-based)
   - Dataset conversion
   - Policy wrapper integration
   - Benchmark evaluation
   ```bash
   mamba activate robomimic
   ```

2. **`ogbench`** (JAX-based)
   - Training GCRL algorithms
   ```bash
   mamba activate ogbench
   ```

### Install OGBench Dependencies

```bash
cd ogbench/impls
pip install -r requirements.txt
```

## Workflow

### Step 1: Dataset Conversion (HDF5 → NPZ)

Convert RobOMimic HDF5 datasets to OGBench NPZ format.

**Environment:** `faildetect`

```python
from diffusion_policy.dataset.robomimic_to_ogbench_adapter import convert_robomimic_hdf5_to_ogbench

train_path, val_path = convert_robomimic_hdf5_to_ogbench(
    hdf5_path='data/robomimic/datasets/square/ph/image_abs.hdf5',
    output_path='data/ogbench/square_train.npz',
    task_name='square',
    val_ratio=0.1
)
```

**NPZ Format:**
```python
{
    'observations': (N, D_obs),   # State observations
    'actions': (N, D_a),          # Actions
    'terminals': (N,),            # Episode boundaries (1 at episode end)
    'next_observations': (N, D_obs),  # Optional
}
```

### Step 2: Training

Train goal-conditioned RL algorithms on the converted dataset.

**Environment:** `ogbench`

```bash
cd ogbench/impls

python train_robomimic.py \
    --train_path=/path/to/square_train.npz \
    --val_path=/path/to/square_val.npz \
    --agent=agents/gcivl.py \
    --agent.alpha=10.0 \
    --train_steps=500000 \
    --save_dir=data/outputs/ogbench/train/gcivl_square_500k
```

**Key Flags:**

| Flag | Description | Default |
|------|-------------|---------|
| `--train_path` | Path to training NPZ file | Required |
| `--val_path` | Path to validation NPZ file | Optional |
| `--agent` | Agent config file | `agents/gcivl.py` |
| `--agent.alpha` | Temperature parameter (critical) | Algorithm-dependent |
| `--train_steps` | Total training steps | 1,000,000 |
| `--save_dir` | Output directory | Required |
| `--use_wandb` | Enable W&B logging | False |

**Output Structure:**
```
save_dir/
├── params_100000.pkl    # Checkpoint at step 100k
├── params_200000.pkl
├── ...
├── params_500000.pkl    # Final checkpoint
└── metrics.csv          # Training metrics
```

### Step 3: Evaluation on Benchmark

Evaluate trained agents on the FAIL-Detect benchmark.

**Environment:** `faildetect`

```bash
python ogbench/impls/eval_robomimic_benchmark.py \
    --checkpoint_path=data/outputs/ogbench/train/gcivl_square_500k/params_500000.pkl \
    --agent_config=ogbench/impls/agents/gcivl.py \
    --benchmark_path=experiments/benchmark/square/final1000/testcases.pkl \
    --task_name=square \
    --output_dir=experiments/benchmark_evaluation/square/ogbench_gcivl
```

### Step 4: Policy Wrapper (Optional)

Use `OGBenchPolicyWrapper` to integrate JAX checkpoints with PyTorch-based evaluation.

```python
from diffusion_policy.policy.ogbench_policy_wrapper import OGBenchPolicyWrapper

# Load OGBench policy
policy = OGBenchPolicyWrapper(
    checkpoint_path='data/outputs/ogbench/train/gcivl_square_500k/params_500000.pkl',
    agent_config_path='ogbench/impls/agents/gcivl.py',
    obs_dim=23,
    action_dim=7
)

# Predict action given observation and goal
action = policy.predict_action_with_obs_feature_and_goal_feature(
    obs_feature=current_obs,    # (B, obs_dim)
    goal_feature=goal_obs       # (B, obs_dim)
)
```

**Switching Policy (BC + OGBench):**
```python
from diffusion_policy.policy.ogbench_policy_wrapper import SwitchingBCOGBenchPolicy

switching_policy = SwitchingBCOGBenchPolicy(
    bc_policy=bc_policy,
    ogbench_policy=ogbench_policy
)
```

## Available Algorithms

| Algorithm | Config File | Key Hyperparameter | Description |
|-----------|-------------|-------------------|-------------|
| GCBC | `agents/gcbc.py` | - | Goal-conditioned behavioral cloning |
| GCIVL | `agents/gcivl.py` | `alpha=10.0` | Goal-conditioned implicit V-learning |
| GCIQL | `agents/gciql.py` | `alpha=0.3` | Goal-conditioned implicit Q-learning |
| CRL | `agents/crl.py` | `alpha=0.1` | Contrastive reinforcement learning |
| QRL | `agents/qrl.py` | `alpha=0.003` | Quasimetric reinforcement learning |
| HIQL | `agents/hiql.py` | `high_alpha=3.0` | Hierarchical implicit Q-learning |

### Hyperparameter Tuning

The `alpha` parameter is critical and varies significantly across algorithms:

```bash
# GCIVL (higher alpha)
python train_robomimic.py --agent=agents/gcivl.py --agent.alpha=10.0

# CRL (lower alpha, enable actor_log_q for locomotion)
python train_robomimic.py --agent=agents/crl.py --agent.alpha=0.1 --agent.actor_log_q=True

# QRL (very low alpha)
python train_robomimic.py --agent=agents/qrl.py --agent.alpha=0.003
```

## File Locations

| Purpose | Location |
|---------|----------|
| Training script | `ogbench/impls/train_robomimic.py` |
| Dataset loader | `ogbench/impls/dataset_robomimic.py` |
| Benchmark evaluation | `ogbench/impls/eval_robomimic_benchmark.py` |
| Agent configs | `ogbench/impls/agents/*.py` |
| Policy wrapper | `diffusion_policy/policy/ogbench_policy_wrapper.py` |
| Dataset converter | `diffusion_policy/dataset/robomimic_to_ogbench_adapter.py` |

## Benchmark Evaluation Groups

The benchmark evaluates across 7 OOD (out-of-distribution) groups:

| Group | Start State | Goal State |
|-------|-------------|------------|
| `in_dist_start_in_dist_goal` | In-distribution | In-distribution |
| `in_dist_start_in_dist_goal_pert` | ID (perturbed actions) | In-distribution |
| `in_dist_start_ood_goal` | In-distribution | Out-of-distribution |
| `near_ood_start_in_dist_goal` | Near-OOD | In-distribution |
| `near_ood_start_ood_goal` | Near-OOD | Out-of-distribution |
| `ood_start_in_dist_goal` | Out-of-distribution | In-distribution |
| `ood_start_ood_goal` | Out-of-distribution | Out-of-distribution |

## Metrics

| Metric | Description |
|--------|-------------|
| `success_rate` | Task completion rate |
| `state_l2_final` | L2 distance to goal at episode end |
| `state_l2_best` | Best L2 distance achieved during episode |
| `latent_sim_final` | Cosine similarity in latent space |
| `delta_*_vs_bc` | Improvement over BC baseline |

## Example: Full Training Pipeline

```bash
# 1. Activate faildetect environment for dataset conversion
mamba activate robomimic

# 2. Convert dataset (run in Python or as script)
python -c "
from diffusion_policy.dataset.robomimic_to_ogbench_adapter import convert_robomimic_hdf5_to_ogbench
convert_robomimic_hdf5_to_ogbench(
    'data/robomimic/datasets/square/ph/image_abs.hdf5',
    'data/ogbench/square.npz',
    'square'
)
"

# 3. Activate ogbench environment for training
mamba activate ogbench

# 4. Train GCIVL
cd ogbench/impls
python train_robomimic.py \
    --train_path=../../data/ogbench/square_train.npz \
    --val_path=../../data/ogbench/square_val.npz \
    --agent=agents/gcivl.py \
    --agent.alpha=10.0 \
    --train_steps=500000 \
    --save_dir=../../data/outputs/ogbench/train/gcivl_square_500k

# 5. Activate faildetect environment for evaluation
mamba activate robomimic

# 6. Evaluate on benchmark
python ogbench/impls/eval_robomimic_benchmark.py \
    --checkpoint_path=data/outputs/ogbench/train/gcivl_square_500k/params_500000.pkl \
    --agent_config=ogbench/impls/agents/gcivl.py \
    --benchmark_path=experiments/benchmark/square/final1000/testcases.pkl \
    --task_name=square
```

## Current Results

| Agent | Success Rate | Mean Distance Improvement |
|-------|-------------|--------------------------|
| CRL (500k) | 0.14% | -0.67 |
| GCIVL (500k) | 0.29% | -0.74 |

## Troubleshooting

### JAX/CUDA Issues
Ensure JAX is installed with CUDA support:
```bash
pip install --upgrade "jax[cuda12]"
```

### Memory Issues
Reduce batch size if OOM:
```bash
python train_robomimic.py --agent.batch_size=128 ...
```

### Policy Wrapper Import Errors
The policy wrapper lazily imports JAX. Ensure the `ogbench` conda environment is accessible or JAX is installed in the current environment.

## References

- [OGBench Paper](https://arxiv.org/abs/2410.20092)
- [OGBench GitHub](https://github.com/seohongpark/ogbench)
- [FAIL-Detect Paper](https://arxiv.org/abs/2503.08558)
