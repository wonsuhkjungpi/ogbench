# OGBench Project Context

## Project Overview

**OGBench** is a comprehensive benchmark for offline goal-conditioned reinforcement learning (RL), offline unsupervised RL, and standard offline RL.

**Key Paper**: "OGBench: Benchmarking Offline Goal-Conditioned RL" (ICLR 2025)
- arXiv: https://arxiv.org/abs/2410.20092
- Project page: https://seohong.me/projects/ogbench/

**Version**: 1.2.0 (as of 2025-10-20)

---

## Key Features

### Environments (8 types)
**Locomotion:**
- PointMaze, AntMaze, HumanoidMaze, AntSoccer

**Manipulation:**
- Cube, Scene, Puzzle

**Drawing:**
- Powderworld

### Dataset Scale
- **85 datasets** for offline goal-conditioned RL
- **410 tasks** for standard offline RL
- Support for both pixel-based and state-based observations

### Reference Implementations (JAX-based)
Six offline goal-conditioned RL algorithms:
1. **GCBC** - Goal-Conditioned Behavioral Cloning
2. **GCIVL** - Goal-Conditioned Implicit V-Learning
3. **GCIQL** - Goal-Conditioned Implicit Q-Learning
4. **QRL** - Quasimetric Reinforcement Learning
5. **CRL** - Contrastive Reinforcement Learning
6. **HIQL** - Hierarchical Implicit Q-Learning

---

## Installation

### Basic Installation (environments only)
```bash
pip install ogbench
```

Dependencies: Python 3.8+, mujoco >= 3.1.6, dm_control >= 1.0.20, gymnasium

### Training Installation (reference implementations)
```bash
cd ogbench/impls
pip install -r requirements.txt
```

Additional dependencies: jax[cuda12] >= 0.4.26, flax >= 0.8.4, distrax >= 0.1.5, ml_collections, matplotlib, moviepy, wandb

### Development Installation (editable mode)
```bash
cd ogbench
pip install -e ".[train]"
```

---

## Directory Structure

```
ogbench/
├── ogbench/                  # Main package
│   ├── __init__.py          # API entry point
│   ├── utils.py             # Dataset utilities
│   ├── relabel_utils.py     # Goal relabeling utilities
│   ├── locomaze/            # Locomotion maze environments
│   ├── manipspace/          # Manipulation environments
│   ├── online_locomotion/   # Online locomotion envs
│   └── powderworld/         # Drawing environment
├── impls/                    # Reference implementations
│   ├── main.py              # Training entry point
│   ├── requirements.txt     # Training dependencies
│   ├── hyperparameters.sh   # Full benchmark commands
│   ├── agents/              # Algorithm implementations
│   │   ├── gcbc.py
│   │   ├── gcivl.py
│   │   ├── gciql.py
│   │   ├── qrl.py
│   │   ├── crl.py
│   │   ├── hiql.py
│   │   └── sac.py
│   └── utils/               # Training utilities
├── data_gen_scripts/        # Dataset generation scripts
├── assets/                  # Images and assets
├── pyproject.toml           # Package configuration
└── README.md                # Full documentation
```

---

## Quick Start

### Goal-Conditioned RL Usage

```python
import ogbench

# Create environment and load datasets (auto-downloaded)
dataset_name = 'antmaze-large-navigate-v0'
env, train_dataset, val_dataset = ogbench.make_env_and_datasets(dataset_name)

# Evaluate on tasks
for task_id in [1, 2, 3, 4, 5]:
    ob, info = env.reset(options=dict(task_id=task_id, render_goal=True))
    goal = info['goal']

    done = False
    while not done:
        action = your_agent.get_action(ob, goal)
        ob, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    success = info['success']  # 0 or 1
```

### Standard Offline RL Usage

```python
import ogbench

# Single-task environment (no goal conditioning)
dataset_name = 'antmaze-large-navigate-singletask-v0'
env, train_dataset, val_dataset = ogbench.make_env_and_datasets(dataset_name)

# Dataset contains 'masks' field for Q-learning (not 'terminals')
# masks = 0 when task is complete, 1 otherwise
```

### Training Reference Implementations

```bash
cd impls

# GCBC on AntMaze
python main.py --env_name=antmaze-large-navigate-v0 --agent=agents/gcbc.py

# GCIQL with custom alpha
python main.py --env_name=antmaze-large-navigate-v0 --agent=agents/gciql.py --agent.alpha=0.3

# HIQL
python main.py --env_name=antmaze-large-navigate-v0 --agent=agents/hiql.py \
    --agent.high_alpha=3.0 --agent.low_alpha=3.0
```

---

## Important Hyperparameters

### Universal
- `--train_steps`: Training steps (default: 1,000,000)
- `--eval_interval`: Evaluation frequency (default: 100,000)
- `--seed`: Random seed
- `--agent.alpha`: Temperature/BC coefficient (CRITICAL - tune per environment)

### Algorithm-Specific

**GCIQL/CRL/QRL:**
- `--agent.actor_p_trajgoal`: Probability of trajectory goals (default: 1.0)
- `--agent.actor_p_randomgoal`: Probability of random goals (0.0-0.5 for stitching datasets)
- Policy extraction: AWR vs DDPG+BC (DDPG+BC generally better but more sensitive)

**QRL:**
- `--agent.quasimetric`: MRN or IQE (IQE default, better but slower)

**CRL:**
- `--agent.actor_log_q`: Use log Q values (default: True, important for locomotion)

**HIQL:**
- `--agent.low_actor_rep_grad`: Gradient flow to representation (True for pixel envs)

**Pixel-based:**
- `--agent.encoder=impala_small`
- `--agent.discrete=True` (for Powderworld)

**Powderworld:**
- `--eval_temperature=0.3`

---

## Dataset Information

### Dataset Format
```python
train_dataset = {
    'observations': array,      # State observations
    'actions': array,           # Actions taken
    'next_observations': array, # Next states (if not compact)
    'terminals': array,         # Trajectory boundaries
    'masks': array,             # For Q-learning (singletask only)
    'valids': array,            # Valid transitions (if compact)
}
```

### Dataset Storage
- Default location: `~/.ogbench/data`
- Auto-downloaded on first use
- Configure via `dataset_dir` parameter

### Compact Datasets
Use `compact_dataset=True` to omit `next_observations` field (saves memory).

---

## Evaluation

### Standard Evaluation
- 5 evaluation tasks per environment (task_id 1-5)
- 20 episodes per task by default
- Success metric: `info['success']` (0 or 1)

### Singletask Default Tasks
| Environment | Default Task |
|:-----------:|:------------:|
| pointmaze-* | task1 |
| antmaze-* | task1 |
| humanoidmaze-* | task1 |
| antsoccer-* | task4 |
| cube-* | task2 |
| scene-* | task2 |
| puzzle-{3x3, 4x4} | task4 |
| puzzle-{4x5, 4x6} | task2 |

---

## Common Workflows

### Train GCBC Agent
```bash
cd impls
python main.py \
    --env_name=antmaze-large-navigate-v0 \
    --agent=agents/gcbc.py \
    --train_steps=1000000 \
    --eval_interval=100000 \
    --seed=0
```

### Quick Validation Run
```bash
python main.py \
    --env_name=antmaze-large-navigate-v0 \
    --agent=agents/gcbc.py \
    --train_steps=1000 \
    --eval_interval=1000 \
    --log_interval=100
```

### Generate Dataset
```bash
cd data_gen_scripts
# Download expert policies first
wget https://rail.eecs.berkeley.edu/datasets/ogbench/experts.tar.gz
tar xf experts.tar.gz && rm experts.tar.gz

export PYTHONPATH="../impls:${PYTHONPATH}"
python generate_locomaze.py --env_name=antmaze-large-v0 --save_path=data/antmaze-large-navigate-v0.npz
```

---

## Runtime Estimates

| Environment Type | Training Time (A5000 GPU) |
|:---------------:|:-------------------------:|
| State-based | 2-5 hours |
| Pixel-based | 5-12 hours |

**Memory:** Large pixel datasets (e.g., visual-puzzle-4x6-play-v0) may require up to 120GB RAM.

---

## Tips for Claude

### When running experiments:
- Always specify `--agent` to select algorithm
- Tune `--agent.alpha` for new environments (most critical hyperparameter)
- Use `--train_steps=1000` for quick validation
- Check wandb for training metrics

### When debugging:
- Verify JAX/CUDA installation: `python -c "import jax; print(jax.devices())"`
- Check dataset download: `ls ~/.ogbench/data`
- Use `env_only=True` to test environment without dataset download

### When modifying code:
- Agents are self-contained in `impls/agents/`
- Dataset utilities in `impls/utils/datasets.py`
- Environment wrappers in `impls/utils/env_utils.py`

---

## Caveats

1. **Do NOT use `gymnasium.make`** - Always use `ogbench.make_env_and_datasets`

2. **Singletask rewards** (v1.2.0+): Computed based on current state `s` (not next state `s'`)
   - Use `success_timing='post'` to restore old behavior if needed

3. **Dataset fields**:
   - `terminals`: Trajectory boundaries
   - `masks`: Bellman backup validity (for Q-learning)

---

## Citation

```bibtex
@inproceedings{ogbench_park2025,
  title={OGBench: Benchmarking Offline Goal-Conditioned RL},
  author={Park, Seohong and Frans, Kevin and Eysenbach, Benjamin and Levine, Sergey},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025},
}
```

---

## Contact

- Repository: https://github.com/seohongpark/ogbench
- Maintainer: Seohong Park (seohong@berkeley.edu)
