"""
Training script for OGBench algorithms on robomimic datasets.

This script is adapted from main.py to work with pre-converted robomimic NPZ datasets.
It skips environment evaluation (since robomimic envs are not OGBench-compatible).

Usage:
    python train_robomimic.py \
        --train_path=/path/to/square_train.npz \
        --val_path=/path/to/square_train-val.npz \
        --agent=agents/gcivl.py \
        --agent.alpha=10.0 \
        --train_steps=500000 \
        --save_dir=/path/to/output
"""

import json
import os
import random
import time

import jax
import numpy as np
import tqdm
import wandb
from absl import app, flags
from agents import agents
from ml_collections import config_flags
from dataset_robomimic import load_npz_as_dataset
from utils.datasets import GCDataset, HGCDataset
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, setup_wandb

FLAGS = flags.FLAGS

# Dataset flags
flags.DEFINE_string('train_path', None, 'Path to training NPZ file.')
flags.DEFINE_string('val_path', None, 'Path to validation NPZ file (optional).')
flags.DEFINE_string('dataset_name', 'robomimic-square', 'Dataset name for logging.')

# Training flags
flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

flags.DEFINE_integer('train_steps', 1000000, 'Number of training steps.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('save_interval', 100000, 'Saving interval.')

# Disable wandb by default for testing
flags.DEFINE_bool('use_wandb', False, 'Whether to use wandb for logging.')

config_flags.DEFINE_config_file('agent', 'agents/gcivl.py', lock_config=False)


def main(_):
    # Validate required flags
    if FLAGS.train_path is None:
        raise ValueError("--train_path is required")

    # Set up logger
    exp_name = get_exp_name(FLAGS.seed)

    if FLAGS.use_wandb:
        setup_wandb(project='OGBench-Robomimic', group=FLAGS.run_group, name=exp_name)
        FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name)
    else:
        # Create save directory without wandb
        FLAGS.save_dir = os.path.join(FLAGS.save_dir, FLAGS.run_group, exp_name)

    os.makedirs(FLAGS.save_dir, exist_ok=True)

    # Save flags
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f, indent=2)

    print(f"Save directory: {FLAGS.save_dir}")
    print(f"Training on: {FLAGS.train_path}")

    # Load datasets
    config = FLAGS.agent
    print(f"Agent: {config['agent_name']}")
    print(f"Batch size: {config['batch_size']}")

    train_dataset = load_npz_as_dataset(FLAGS.train_path)
    val_dataset = load_npz_as_dataset(FLAGS.val_path) if FLAGS.val_path else None

    print(f"Training dataset size: {train_dataset.size}")
    if val_dataset:
        print(f"Validation dataset size: {val_dataset.size}")

    # Wrap with GCDataset
    dataset_class = {
        'GCDataset': GCDataset,
        'HGCDataset': HGCDataset,
    }[config['dataset_class']]

    train_dataset = dataset_class(train_dataset, config)
    if val_dataset is not None:
        val_dataset = dataset_class(val_dataset, config)

    # Initialize agent
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    example_batch = train_dataset.sample(1)
    if config['discrete']:
        raise ValueError("Discrete actions not supported for robomimic")

    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    # Restore agent if specified
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    # Training loop
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    first_time = time.time()
    last_time = time.time()

    print(f"\nStarting training for {FLAGS.train_steps} steps...")

    for i in tqdm.tqdm(range(1, FLAGS.train_steps + 1), smoothing=0.1, dynamic_ncols=True):
        # Update agent
        batch = train_dataset.sample(config['batch_size'])
        agent, update_info = agent.update(batch)

        # Log metrics
        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}

            if val_dataset is not None:
                val_batch = val_dataset.sample(config['batch_size'])
                _, val_info = agent.total_loss(val_batch, grad_params=None)
                train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})

            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            last_time = time.time()

            if FLAGS.use_wandb:
                wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

            # Print key metrics
            actor_loss = train_metrics.get('training/actor/actor_loss', 0)
            value_loss = train_metrics.get('training/value/value_loss', 0)
            v_mean = train_metrics.get('training/value/v_mean', 0)
            tqdm.tqdm.write(
                f"Step {i}: actor_loss={actor_loss:.4f}, value_loss={value_loss:.4f}, v_mean={v_mean:.4f}"
            )

        # Save agent
        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)
            print(f"Saved checkpoint at step {i}")

    # Save final checkpoint
    save_agent(agent, FLAGS.save_dir, FLAGS.train_steps)
    print(f"\nTraining complete. Final checkpoint saved at {FLAGS.save_dir}")

    train_logger.close()


if __name__ == '__main__':
    app.run(main)
