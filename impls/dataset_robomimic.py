"""
Robomimic dataset loader for OGBench.

This module provides functions to load converted robomimic NPZ datasets
into OGBench's Dataset and GCDataset format.

Usage:
    from dataset_robomimic import load_robomimic_dataset

    train_dataset, val_dataset = load_robomimic_dataset(
        train_path='path/to/square_train.npz',
        val_path='path/to/square_train-val.npz',
        config=agent_config,
    )
"""

import numpy as np
from utils.datasets import Dataset, GCDataset, HGCDataset


def load_npz_as_dataset(npz_path: str) -> Dataset:
    """
    Load an NPZ file and convert to OGBench Dataset.

    Args:
        npz_path: Path to NPZ file with keys:
            - observations: (N, obs_dim)
            - actions: (N, action_dim)
            - terminals: (N,)
            - next_observations: (N, obs_dim) [optional]
            - valids: (N,) [optional, for compact format]

    Returns:
        OGBench Dataset object
    """
    data = np.load(npz_path, allow_pickle=True)

    dataset_dict = {
        'observations': data['observations'],
        'actions': data['actions'],
        'terminals': data['terminals'],
    }

    # Add optional fields if present
    if 'next_observations' in data:
        dataset_dict['next_observations'] = data['next_observations']
    if 'valids' in data:
        dataset_dict['valids'] = data['valids']

    return Dataset.create(**dataset_dict)


def load_robomimic_dataset(
    train_path: str,
    val_path: str = None,
    config: dict = None,
) -> tuple:
    """
    Load robomimic NPZ datasets for OGBench training.

    Args:
        train_path: Path to training NPZ file
        val_path: Path to validation NPZ file (optional)
        config: Agent config dict with GCDataset parameters

    Returns:
        Tuple of (train_gc_dataset, val_gc_dataset) if config provided,
        or (train_dataset, val_dataset) if config is None
    """
    train_dataset = load_npz_as_dataset(train_path)
    val_dataset = load_npz_as_dataset(val_path) if val_path else None

    if config is not None:
        # Determine dataset class from config
        dataset_class = {
            'GCDataset': GCDataset,
            'HGCDataset': HGCDataset,
        }.get(config.get('dataset_class', 'GCDataset'), GCDataset)

        train_gc_dataset = dataset_class(train_dataset, config)
        val_gc_dataset = dataset_class(val_dataset, config) if val_dataset else None
        return train_gc_dataset, val_gc_dataset

    return train_dataset, val_dataset


def get_dataset_info(npz_path: str) -> dict:
    """
    Get information about a robomimic NPZ dataset.

    Args:
        npz_path: Path to NPZ file

    Returns:
        Dict with dataset statistics
    """
    data = np.load(npz_path, allow_pickle=True)

    n_transitions = len(data['observations'])
    n_episodes = int(data['terminals'].sum())
    obs_dim = data['observations'].shape[-1]
    action_dim = data['actions'].shape[-1]

    # Check if image data (3D or 4D observations)
    is_image = len(data['observations'].shape) > 2

    info = {
        'n_transitions': n_transitions,
        'n_episodes': n_episodes,
        'obs_dim': obs_dim if not is_image else data['observations'].shape[1:],
        'action_dim': action_dim,
        'is_image': is_image,
        'has_next_obs': 'next_observations' in data,
        'is_compact': 'valids' in data,
    }

    return info


if __name__ == '__main__':
    # Test loading
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
        info = get_dataset_info(path)
        print(f"Dataset info for {path}:")
        for k, v in info.items():
            print(f"  {k}: {v}")
