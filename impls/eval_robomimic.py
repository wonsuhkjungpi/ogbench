"""
Evaluation script for OGBench agents on robomimic environments.

This script loads a trained OGBench agent and evaluates it on the robomimic
square environment using goal observations from the end of demonstrations.

Usage (in ogbench conda environment):
    python eval_robomimic.py \
        --agent_path=/path/to/params_500000.pkl \
        --agent_config=agents/gcivl.py \
        --dataset_path=/path/to/square_train.npz \
        --n_episodes=10
"""

import os
import sys
import json
import pickle
import numpy as np
import jax
import jax.numpy as jnp
from absl import app, flags
from ml_collections import config_flags

# Add parent paths for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents import agents
from dataset_robomimic import load_npz_as_dataset
from utils.datasets import GCDataset
from utils.flax_utils import TrainState

FLAGS = flags.FLAGS

flags.DEFINE_string('agent_path', None, 'Path to agent checkpoint (params_*.pkl).')
flags.DEFINE_string('dataset_path', None, 'Path to training NPZ (for goal extraction).')
flags.DEFINE_integer('n_episodes', 10, 'Number of evaluation episodes.')
flags.DEFINE_integer('max_steps', 400, 'Maximum steps per episode.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_float('temperature', 1.0, 'Sampling temperature.')
flags.DEFINE_string('output_path', None, 'Path to save results JSON.')
flags.DEFINE_bool('render', False, 'Whether to render episodes.')

config_flags.DEFINE_config_file('agent_config', 'agents/gcivl.py', lock_config=False)


def load_agent_from_checkpoint(agent_path, config, example_obs, example_action):
    """Load an OGBench agent from checkpoint."""
    # Create agent with example batch
    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_obs,
        example_action,
        config,
    )

    # Load checkpoint
    with open(agent_path, 'rb') as f:
        load_dict = pickle.load(f)

    # Restore agent state
    import flax.serialization
    agent = flax.serialization.from_state_dict(agent, load_dict['agent'])

    print(f"Loaded agent from {agent_path}")
    return agent


def extract_goal_observations(dataset, n_goals=10, seed=42):
    """Extract goal observations from the end of episodes in the dataset."""
    np.random.seed(seed)

    # Find episode boundaries (where terminals == 1)
    terminals = dataset['terminals']
    terminal_indices = np.where(terminals == 1.0)[0]

    if len(terminal_indices) == 0:
        # Fallback: use last observations from dataset
        print("Warning: No terminal states found, using last observations")
        total_len = len(dataset['observations'])
        terminal_indices = np.array([total_len - 1])

    # Sample random episode endings as goals
    selected_indices = np.random.choice(terminal_indices, size=min(n_goals, len(terminal_indices)), replace=False)

    goals = []
    for idx in selected_indices:
        goal_obs = dataset['observations'][idx]
        goals.append(goal_obs)

    return np.array(goals), selected_indices


def create_robomimic_env(task_name='square'):
    """Create a robomimic environment for evaluation."""
    try:
        import robosuite as suite
        from robosuite.wrappers import GymWrapper

        # Create environment
        env = suite.make(
            env_name="NutAssemblySquare",
            robots="Panda",
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            reward_shaping=True,
            control_freq=20,
            horizon=400,
        )

        return env
    except ImportError as e:
        print(f"Error importing robosuite: {e}")
        print("Make sure you're in the faildetect conda environment for robosuite.")
        return None


def get_state_observation(env, obs_keys=None):
    """Extract state observation from robosuite environment observation."""
    if obs_keys is None:
        obs_keys = ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'object']

    obs = env._get_observations()
    state_obs = []
    for key in obs_keys:
        if key in obs:
            state_obs.append(obs[key].flatten())

    return np.concatenate(state_obs)


def compute_state_distance(obs1, obs2):
    """Compute L2 distance between two observations."""
    return np.linalg.norm(obs1 - obs2)


def run_evaluation(agent, goals, config, n_episodes=10, max_steps=400, temperature=1.0, seed=42):
    """Run evaluation episodes."""
    # For now, we'll do a simplified evaluation without the actual environment
    # since robosuite requires the faildetect environment

    print(f"\n{'='*60}")
    print("OGBench Agent Evaluation (Simplified - No Env)")
    print(f"{'='*60}")
    print(f"Number of goals: {len(goals)}")
    print(f"Goal observation shape: {goals[0].shape}")
    print(f"Temperature: {temperature}")

    # Create random observations for testing action prediction
    np.random.seed(seed)
    rng = jax.random.PRNGKey(seed)

    results = []
    for ep in range(n_episodes):
        goal_idx = ep % len(goals)
        goal = goals[goal_idx]

        # Create a random starting observation (same shape as goal)
        obs = np.random.randn(*goal.shape).astype(np.float32) * 0.1

        # Test action prediction
        rng, action_key = jax.random.split(rng)

        # Expand dims for batch
        obs_batch = obs[np.newaxis, :]
        goal_batch = goal[np.newaxis, :]

        # Sample action from agent
        action = agent.sample_actions(
            observations=obs_batch,
            goals=goal_batch,
            seed=action_key,
            temperature=temperature
        )

        action = np.array(action[0])  # Remove batch dim

        result = {
            'episode': ep,
            'goal_idx': goal_idx,
            'action_shape': list(action.shape),
            'action_mean': float(np.mean(action)),
            'action_std': float(np.std(action)),
            'action_min': float(np.min(action)),
            'action_max': float(np.max(action)),
        }
        results.append(result)

        print(f"Episode {ep+1}/{n_episodes}: action shape={action.shape}, "
              f"mean={np.mean(action):.4f}, std={np.std(action):.4f}")

    return results


def main(_):
    if FLAGS.agent_path is None:
        raise ValueError("--agent_path is required")
    if FLAGS.dataset_path is None:
        raise ValueError("--dataset_path is required")

    print(f"Loading dataset from {FLAGS.dataset_path}")

    # Load dataset
    dataset = np.load(FLAGS.dataset_path, allow_pickle=True)

    # Get example observation and action for agent initialization
    example_obs = dataset['observations'][:1]
    example_action = dataset['actions'][:1]

    print(f"Observation shape: {example_obs.shape}")
    print(f"Action shape: {example_action.shape}")

    # Load agent config
    config = FLAGS.agent_config
    print(f"Agent: {config['agent_name']}")

    # Load agent
    agent = load_agent_from_checkpoint(
        FLAGS.agent_path,
        config,
        example_obs,
        example_action
    )

    # Extract goal observations
    goals, goal_indices = extract_goal_observations(
        dataset,
        n_goals=FLAGS.n_episodes,
        seed=FLAGS.seed
    )
    print(f"Extracted {len(goals)} goal observations from episodes ending at indices: {goal_indices}")

    # Run evaluation
    results = run_evaluation(
        agent=agent,
        goals=goals,
        config=config,
        n_episodes=FLAGS.n_episodes,
        max_steps=FLAGS.max_steps,
        temperature=FLAGS.temperature,
        seed=FLAGS.seed
    )

    # Summary
    print(f"\n{'='*60}")
    print("Evaluation Summary")
    print(f"{'='*60}")
    print(f"Total episodes: {len(results)}")
    print(f"Agent successfully loaded and predicts actions")

    # Save results if output path specified
    if FLAGS.output_path:
        output_dir = os.path.dirname(FLAGS.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(FLAGS.output_path, 'w') as f:
            json.dump({
                'config': {
                    'agent_path': FLAGS.agent_path,
                    'dataset_path': FLAGS.dataset_path,
                    'n_episodes': FLAGS.n_episodes,
                    'temperature': FLAGS.temperature,
                    'seed': FLAGS.seed,
                },
                'results': results,
            }, f, indent=2)
        print(f"\nResults saved to {FLAGS.output_path}")

    return results


if __name__ == '__main__':
    app.run(main)
