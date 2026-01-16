"""
Benchmark evaluation script for OGBench agents on robomimic testcases.

This script evaluates trained OGBench agents (GCIVL, CRL) on the benchmark
testcases from experiments/benchmark/square/final100/.

Usage (in faildetect conda environment):
    python eval_robomimic_benchmark.py \
        --agent_path=/path/to/params_500000.pkl \
        --agent_config=agents/crl.py \
        --dataset_path=/path/to/square_train.npz \
        --benchmark_path=/path/to/benchmark/square/final100 \
        --output_path=/path/to/results.json
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
from pathlib import Path
from tqdm import tqdm

# Add parent paths for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents import agents
import flax.serialization

FLAGS = flags.FLAGS

flags.DEFINE_string('agent_path', None, 'Path to agent checkpoint (params_*.pkl).')
flags.DEFINE_string('dataset_path', None, 'Path to training NPZ (for agent initialization).')
flags.DEFINE_string('benchmark_path', None, 'Path to benchmark directory (e.g., .../square/final100).')
flags.DEFINE_integer('max_steps', 400, 'Maximum steps per episode.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_float('temperature', 1.0, 'Sampling temperature.')
flags.DEFINE_string('output_path', None, 'Path to save results JSON.')
flags.DEFINE_bool('render', False, 'Whether to render episodes.')
flags.DEFINE_integer('max_testcases_per_group', None, 'Max testcases to run per group (for testing).')
flags.DEFINE_string('algorithm', 'crl', 'Algorithm name (gcivl, crl, gcbc, gciql).')

config_flags.DEFINE_config_file('agent_config', 'agents/crl.py', lock_config=False)


# Observation keys for robomimic square task (matching OGBench training)
# Note: robosuite uses 'object-state' but robomimic HDF5 uses 'object'
ROBOMIMIC_OBS_KEYS = ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'object-state']


def load_agent_from_checkpoint(agent_path, config, example_obs, example_action):
    """Load an OGBench agent from checkpoint."""
    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_obs,
        example_action,
        config,
    )

    with open(agent_path, 'rb') as f:
        load_dict = pickle.load(f)

    agent = flax.serialization.from_state_dict(agent, load_dict['agent'])
    print(f"Loaded agent from {agent_path}")
    return agent


def create_robosuite_env():
    """Create a robosuite environment for evaluation."""
    try:
        import robosuite as suite
        from robosuite.controllers import load_controller_config

        # Use OSC_POSE controller for 7-dim actions (matching robomimic)
        controller_config = load_controller_config(default_controller='OSC_POSE')

        env = suite.make(
            env_name="NutAssemblySquare",
            robots="Panda",
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            use_object_obs=True,  # Enable object observations
            reward_shaping=True,
            control_freq=20,
            horizon=400,
            controller_configs=controller_config,
        )
        print(f"Environment action dim: {env.action_dim}")
        return env
    except ImportError as e:
        print(f"Error importing robosuite: {e}")
        return None


def get_state_observation(env, obs_dict=None):
    """
    Extract state observation matching OGBench training format.

    Returns 23-dim vector: [eef_pos(3), eef_quat(4), gripper_qpos(2), object(14)]
    """
    if obs_dict is None:
        obs_dict = env._get_observations()

    state_obs = []
    for key in ROBOMIMIC_OBS_KEYS:
        if key in obs_dict:
            data = obs_dict[key]
            if isinstance(data, np.ndarray):
                state_obs.append(data.flatten())
            else:
                state_obs.append(np.array(data).flatten())

    return np.concatenate(state_obs).astype(np.float32)


def set_env_state(env, state):
    """
    Reset environment to a specific state.

    Args:
        env: Robosuite environment
        state: 45-dim mujoco state vector (or dict with 'state' key)
    """
    if isinstance(state, dict):
        state = state['state']

    # Reset environment fully, then set the specific state
    env.reset()
    env.sim.set_state_from_flattened(state)
    env.sim.forward()


def load_benchmark_testcases(benchmark_path):
    """Load all testcases from benchmark directory."""
    benchmark_path = Path(benchmark_path)
    groups_dir = benchmark_path / 'groups'

    if not groups_dir.exists():
        raise FileNotFoundError(f"Groups directory not found: {groups_dir}")

    all_testcases = {}

    # Iterate over each group
    for group_dir in sorted(groups_dir.iterdir()):
        if not group_dir.is_dir():
            continue

        group_name = group_dir.name
        testcases_dir = group_dir / 'testcases'
        testcases_json = group_dir / 'testcases.json'

        if not testcases_dir.exists():
            print(f"Warning: No testcases directory for group {group_name}")
            continue

        # Load testcase metadata from JSON (if exists)
        testcase_meta = {}
        if testcases_json.exists():
            with open(testcases_json, 'r') as f:
                # Parse JSONL format (one JSON per block)
                content = f.read()
                # Split by "}\n{" pattern
                blocks = content.replace('}\n{', '}|||{').split('|||')
                for block in blocks:
                    try:
                        meta = json.loads(block)
                        testcase_meta[meta['testcase_id']] = meta
                    except json.JSONDecodeError:
                        continue

        # Load testcase pkl files
        group_testcases = []
        for pkl_file in sorted(testcases_dir.glob('*.pkl')):
            with open(pkl_file, 'rb') as f:
                tc = pickle.load(f)

            # Add metadata
            tc_id = pkl_file.stem
            if tc_id in testcase_meta:
                tc['meta'].update(testcase_meta[tc_id])

            group_testcases.append(tc)

        if group_testcases:
            all_testcases[group_name] = group_testcases
            print(f"Loaded {len(group_testcases)} testcases from group: {group_name}")

    return all_testcases


def run_evaluation_episode(env, agent, initial_state, goal_obs_np, max_steps, temperature, rng):
    """
    Run a single evaluation episode.

    Args:
        env: Robosuite environment
        agent: OGBench agent
        initial_state: Initial state dict from testcase (or raw state array)
        goal_obs_np: Goal observation (23-dim numpy array)
        max_steps: Maximum steps
        temperature: Sampling temperature
        rng: JAX random key

    Returns:
        Dictionary with episode results
    """
    # Reset environment to initial state
    set_env_state(env, initial_state)

    # Get initial observation
    obs = get_state_observation(env)
    initial_obs = obs.copy()

    # Compute initial distance to goal
    initial_distance = np.linalg.norm(obs - goal_obs_np)

    trajectory = {
        'observations': [obs.copy()],
        'actions': [],
        'distances': [initial_distance],
    }

    terminated = False
    success = False
    best_distance = initial_distance
    step = 0

    for step in range(max_steps):
        # Sample action from agent
        rng, action_key = jax.random.split(rng)

        obs_batch = obs[np.newaxis, :]
        goal_batch = goal_obs_np[np.newaxis, :]

        action = agent.sample_actions(
            observations=obs_batch,
            goals=goal_batch,
            seed=action_key,
            temperature=temperature
        )
        action = np.array(action[0])  # Remove batch dim

        # Step environment (with error handling)
        try:
            obs_dict, reward, done, info = env.step(action)
        except ValueError as e:
            if "terminated episode" in str(e):
                terminated = True
                break
            raise

        obs = get_state_observation(env, obs_dict)

        # Compute distance to goal
        distance = np.linalg.norm(obs - goal_obs_np)
        best_distance = min(best_distance, distance)

        trajectory['observations'].append(obs.copy())
        trajectory['actions'].append(action.copy())
        trajectory['distances'].append(distance)

        # Check success (robosuite-specific)
        if hasattr(env, '_check_success') and env._check_success():
            success = True

        if done:
            terminated = True
            break

    final_obs = obs
    final_distance = np.linalg.norm(final_obs - goal_obs_np)

    return {
        'success': success,
        'terminated': terminated,
        'n_steps': step + 1,
        'initial_distance': float(initial_distance),
        'final_distance': float(final_distance),
        'best_distance': float(best_distance),
        'distance_improvement': float(initial_distance - final_distance),
        'trajectory': trajectory,
        'rng': rng,
    }


def extract_goal_obs_from_testcase(testcase, env):
    """
    Extract goal observation (23-dim) from testcase.

    The testcase has goal_obs dict with images and low-dim obs, but not object.
    We need to set the env to goal_state and get full observation.
    """
    # Set environment to goal state and get observations
    set_env_state(env, testcase['goal_state'])
    goal_obs = get_state_observation(env)
    return goal_obs


def main(_):
    if FLAGS.agent_path is None:
        raise ValueError("--agent_path is required")
    if FLAGS.dataset_path is None:
        raise ValueError("--dataset_path is required")
    if FLAGS.benchmark_path is None:
        raise ValueError("--benchmark_path is required")

    print(f"Loading dataset from {FLAGS.dataset_path}")

    # Load dataset for agent initialization
    dataset = np.load(FLAGS.dataset_path, allow_pickle=True)
    example_obs = dataset['observations'][:1]
    example_action = dataset['actions'][:1]

    print(f"Observation shape: {example_obs.shape}")
    print(f"Action shape: {example_action.shape}")

    # Load agent config and create agent
    config = FLAGS.agent_config
    print(f"Agent: {config['agent_name']}")

    agent = load_agent_from_checkpoint(
        FLAGS.agent_path,
        config,
        example_obs,
        example_action
    )

    # Create environment
    print("\nCreating robosuite environment...")
    env = create_robosuite_env()
    if env is None:
        raise RuntimeError("Failed to create robosuite environment")

    # Reset environment once to initialize
    env.reset()

    # Load benchmark testcases
    print(f"\nLoading benchmark from {FLAGS.benchmark_path}")
    testcases_by_group = load_benchmark_testcases(FLAGS.benchmark_path)

    # Run evaluation
    print(f"\n{'='*60}")
    print("OGBench Agent Benchmark Evaluation")
    print(f"{'='*60}")
    print(f"Algorithm: {FLAGS.algorithm}")
    print(f"Temperature: {FLAGS.temperature}")
    print(f"Max steps per episode: {FLAGS.max_steps}")

    rng = jax.random.PRNGKey(FLAGS.seed)
    all_results = {}

    for group_name, testcases in testcases_by_group.items():
        print(f"\n--- Evaluating group: {group_name} ---")

        # Limit testcases if specified
        if FLAGS.max_testcases_per_group is not None:
            testcases = testcases[:FLAGS.max_testcases_per_group]

        group_results = []

        for tc_idx, testcase in enumerate(tqdm(testcases, desc=f"Evaluating {group_name}")):
            # Extract goal observation
            goal_obs = extract_goal_obs_from_testcase(testcase, env)

            # Run evaluation episode
            result = run_evaluation_episode(
                env=env,
                agent=agent,
                initial_state=testcase['initial_state'],
                goal_obs_np=goal_obs,
                max_steps=FLAGS.max_steps,
                temperature=FLAGS.temperature,
                rng=rng,
            )
            rng = result['rng']

            # Store result (without trajectory to save space)
            tc_result = {
                'testcase_id': testcase.get('source_traj_id', f'tc_{tc_idx}'),
                'H': testcase.get('H', -1),
                'success': result['success'],
                'n_steps': result['n_steps'],
                'initial_distance': result['initial_distance'],
                'final_distance': result['final_distance'],
                'best_distance': result['best_distance'],
                'distance_improvement': result['distance_improvement'],
            }
            group_results.append(tc_result)

        # Compute group statistics
        successes = [r['success'] for r in group_results]
        distances_initial = [r['initial_distance'] for r in group_results]
        distances_final = [r['final_distance'] for r in group_results]
        distances_best = [r['best_distance'] for r in group_results]
        improvements = [r['distance_improvement'] for r in group_results]

        group_summary = {
            'n_testcases': len(group_results),
            'success_rate': np.mean(successes) if successes else 0.0,
            'mean_initial_distance': np.mean(distances_initial),
            'mean_final_distance': np.mean(distances_final),
            'mean_best_distance': np.mean(distances_best),
            'mean_improvement': np.mean(improvements),
            'std_improvement': np.std(improvements),
        }

        all_results[group_name] = {
            'summary': group_summary,
            'testcases': group_results,
        }

        print(f"  Success rate: {group_summary['success_rate']:.2%}")
        print(f"  Mean distance: {group_summary['mean_initial_distance']:.4f} -> {group_summary['mean_final_distance']:.4f}")
        print(f"  Mean improvement: {group_summary['mean_improvement']:.4f}")

    # Compute overall statistics
    all_successes = []
    all_improvements = []
    for group_name, group_data in all_results.items():
        for tc in group_data['testcases']:
            all_successes.append(tc['success'])
            all_improvements.append(tc['distance_improvement'])

    overall_summary = {
        'algorithm': FLAGS.algorithm,
        'agent_path': FLAGS.agent_path,
        'n_groups': len(all_results),
        'n_testcases_total': len(all_successes),
        'overall_success_rate': np.mean(all_successes) if all_successes else 0.0,
        'overall_mean_improvement': np.mean(all_improvements) if all_improvements else 0.0,
        'overall_std_improvement': np.std(all_improvements) if all_improvements else 0.0,
    }

    print(f"\n{'='*60}")
    print("Overall Results")
    print(f"{'='*60}")
    print(f"Total testcases: {overall_summary['n_testcases_total']}")
    print(f"Overall success rate: {overall_summary['overall_success_rate']:.2%}")
    print(f"Overall mean improvement: {overall_summary['overall_mean_improvement']:.4f} (+/- {overall_summary['overall_std_improvement']:.4f})")

    # Save results
    if FLAGS.output_path:
        output_dir = os.path.dirname(FLAGS.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        output_data = {
            'config': {
                'agent_path': FLAGS.agent_path,
                'dataset_path': FLAGS.dataset_path,
                'benchmark_path': FLAGS.benchmark_path,
                'algorithm': FLAGS.algorithm,
                'temperature': FLAGS.temperature,
                'max_steps': FLAGS.max_steps,
                'seed': FLAGS.seed,
            },
            'overall_summary': overall_summary,
            'groups': {k: v['summary'] for k, v in all_results.items()},
            'detailed_results': all_results,
        }

        with open(FLAGS.output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {FLAGS.output_path}")

    env.close()
    return all_results


if __name__ == '__main__':
    app.run(main)
