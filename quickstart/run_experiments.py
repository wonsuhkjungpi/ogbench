#!/usr/bin/env python3
"""
OGBench Quickstart - Single-click Experiment Runner

This script provides a convenient way to run OGBench experiments.
It handles environment activation and directory navigation automatically.

Usage:
    python run_experiments.py              # Run full experiment
    python run_experiments.py --dryrun     # Run quick validation
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def get_script_dir():
    """Get the directory containing this script."""
    return Path(__file__).parent.resolve()


def run_experiment(dryrun: bool = False):
    """
    Run OGBench experiment.

    Args:
        dryrun: If True, run minimal validation instead of full training
    """
    script_dir = get_script_dir()

    if dryrun:
        script_name = "run_experiments_dryrun.sh"
        print("Running DRYRUN experiment (quick validation)...")
    else:
        script_name = "run_experiments.sh"
        print("Running FULL experiment...")

    script_path = script_dir / script_name

    if not script_path.exists():
        print(f"Error: Script not found: {script_path}")
        sys.exit(1)

    # Make sure the script is executable
    os.chmod(script_path, 0o755)

    # Run the bash script
    result = subprocess.run(
        ["bash", str(script_path)],
        cwd=str(script_dir),
    )

    sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="OGBench Quickstart Experiment Runner"
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Run quick validation instead of full training"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full training (default if no flag specified)"
    )

    args = parser.parse_args()

    # Default to dryrun for safety
    dryrun = args.dryrun or not args.full

    run_experiment(dryrun=dryrun)


if __name__ == "__main__":
    main()
