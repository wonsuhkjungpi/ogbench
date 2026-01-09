# Quickstart Environment Setup Changelog

This file documents all environment fixes and changes made during setup.

## Initial Setup (2026-01-09)

- [CREATE] Created conda environment `ogbench` with Python 3.10
- [INSTALL] Installed ogbench with training dependencies: `pip install -e .[train]`
- [VERIFY] JAX successfully detects 2 CUDA devices
- [VERIFY] ogbench package imports correctly

## Environment Fixes

- [FIX] Modified `impls/utils/log_utils.py` to respect `WANDB_MODE` environment variable
  - Issue: The `setup_wandb()` function had hardcoded `mode='online'` which caused permission errors
  - Solution: Added `wandb_mode = os.environ.get('WANDB_MODE', mode)` to read from env var
  - This allows running experiments with `WANDB_MODE=offline` to avoid wandb API permission issues

## Dryrun Test Results (2026-01-09)

- [SUCCESS] Dryrun completed in ~15 seconds
- Dataset downloaded: `antmaze-large-navigate-v0.npz` (232MB)
- Validation dataset: `antmaze-large-navigate-v0-val.npz` (23MB)
- 1000 training steps completed with GCBC agent
- Evaluation on 2 tasks (2 episodes each) successful

## Notes

- Environment name: `ogbench`
- Python version: 3.10
- JAX version: 0.6.2 with CUDA 12 support
- ogbench version: 1.2.0 (editable install)
- Dataset location: `~/.ogbench/data/`

## Known Issues

- Minor: botocore dependency conflict (jmespath not installed), but does not affect core functionality
- wandb needs offline mode if API key has permission issues
