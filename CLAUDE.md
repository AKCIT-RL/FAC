# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Implementation of **Flow Actor-Critic (FAC)** for offline reinforcement learning (ICLR 2026). JAX/Flax-based. Supports OGBench and D4RL environments. Built on top of the FQL codebase.

Paper: https://arxiv.org/abs/2602.18015v1

## Commands

### Setup
```bash
uv sync                        # install dependencies (uses pyproject.toml + uv.lock)
python get_dataset.py --env_type=d4rl-antmaze   # download datasets by type
```

### Training (FAC on OGBench/D4RL)
```bash
python main.py --env_name=antmaze-umaze-v2    # default FAC training
python main.py --env_name=puzzle-3x3-play-singletask-v0 --agent.fac_alpha=1.0 --agent.fac_lambda=1.0 --agent.q_agg=mean
```

### Training (baselines: FQL, IFQL, ReBRAC, FBRAC)
```bash
python main_baselines.py --agent='agents/fql.py' --env_name=halfcheetah-medium-v2 --agent.alpha=3.0
```

### Training (Go2 robot — custom HuggingFace dataset)
```bash
uv run train_go2_fac.py --save_dir mujoco_scene/ --save_interval 100000 --agent.fac_alpha=0.5 --agent.fac_lambda=0.5 --agent.q_agg=mean
# or: bash run_go2_train.sh
```

### Restore / Evaluate only
```bash
python main.py --env_name=antmaze-umaze-v2 --offline_steps 0 --online_steps 0 \
  --restore_path "exp/Project/Group/sd000_*" --restore_epoch 100000 --eval_interval 1
```

## Architecture

### Training Pipeline (main.py)
Three-phase pipeline, all in one script:
1. **BC flow pre-training** — trains a behavior-cloning flow model as a density proxy. Auto-saved to `pretrained_bc/` and reloaded on subsequent runs.
2. **Log-density computation** — attaches `estimated_logp` to the offline dataset using the trained BC flow model.
3. **Actor-Critic training** — trains the FAC policy+critic using flow-based conservative Q-penalization.

`main_baselines.py` is a simpler version for non-FAC agents (no BC phase, no log-density).

### Agent System (`agents/`)
- Each agent file (e.g., `agents/fac.py`) serves dual purpose: it defines both a `ml_collections` config (via `get_config()`) and the agent class (e.g., `FACAgent`).
- Agent classes are Flax PyTreeNodes with `create()`, `update()`, `sample_actions()`, and loss methods.
- `agents/__init__.py` registers all agents in a dict keyed by `agent_name`.
- `FACAgent.update()` accepts a `mode` parameter: `"train_bc"` (BC flow phase) or `"train_ac"` (actor-critic phase).
- Baseline agents use a single `update()` without mode switching.

### Key Hyperparameters (FAC-specific)
- `fac_alpha` — conservative penalty strength
- `fac_lambda` — penalty weighting
- `fac_threshold` — threshold scheme: `'batch_adaptive'`, `'batch_wide_constant'`, `'dataset_wide_constant'`
- `q_agg` — Q-value aggregation: `'min'` (default) or `'mean'`
- `logp_method` — log-density estimation method (e.g., `'hutch-rade'`)

### Environment Routing (`envs/env_utils.py`)
`make_env_and_datasets()` dispatches based on `env_name` string:
- `'singletask'` in name → OGBench
- `'antmaze'` / `'halfcheetah'` / `'hopper'` / `'walker2d'` → D4RL MuJoCo
- `'pen'` / `'hammer'` / `'door'` / `'relocate'` → D4RL Adroit

### Utilities (`utils/`)
- `datasets.py` — `Dataset` and `ReplayBuffer` (Flax PyTreeNodes). `Dataset.compute_and_attach_estimated_logp()` is the bridge between BC and AC phases.
- `networks.py` — neural network modules (`ActorVectorField`, `Actor`, `Value`, `Scalar`)
- `encoders.py` — image encoders for visual (pixel-based) OGBench tasks
- `flax_utils.py` — checkpoint save/restore, `ModuleDict`, `TrainState`
- `evaluation.py` — evaluation loop and trajectory collection
- `log_utils.py` — W&B setup, CSV logging

### Output Structure
```
exp/<run_project>/<run_group>/<exp_name>/
  flags.json, train.csv, eval.csv, params_<step>.pkl

pretrained_bc/<env_name>_seed_<seed>/   # auto-saved BC flow model
```

## Key Details

- Uses `absl` flags + `ml_collections.config_flags` — agent hyperparams are passed as `--agent.<param>=<value>`.
- W&B logging is enabled by default in `main.py` (entity hardcoded to `jpaguiar399`). `main_baselines.py` requires setting entity/key manually.
- D4RL environments need `mujoco210` installed (see `install_mujoco.sh`).
- Python 3.12, managed with `uv`. JAX with CUDA 12.
- No test suite exists in this repo.
