"""
Standalone FAC training script for the Go2JoystickFlatTerrain dataset.

Downloads the Minari HDF5 dataset from HuggingFace
(akcit-rl/playground :: Go2JoystickFlatTerrain-fowardfixed-expert-v1),
converts it to the FAC offline RL format, trains the full FAC pipeline
(BC flow pre-training -> log-density computation -> actor-critic), and
saves checkpoints to the specified save directory.

Usage:
    uv run train_go2_fac.py \
        --save_dir mujoco_scene/ \
        --save_interval 100000 \
        --agent.fac_alpha=1.0 \
        --agent.fac_lambda=1.0 \
        --agent.q_agg=mean
"""

import json
import os
import random
import time
import warnings

import jax
import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from agents import agents
from utils.datasets import Dataset, ReplayBuffer
from utils.flax_utils import (
    restore_agent,
    restore_only_bcmodel,
    save_agent,
    save_only_bcmodel,
)
from utils.log_utils import CsvLogger, get_exp_name, setup_wandb

warnings.filterwarnings(action="ignore")

# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------
FLAGS = flags.FLAGS

flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_string(
    "hf_repo_id",
    "akcit-rl/playground",
    "HuggingFace dataset repository ID.",
)
flags.DEFINE_string(
    "hf_dataset_name",
    "Go2JoystickFlatTerrain-fowardfixed-expert-v1",
    "Dataset name (subfolder) within the HuggingFace repo.",
)
flags.DEFINE_string("save_dir", "mujoco_scene/", "Save directory.")
flags.DEFINE_string("restore_path", None, "Restore path.")
flags.DEFINE_integer("restore_epoch", None, "Restore epoch.")

flags.DEFINE_integer("offline_steps", 1000000, "Number of offline training steps.")
flags.DEFINE_integer("buffer_size", 2000000, "Replay buffer size.")
flags.DEFINE_integer("log_interval", 10000, "Logging interval.")
flags.DEFINE_integer("save_interval", 100000, "Saving interval (0 = disabled).")
flags.DEFINE_integer("epoch_flow_proxy", 250, "Epochs for BC flow pre-training.")

flags.DEFINE_float(
    "reward_scale", 1.0, "Scale factor applied to all rewards from the dataset."
)

flags.DEFINE_string("run_project", "Go2_FAC", "W&B project name.")
flags.DEFINE_string("run_group", "Go2_FAC", "W&B run group.")
flags.DEFINE_string("run_job", "train", "W&B job type.")
flags.DEFINE_string("wandb_entity", "jpaguiar399", "W&B entity.")

config_flags.DEFINE_config_file("agent", "agents/fac.py", lock_config=False)

# Name used for pretrained_bc cache (mirrors the env_name convention in main.py)
ENV_NAME = "Go2JoystickFlatTerrain-fowardfixed-expert"


# ---------------------------------------------------------------------------
# Dataset loading from HuggingFace Minari HDF5
# ---------------------------------------------------------------------------

def load_go2_dataset(
    hf_repo_id: str,
    hf_dataset_name: str,
    reward_scale: float = 1.0,
    action_clip_eps: float = 1e-5,
):
    """
    Download and load the Go2 Minari HDF5 dataset from HuggingFace.

    The HDF5 file has episodes stored as:
        episode_N/observations  (T+1, 48)
        episode_N/actions       (T, 12)
        episode_N/rewards       (T,)
        episode_N/terminations  (T,)
        episode_N/truncations   (T,)

    Returns a dict with:
        observations, actions, rewards, terminals, masks, next_observations
    all as float32 numpy arrays ready for FAC's Dataset class.
    """
    import h5py
    from huggingface_hub import hf_hub_download

    hdf5_path = hf_hub_download(
        repo_id=hf_repo_id,
        filename=f"{hf_dataset_name}/data/main_data.hdf5",
        repo_type="dataset",
    )
    print(f"Dataset file: {hdf5_path}")

    all_obs, all_next_obs, all_actions = [], [], []
    all_rewards, all_terminals, all_masks = [], [], []

    with h5py.File(hdf5_path, "r") as f:
        episode_keys = sorted(
            [k for k in f.keys() if k.startswith("episode_")],
            key=lambda k: int(k.split("_")[1]),
        )
        print(f"Found {len(episode_keys)} episodes")

        for ep_key in tqdm.tqdm(episode_keys, desc="Loading episodes"):
            ep = f[ep_key]
            obs = ep["observations"][:].astype(np.float32)       # (T+1, 48)
            actions = ep["actions"][:].astype(np.float32)         # (T, 12)
            rewards = ep["rewards"][:].astype(np.float32)         # (T,)
            terminations = ep["terminations"][:].astype(np.float32)  # (T,)
            truncations = ep["truncations"][:].astype(np.float32)    # (T,)

            # obs[:-1] = states, obs[1:] = next_states
            all_obs.append(obs[:-1])
            all_next_obs.append(obs[1:])
            all_actions.append(actions)
            all_rewards.append(rewards * reward_scale)

            # terminals: 1 at episode end (either termination or truncation)
            dones = np.clip(terminations + truncations, 0.0, 1.0)
            all_terminals.append(dones)

            # masks: 0 when truly terminated (not just truncated)
            # For truncated episodes, we still bootstrap (mask=1)
            masks = 1.0 - terminations
            all_masks.append(masks)

    observations = np.concatenate(all_obs, axis=0)
    next_observations = np.concatenate(all_next_obs, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    rewards = np.concatenate(all_rewards, axis=0)
    terminals = np.concatenate(all_terminals, axis=0)
    masks = np.concatenate(all_masks, axis=0)

    # Clip actions to (-1+eps, 1-eps) as the main codebase does
    if action_clip_eps is not None:
        actions = np.clip(actions, -1 + action_clip_eps, 1 - action_clip_eps)

    n = len(observations)
    n_episodes = int(terminals.sum())
    print(
        f"Loaded dataset: obs={observations.shape}, actions={actions.shape}, "
        f"N={n}, episodes={n_episodes}, "
        f"reward range=[{rewards.min():.2f}, {rewards.max():.2f}]"
    )

    return dict(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        masks=masks,
        next_observations=next_observations,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(_):
    config = FLAGS.agent
    seed = FLAGS.seed
    random.seed(seed)
    np.random.seed(seed)

    # --- Directories --------------------------------------------------------
    exp_name = get_exp_name(seed)
    save_dir = os.path.join(FLAGS.save_dir, exp_name)
    os.makedirs(save_dir, exist_ok=True)

    flag_dict = {
        "seed": seed,
        "hf_repo_id": FLAGS.hf_repo_id,
        "hf_dataset_name": FLAGS.hf_dataset_name,
        "save_dir": FLAGS.save_dir,
        "offline_steps": FLAGS.offline_steps,
        "log_interval": FLAGS.log_interval,
        "save_interval": FLAGS.save_interval,
        "epoch_flow_proxy": FLAGS.epoch_flow_proxy,
        "reward_scale": FLAGS.reward_scale,
        "agent": config.to_dict(),
    }
    with open(os.path.join(save_dir, "flags.json"), "w") as f:
        json.dump(flag_dict, f, indent=2)

    # --- W&B setup ----------------------------------------------------------
    setup_wandb(
        entity=FLAGS.wandb_entity,
        project=FLAGS.run_project,
        name=exp_name,
    )

    # --- Load dataset -------------------------------------------------------
    raw_dataset = load_go2_dataset(
        hf_repo_id=FLAGS.hf_repo_id,
        hf_dataset_name=FLAGS.hf_dataset_name,
        reward_scale=FLAGS.reward_scale,
    )

    train_dataset = Dataset.create(**raw_dataset)
    train_dataset = ReplayBuffer.create_from_initial_dataset(
        dict(train_dataset),
        size=max(FLAGS.buffer_size, train_dataset.size + 1),
    )
    train_dataset.p_aug = None
    train_dataset.frame_stack = None

    # --- Create agent -------------------------------------------------------
    example_batch = train_dataset.sample(1)
    agent_class = agents[config["agent_name"]]
    agent = agent_class.create(
        seed,
        example_batch["observations"],
        example_batch["actions"],
        config,
    )

    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    # --- BC flow pre-training -----------------------------------------------
    bc_batch_size = int(
        config["batch_size"]
        * (
            1
            if train_dataset.size < 100000
            else 4
            if train_dataset.size < 500000
            else 16
        )
    )

    pretrained_bc_dir = os.path.join(os.getcwd(), "pretrained_bc")
    pretrained_bc_path = os.path.join(pretrained_bc_dir, f"{ENV_NAME}_seed_{seed}")

    if os.path.isdir(pretrained_bc_path):
        print("Found pretrained BC model, loading and skipping BC flow training.")
        agent = restore_only_bcmodel(agent, pretrained_bc_dir, ENV_NAME, str(seed))
    else:
        iters_bc = (train_dataset.size + bc_batch_size - 1) // bc_batch_size
        for _ in tqdm.tqdm(
            range(1, FLAGS.epoch_flow_proxy * iters_bc + 1),
            smoothing=0.1,
            dynamic_ncols=True,
            desc="Train BC",
        ):
            batch = train_dataset.sample(bc_batch_size)
            agent, _ = agent.update(batch, mode="train_bc")

        save_only_bcmodel(agent, pretrained_bc_dir, ENV_NAME, str(seed))

    # --- Log-density computation --------------------------------------------
    train_dataset.compute_and_attach_estimated_logp(
        agent=agent,
        method=config["logp_method"],
        batch_size=bc_batch_size,
    )

    # --- Actor-critic training ----------------------------------------------
    train_logger = CsvLogger(os.path.join(save_dir, "train.csv"))
    first_time = time.time()
    last_time = time.time()

    for i in tqdm.tqdm(
        range(1, FLAGS.offline_steps + 1),
        smoothing=0.1,
        dynamic_ncols=True,
        desc="Train AC",
    ):
        batch = train_dataset.sample(config["batch_size"])
        agent, update_info = agent.update(batch, mode="train_ac")

        if i % FLAGS.log_interval == 0:
            train_metrics = {f"training/{k}": v for k, v in update_info.items()}
            train_metrics["time/epoch_time"] = (
                time.time() - last_time
            ) / FLAGS.log_interval
            train_metrics["time/total_time"] = time.time() - first_time
            last_time = time.time()
            train_logger.log(train_metrics, step=i)
            wandb.log(train_metrics, step=i)

            q_mean = update_info.get("critic/q_mean", float("nan"))
            actor_loss = update_info.get("actor/actor_loss", float("nan"))
            print(
                f"  step={i:>8d}  q_mean={float(q_mean):.4f}  "
                f"actor_loss={float(actor_loss):.4f}"
            )

        if FLAGS.save_interval != 0 and i % FLAGS.save_interval == 0:
            save_agent(agent, save_dir, i)

    train_logger.close()
    wandb.finish()
    print(f"\nTraining complete. Checkpoints saved to: {save_dir}")


if __name__ == "__main__":
    app.run(main)
