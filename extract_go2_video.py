"""
Roll out a trained FAC Go2 policy in MuJoCo and render a video.

Loads the Go2 MuJoCo model (go2_env/scene.xml), restores a FAC checkpoint,
constructs observations matching the training pipeline (48-dim), and renders
the rollout to an MP4 file.

Usage:
    uv run extract_go2_video.py \
        --restore_path mujoco_scene/sd000_YYYYMMDD_HHMMSS \
        --restore_epoch 100000 \
        --output go2_policy.mp4
"""

import json
import os

import imageio
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from absl import app, flags
from ml_collections import config_flags

from agents import agents
from utils.datasets import Dataset, ReplayBuffer
from utils.flax_utils import restore_agent

FLAGS = flags.FLAGS

flags.DEFINE_string("restore_path", None, "Path to checkpoint directory.", required=True)
flags.DEFINE_integer("restore_epoch", None, "Checkpoint epoch to load.", required=True)
flags.DEFINE_integer("num_episodes", 3, "Number of episodes to record.")
flags.DEFINE_integer("episode_length", 1000, "Max steps per episode (at ctrl_dt=0.02s).")
flags.DEFINE_string("output", "go2_policy.mp4", "Output video filename.")
flags.DEFINE_integer("fps", 50, "Video frames per second.")
flags.DEFINE_integer("render_width", 640, "Render width.")
flags.DEFINE_integer("render_height", 480, "Render height.")
flags.DEFINE_float("forward_cmd", 1.0, "Forward velocity command (x).")
flags.DEFINE_float("lateral_cmd", 0.0, "Lateral velocity command (y).")
flags.DEFINE_float("yaw_cmd", 0.0, "Yaw velocity command.")

# Env physics parameters matching the training env (joystick.py default_config)
ACTION_SCALE = 0.5
KP = 35.0
KD = 0.5
CTRL_DT = 0.02
SIM_DT = 0.004
N_SUBSTEPS = int(CTRL_DT / SIM_DT)  # 5

GO2_XML = os.path.join(os.path.dirname(__file__), "go2_env", "scene.xml")


def get_obs(model, data, default_pose, last_act, command):
    """Construct the 48-dim observation matching the Go2 joystick training env."""
    imu_site_id = model.site("imu").id

    # local_linvel (velocimeter sensor)
    linvel_adr = model.sensor_adr[model.sensor("local_linvel").id]
    local_linvel = data.sensordata[linvel_adr : linvel_adr + 3]

    # gyro sensor
    gyro_adr = model.sensor_adr[model.sensor("gyro").id]
    gyro = data.sensordata[gyro_adr : gyro_adr + 3]

    # gravity in body frame: site_xmat.T @ [0, 0, -1]
    site_xmat = data.site_xmat[imu_site_id].reshape(3, 3)
    gravity = site_xmat.T @ np.array([0.0, 0.0, -1.0])

    # joint angles relative to default pose
    joint_angles = data.qpos[7:] - default_pose

    # joint velocities
    joint_vel = data.qvel[6:]

    obs = np.concatenate([
        local_linvel,    # 3
        gyro,            # 3
        gravity,         # 3
        joint_angles,    # 12
        joint_vel,       # 12
        last_act,        # 12
        command,         # 3
    ]).astype(np.float32)

    return obs


def run_episode(model, agent, config, default_pose, command, episode_length, rng):
    """Run one episode and return rendered frames."""
    data = mujoco.MjData(model)

    # Reset to home keyframe
    home_qpos = model.keyframe("home").qpos.copy()
    data.qpos[:] = home_qpos
    data.qvel[:] = 0.0
    data.ctrl[:] = home_qpos[7:]
    mujoco.mj_forward(model, data)

    # Renderer
    renderer = mujoco.Renderer(model, FLAGS.render_height, FLAGS.render_width)

    frames = []
    last_act = np.zeros(12, dtype=np.float32)

    for step in range(episode_length):
        obs = get_obs(model, data, default_pose, last_act, command)

        # Query FAC policy
        rng, action_rng = jax.random.split(rng)
        obs_jax = jnp.array(obs[None])  # (1, 48)
        action = agent.sample_actions(obs_jax, seed=action_rng, temperature=0.0)
        action = np.array(action[0])  # (12,)
        action = np.clip(action, -1.0, 1.0)

        # PD target = default_pose + action * action_scale
        motor_targets = default_pose + action * ACTION_SCALE
        data.ctrl[:] = motor_targets

        # Step simulation
        for _ in range(N_SUBSTEPS):
            mujoco.mj_step(model, data)

        last_act = action.astype(np.float32)

        # Render
        renderer.update_scene(data, camera="track")
        frame = renderer.render()
        frames.append(frame.copy())

        # Check termination: robot fell over (upvector z < 0)
        upvec_adr = model.sensor_adr[model.sensor("upvector").id]
        upvec_z = data.sensordata[upvec_adr + 2]
        if upvec_z < 0.0:
            break

    renderer.close()
    return frames


def main(_):
    # Load flags from checkpoint
    flags_path = os.path.join(FLAGS.restore_path, "flags.json")
    with open(flags_path, "r") as f:
        saved_flags = json.load(f)

    seed = saved_flags["seed"]
    config = saved_flags["agent"]

    # Load Go2 MuJoCo model with training PD gains
    model = mujoco.MjModel.from_xml_path(GO2_XML)
    model.opt.timestep = SIM_DT
    model.dof_damping[6:] = KD
    model.actuator_gainprm[:, 0] = KP
    model.actuator_biasprm[:, 1] = -KP

    default_pose = model.keyframe("home").qpos[7:].copy()

    # Create agent from a dummy batch (need correct shapes)
    obs_dim = 48
    act_dim = 12
    dummy_obs = np.zeros((1, obs_dim), dtype=np.float32)
    dummy_act = np.zeros((1, act_dim), dtype=np.float32)

    agent_class = agents[config["agent_name"]]
    agent = agent_class.create(seed, dummy_obs, dummy_act, config)
    agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    command = np.array(
        [FLAGS.forward_cmd, FLAGS.lateral_cmd, FLAGS.yaw_cmd], dtype=np.float32
    )

    print(f"Command: forward={command[0]}, lateral={command[1]}, yaw={command[2]}")
    print(f"Running {FLAGS.num_episodes} episodes, max {FLAGS.episode_length} steps each...")

    rng = jax.random.PRNGKey(seed)
    all_frames = []

    for ep in range(FLAGS.num_episodes):
        rng, ep_rng = jax.random.split(rng)
        frames = run_episode(
            model, agent, config, default_pose, command, FLAGS.episode_length, ep_rng
        )
        all_frames.extend(frames)
        print(f"  Episode {ep + 1}: {len(frames)} steps")

    # Write video
    writer = imageio.get_writer(FLAGS.output, fps=FLAGS.fps)
    for frame in all_frames:
        writer.append_data(frame)
    writer.close()

    print(f"Saved {len(all_frames)} frames to {FLAGS.output}")


if __name__ == "__main__":
    app.run(main)
