import json
import os

import imageio
import jax
import numpy as np
from absl import app, flags

from agents import agents
from envs.env_utils import make_env_and_datasets
from utils.datasets import Dataset, ReplayBuffer
from utils.evaluation import evaluate
from utils.flax_utils import restore_agent

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "restore_path", None, "Path to the checkpoint directory.", required=True
)
flags.DEFINE_integer("restore_epoch", None, "Checkpoint epoch to load.", required=True)
flags.DEFINE_integer("num_episodes", 10, "Number of episodes to record.")
flags.DEFINE_integer("video_frame_skip", 3, "Frame skip for video rendering.")
flags.DEFINE_string("output", "policy_video.mp4", "Output video filename.")
flags.DEFINE_integer("fps", 30, "Video frames per second.")


def main(_):
    # Load flags from checkpoint dir to recover env_name, seed, and agent config.
    flags_path = os.path.join(FLAGS.restore_path, "flags.json")
    with open(flags_path, "r") as f:
        saved_flags = json.load(f)

    env_name = saved_flags["env_name"]
    seed = saved_flags["seed"]
    config = saved_flags["agent"]

    assert "singletask" in env_name, (
        "Video rendering is only supported for OGBench singletask environments."
    )

    # Build env and get an example batch to initialize the agent.
    _, eval_env, train_dataset, _ = make_env_and_datasets(env_name)
    train_dataset = Dataset.create(**train_dataset)
    train_dataset = ReplayBuffer.create_from_initial_dataset(
        dict(train_dataset), size=train_dataset.size + 1
    )
    example_batch = train_dataset.sample(1)

    # Initialize and restore agent.
    agent_class = agents[config["agent_name"]]
    agent = agent_class.create(
        seed,
        example_batch["observations"],
        example_batch["actions"],
        config,
    )
    agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    # Run evaluation episodes with rendering.
    _, _, renders = evaluate(
        agent=agent,
        env=eval_env,
        config=config,
        num_eval_episodes=0,
        num_video_episodes=FLAGS.num_episodes,
        video_frame_skip=FLAGS.video_frame_skip,
    )

    # Save all episode frames to a single video file.
    writer = imageio.get_writer(FLAGS.output, fps=FLAGS.fps)
    for episode_frames in renders:
        for frame in episode_frames:
            writer.append_data(frame)
    writer.close()

    print(f"Saved {FLAGS.num_episodes} episode(s) to {FLAGS.output}")


if __name__ == "__main__":
    app.run(main)
