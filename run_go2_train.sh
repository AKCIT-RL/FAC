#!/bin/bash

  uv run train_go2_fac.py \
      --reward_scale 100.0 \
      --save_dir mujoco_scene/ \
      --save_interval 100000 \
      --agent.fac_alpha=0.5 \
      --agent.fac_lambda=0.5 \
      --agent.q_agg=mean \
      --agent.discount=0.99
