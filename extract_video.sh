#!/bin/bash

POLICY_NUM=1000000
uv run extract_policy_video.py --restore_path "/home/joao/Projetos/AKCIT_RL/FAC/ogbench_scene/Debug/Debug/sd000_20260310_093026" --restore_epoch $POLICY_NUM --num_episodes 10 --output policy_video_${POLICY_NUM}.mp4
