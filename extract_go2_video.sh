#!/bin/bash

RESTORE_PATH="mujoco_scene/sd000_20260326_095529"
POLICY_NUM=200000

uv run extract_go2_video.py \
    --restore_path "$RESTORE_PATH" \
    --restore_epoch $POLICY_NUM \
    --num_episodes 3 \
    --forward_cmd 1.0 \
    --output go2_policy_video_${POLICY_NUM}.mp4
