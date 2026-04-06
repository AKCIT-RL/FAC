#!/bin/bash

MUJOCO_DIR=~/.mujoco
mkdir -p $MUJOCO_DIR

# # 200
# wget https://roboti.us/download/mujoco200_linux.zip -O $MUJOCO_DIR/mujoco200.zip
# unzip $MUJOCO_DIR/mujoco200.zip
# wget https://roboti.us/file/mjkey.txt -P $MUJOCO_DIR
# rm $MUJOCO_DIR/mujoco200.zip

# 210
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O $MUJOCO_DIR/mujoco210.tar.gz
tar -xzf $MUJOCO_DIR/mujoco210.tar.gz -C $MUJOCO_DIR
rm $MUJOCO_DIR/mujoco210.tar.gz
