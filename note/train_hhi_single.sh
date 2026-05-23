#!/usr/bin/env bash
set -e

MOTION_FILE="/home/hlz/datasets/humos_proto/humos_8.pt"

python protomotions/train_agent.py \
    --robot-name smpl_mor \
    --simulator isaacgym \
    --experiment-path examples/experiments/mimic/mlp.py \
    --experiment-name local_test \
    --motion-file /home/hlz/datasets/humos_proto/humos_8.pt \
    --num-envs 16 \
    --batch-size 32

