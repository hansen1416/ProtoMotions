#!/usr/bin/env bash
set -e

MOTION_FILE="/home/hlz/datasets/humos_proto_motionlib/humos_1.pt"

python protomotions/train_agent.py \
    --robot-name hhi_smpl_single \
    --simulator isaacgym \
    --experiment-path examples/experiments/mimic/mlp.py \
    --experiment-name hhi_single_male_0e26b88d_lowmem \
    --motion-file /home/hlz/datasets/humos_proto_motionlib/humos_1.pt \
    --num-envs 16 \
    --batch-size 32

python protomotions/train_agent.py \
    --robot-name smpl_mor \
    --simulator isaacgym \
    --experiment-path examples/experiments/mimic/mlp.py \
    --experiment-name hhi_single_male_0e26b88d_lowmem \
    --motion-file /home/hlz/datasets/humos_proto_motionlib/humos_128.pt \
    --num-envs 16 \
    --batch-size 32

