#!/usr/bin/env bash
set -e

MOTION_FILE="/home/hlz/datasets/humos_proto_motionlib/humos_1.pt"

python protomotions/train_agent.py \
    --robot-name hhi_smpl_single \
    --simulator isaacgym \
    --experiment-path examples/experiments/mimic/mlp.py \
    --experiment-name hhi_single_male_0e26b88d_lowmem \
    --motion-file /home/hlz/datasets/humos_proto_motionlib/humos_1.pt \
    --num-envs 2 \
    --batch-size 2 \
    --headless True


python protomotions/train_agent.py \
    --robot-name hhi_smpl_single \
    --simulator isaacgym \
    --experiment-path examples/experiments/mimic/mlp.py \
    --experiment-name hhi_single_male_0e26b88d_cloud_smoke \
    --motion-file /home/hlz/datasets/humos_proto_motionlib/humos_1.pt \
    --num-envs 2 \
    --batch-size 2 \
    --training-max-steps 100000 \
    --use-wandb \
    --wandb-project hhi-protomotions \
    --wandb-entity yugoamaryl \
    --wandb-group single_shape_smoke \
    --wandb-tags local lowmem single_motion male_0e26b88d \
    --headless True

# --num-envs 4096 --batch-size 16384