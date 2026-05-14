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
    --batch-size 32 \
    --headless True

python protomotions/train_agent.py \
    --robot-name smpl_mor \
    --simulator isaacgym \
    --experiment-path examples/experiments/mimic/mlp.py \
    --experiment-name hhi_single_male_0e26b88d_lowmem \
    --motion-file /home/hlz/datasets/humos_proto_motionlib/humos_8.pt \
    --num-envs 16 \
    --batch-size 32 \
    --headless False

python protomotions/train_agent.py \
    --robot-name hhi_smpl_single \
    --simulator isaacgym \
    --experiment-path examples/experiments/mimic/mlp.py \
    --experiment-name hhi_single_male_0e26b88d_cloud_smoke \
    --motion-file ./humos_1.pt \
    --num-envs 16 \
    --batch-size 128 \
    --use-wandb \
    --wandb-project hhi-protomotions \
    --wandb-entity yugoamaryl \
    --wandb-group single_shape_smoke \
    --wandb-tags local lowmem single_motion male_0e26b88d \
    --headless True

# git clone -b feature/hhi https://github.com/hansen1416/ProtoMotions.git

# pip install gdown
# gdown 1eDoIKUGs8VYXQY1RJskSEJk5WSLUBNhN

# pip install -e .

# wandb login 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


# apt update && apt install -y tmux
# tmux new -s hhi
# tmux kill-session -t hhi



# wandb login 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



python data/scripts/convert_amass_to_proto.py \
    /home/hlz/datasets/amass_data/ \
    --humanoid-type smpl \
    --output-fps 30 \
    --motion-config /home/hlz/datasets/wb13.yaml \
    --force-remake
