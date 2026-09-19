# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""Short full-dataset polish pass on top of the released checkpoint.

Identical to ``mlp_wide_stage2_discover_attention_slot_type.py`` (same
observations, rewards, PPO settings, transformer capacity, slot/type
embeddings) with ONE change: ``motion_lib_config()`` disables the explicit
train/validation split and points GlobalClipPool at
``data/splits/hhi_stage2_v1/full_manifest.jsonl`` -- the union of the
train/validation/test manifests (20,951 clips, no overlap; verified via
``cat train+validation+test > full_manifest.jsonl``, deduplication checked).

This checkpoint is NOT a replacement for
``hhi_wide_stage2_discover_attention_slot_type_refined`` (which stays frozen
as the checkpoint backing the paper's reported validation/test numbers --
those numbers depend on this checkpoint never having trained on that data).
This fork exists purely to feed the downstream text-to-motion dataset
generation pass: every clip the policy will ever need to render gets to be
"trained data" for this checkpoint, so there's no more train/val/test
distinction left to protect.

Warm-start (NOT resume) from the existing checkpoint, into a NEW experiment
name, for a short run (a "polish pass", not a fresh trained-from-scratch
run):

    nohup python -u protomotions/train_agent.py \
    --robot-name smpl_mor --simulator isaacgym \
    --experiment-path examples/experiments/mimic/mlp_wide_stage2_discover_attention_slot_type_fulldata.py \
    --experiment-name hhi_wide_stage2_discover_attention_slot_type_fulldata \
    --checkpoint results/hhi_wide_stage2_discover_attention_slot_type_refined/last.ckpt \
    --global-clip-pool-source r2:proto-data/hhi_stage2_per_clip_refined/ \
    --global-clip-pool-cache-dir /workspace/motion_cache \
    --global-clip-pool-size 256 --global-clip-pool-rebuild-every 256 \
    --global-clip-pool-weight-floor 0.05 \
    --global-clip-pool-random-fraction 0.2 \
    --num-envs 6144 --batch-size 24576 --ngpu 6 \
    --training-max-steps <SET ME: a short polish budget, e.g. a few thousand PPO steps> \
    --use-wandb --wandb-project hhi-protomotions --wandb-entity yugoamaryl \
    --wandb-group hhi_wide_stage2_discover_attention_slot_type_fulldata \
    > /tmp/hhi_wide_stage2_discover_attention_slot_type_fulldata.log 2>&1 &

Before launching: upload the merged manifest alongside the existing split
artifacts (same R2 prefix, so no new --global-clip-pool-source needed):

    rclone copy data/splits/hhi_stage2_v1/full_manifest.jsonl \
        r2:proto-data/hhi_stage2_per_clip_refined/splits/hhi_stage2_v1/

--training-max-steps has no principled value yet -- pick something like
2000-5000 PPO update steps (versus tens of thousands for the original
full-scale run) and watch wandb; this is meant to be short.
"""

import argparse

from protomotions.envs.base_env.config import EnvConfig
from protomotions.robot_configs.base import RobotConfig

from examples.experiments.mimic.mlp_wide_stage2_discover_attention_slot_type import (
    additional_experiment_arguments,
    agent_config,
    apply_inference_overrides,
    configure_robot_and_simulator,
    env_config,
    scene_lib_config,
    terrain_config,
)

__all__ = [
    "terrain_config",
    "scene_lib_config",
    "motion_lib_config",
    "env_config",
    "additional_experiment_arguments",
    "configure_robot_and_simulator",
    "apply_inference_overrides",
    "agent_config",
]

FULL_MANIFEST_NAME = "splits/hhi_stage2_v1/full_manifest.jsonl"


def motion_lib_config(args: argparse.Namespace):
    if getattr(args, "global_clip_pool_source", None):
        from protomotions.components.global_clip_pool import GlobalClipPoolConfig

        return GlobalClipPoolConfig(
            r2_source=args.global_clip_pool_source,
            manifest_name=FULL_MANIFEST_NAME,
            # Explicit split mode is intentionally disabled (both None together --
            # see global_clip_pool.py's `explicit_split` check) since every clip in
            # full_manifest.jsonl is now fair game for training; there is nothing
            # left to hold out for eval_holdout to legitimately measure.
            validation_manifest_name=None,
            split_metadata_name=None,
            local_cache_dir=args.global_clip_pool_cache_dir,
            resident_pool_size=args.global_clip_pool_size,
            cache_size_multiplier=args.global_clip_pool_cache_multiplier,
            pool_rebuild_every=args.global_clip_pool_rebuild_every,
            clip_partition_shuffle_seed=args.global_clip_pool_shuffle_seed,
            exploration_bonus_coefficient=args.global_clip_pool_exploration_coefficient,
            selection_temperature=args.global_clip_pool_selection_temperature,
            weight_ema_alpha=args.global_clip_pool_weight_ema_alpha,
            difficulty_scores_path=args.global_clip_pool_difficulty_scores_path,
            eval_holdout_size=0,
            weight_floor=args.global_clip_pool_weight_floor,
            random_fraction=args.global_clip_pool_random_fraction,
        )
    from protomotions.components.motion_lib import MotionLibConfig

    return MotionLibConfig(motion_file=args.motion_file)
