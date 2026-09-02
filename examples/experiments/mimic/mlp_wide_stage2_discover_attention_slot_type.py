# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""Full-dataset discover attention with learned temporal slot/type embeddings.

This is the GlobalClipPool counterpart of
``mlp_wide_discover_attention_slot_type.py``. It inherits the full Stage-2 data
rotation, versioned global split, observations, rewards, PPO settings, and transformer
capacity from ``mlp_wide_stage2_discover_attention.py``. It enables learned
temporal-slot and token-source-type embeddings in the actor and critic transformers
and evaluates a fixed four-shape panel per clip.

Full-scale launch:

    nohup python -u protomotions/train_agent.py \
    --robot-name smpl_mor --simulator isaacgym \
    --experiment-path examples/experiments/mimic/mlp_wide_stage2_discover_attention_slot_type.py \
    --experiment-name hhi_wide_stage2_discover_attention_slot_type \
    --global-clip-pool-source r2:proto-data/hhi_stage2_per_clip_refined/ \
    --global-clip-pool-cache-dir /workspace/motion_cache \
    --global-clip-pool-size 256 --global-clip-pool-rebuild-every 256 \
    --global-clip-pool-weight-floor 0.05 \
    --global-clip-pool-random-fraction 0.2 \
    --num-envs 6144 --batch-size 24576 --ngpu 6 \
    --use-wandb --wandb-project hhi-protomotions --wandb-entity yugoamaryl \
    --wandb-group hhi_wide_stage2_discover_attention_slot_type \
    > /tmp/hhi_wide_stage2_discover_attention_slot_type.log 2>&1 &
"""

import argparse

from protomotions.agents.common.config import TransformerConfig
from protomotions.agents.ppo.config import PPOAgentConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.robot_configs.base import RobotConfig

from examples.experiments.mimic.mlp_wide_stage2_discover_attention import (
    FUTURE_STEPS,
    HISTORY_STEPS,
    additional_experiment_arguments,
    apply_inference_overrides,
    configure_robot_and_simulator,
    env_config,
    motion_lib_config,
    scene_lib_config,
    terrain_config,
)
from examples.experiments.mimic.mlp_wide_stage2_discover_attention import (
    agent_config as base_agent_config,
)


NUM_TEMPORAL_TOKENS = 1 + len(HISTORY_STEPS) + len(FUTURE_STEPS)

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


def agent_config(
    robot_config: RobotConfig, env_config: EnvConfig, args: argparse.Namespace
) -> PPOAgentConfig:
    """Enable slot/type identity on the unchanged full-scale attention policy."""
    config = base_agent_config(robot_config, env_config, args)

    trunks = [config.model.actor.mu_model, config.model.critic]
    for trunk_name, trunk in zip(("actor", "critic"), trunks):
        transformer_configs = [
            model for model in trunk.models if isinstance(model, TransformerConfig)
        ]
        assert len(transformer_configs) == 1, (
            f"Expected exactly one TransformerConfig in the {trunk_name} trunk, "
            f"found {len(transformer_configs)}"
        )
        transformer_config = transformer_configs[0]
        transformer_config.use_learned_slot_embeddings = True
        transformer_config.max_sequence_length = NUM_TEMPORAL_TOKENS
        transformer_config.use_learned_token_type_embeddings = True

    # Compare checkpoints on the same deterministic morphology panel.
    config.evaluator.eval_shapes_per_motion = 4
    config.evaluator.eval_shape_sampling_seed = 42

    return config
