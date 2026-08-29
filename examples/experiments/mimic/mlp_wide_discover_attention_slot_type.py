# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""Discover attention with learned temporal-slot and token-type embeddings.

This is a controlled extension of ``mlp_wide_discover_attention.py``. The environment,
observations, rewards, token encoders, transformer capacity, MLP heads, PPO settings, and
morphology concatenation are unchanged. The only architectural change is that both actor and
critic transformers receive:

* one learned embedding for each of the 12 temporal slots (current; history at
  ``[-1, -2, -3, -4, -8, -16, -32]``; future at ``[1, 2, 4, 8]``); and
* one learned type embedding for each token source (current, history, future).

These embeddings remove the original transformer's permutation ambiguity while preserving the
current-state token at sequence index zero as the pooling anchor. ``morphology_obs`` and
``previous_actions`` continue to bypass attention and enter the final MLP head directly.

Recommended refined-motion run:

    nohup python -u protomotions/train_agent.py \
    --robot-name smpl_mor --simulator isaacgym \
    --experiment-path examples/experiments/mimic/mlp_wide_discover_attention_slot_type.py \
    --experiment-name hhi_wide_150motion_discover_attention_slot_type_refined \
    --motion-file /workspace/motion_cache/small150_128shape_refined.pt \
    --num-envs 4096 --batch-size 16384 --ngpu 1 \
    --use-wandb --wandb-project hhi-protomotions --wandb-entity yugoamaryl \
    --wandb-group hhi_wide_150motion_discover_attention_slot_type_refined \
    > /tmp/hhi_wide_150motion_discover_attention_slot_type_refined.log 2>&1 &
"""

import argparse

from protomotions.agents.common.config import TransformerConfig
from protomotions.agents.ppo.config import PPOAgentConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.robot_configs.base import RobotConfig

from examples.experiments.mimic.mlp_wide_discover_attention import (
    FUTURE_STEPS,
    HISTORY_STEPS,
    apply_inference_overrides,
    configure_robot_and_simulator,
    env_config,
    motion_lib_config,
    scene_lib_config,
    terrain_config,
)
from examples.experiments.mimic.mlp_wide_discover_attention import (
    agent_config as base_agent_config,
)

NUM_TEMPORAL_TOKENS = 1 + len(HISTORY_STEPS) + len(FUTURE_STEPS)

__all__ = [
    "terrain_config",
    "scene_lib_config",
    "motion_lib_config",
    "env_config",
    "configure_robot_and_simulator",
    "apply_inference_overrides",
    "agent_config",
]


def agent_config(
    robot_config: RobotConfig, env_config: EnvConfig, args: argparse.Namespace
) -> PPOAgentConfig:
    """Enable learned slot/type identity on the otherwise unchanged attention baseline."""
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

    return config
