# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""One-stage 150-motion attention experiment with actor-only morphology adaLN-Zero.

All 128 body shapes and their corresponding refined motions are trained jointly from scratch.
This builds on ``mlp_wide_discover_attention_slot_type.py`` and changes only the actor
transformer: its two blocks use morphology-conditioned adaptive LayerNorm-Zero. The critic keeps
the ordinary slot/type transformer. Both final heads retain direct ``morphology_obs``
concatenation, and all rewards, observations, PPO settings, and model widths remain unchanged.

    nohup python -u protomotions/train_agent.py \
    --robot-name smpl_mor --simulator isaacgym \
    --experiment-path examples/experiments/mimic/mlp_wide_discover_attention_adaln.py \
    --experiment-name hhi_wide_150motion_discover_attention_adaln_refined \
    --motion-file /workspace/motion_cache/small150_128shape_refined.pt \
    --num-envs 4096 --batch-size 16384 --ngpu 1 \
    --use-wandb --wandb-project hhi-protomotions --wandb-entity yugoamaryl \
    --wandb-group hhi_wide_150motion_discover_attention_adaln_refined \
    > /tmp/hhi_wide_150motion_discover_attention_adaln_refined.log 2>&1 &
"""

import argparse

from protomotions.agents.common.config import TransformerConfig
from protomotions.agents.common.morphology_transformer import (
    MorphologyAdaLNZeroTransformerConfig,
)
from protomotions.agents.ppo.config import PPOAgentConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.robot_configs.base import RobotConfig

from examples.experiments.mimic.mlp_wide_discover_attention import (
    apply_inference_overrides,
    configure_robot_and_simulator,
    env_config,
    motion_lib_config,
    scene_lib_config,
    terrain_config,
)
from examples.experiments.mimic.mlp_wide_discover_attention_slot_type import (
    NUM_TEMPORAL_TOKENS,
)
from examples.experiments.mimic.mlp_wide_discover_attention_slot_type import (
    agent_config as slot_type_agent_config,
)

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
    """Replace only the actor transformer with morphology-conditioned adaLN-Zero."""
    config = slot_type_agent_config(robot_config, env_config, args)
    actor_trunk = config.model.actor.mu_model
    transformer_indices = [
        index
        for index, model in enumerate(actor_trunk.models)
        if isinstance(model, TransformerConfig)
    ]
    assert len(transformer_indices) == 1, (
        f"Expected one actor TransformerConfig, found {len(transformer_indices)}"
    )

    index = transformer_indices[0]
    baseline = actor_trunk.models[index]
    actor_trunk.models[index] = MorphologyAdaLNZeroTransformerConfig(
        in_keys=[*baseline.in_keys, "morphology_obs"],
        out_keys=baseline.out_keys,
        input_and_mask_mapping=baseline.input_and_mask_mapping,
        transformer_token_size=baseline.transformer_token_size,
        latent_dim=baseline.latent_dim,
        num_heads=baseline.num_heads,
        ff_size=baseline.ff_size,
        num_layers=baseline.num_layers,
        dropout=baseline.dropout,
        activation=baseline.activation,
        output_activation=baseline.output_activation,
        use_learned_slot_embeddings=True,
        max_sequence_length=NUM_TEMPORAL_TOKENS,
        use_learned_token_type_embeddings=True,
        condition_key="morphology_obs",
        condition_hidden_dim=128,
        beta_norm_scale=3.0,
    )
    return config
