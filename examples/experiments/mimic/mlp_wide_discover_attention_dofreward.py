# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Mimic Environment — "Discover" Stage: Self-Attention Architecture + Shape-Invariant DOF-Space
Reward, Combined

Combines two independently-confirmed levers on top of `mlp_wide_discover.py`'s relaxed
termination/reward baseline, for the first time together:
  - The self-attention temporal-encoder architecture from `mlp_wide_discover_attention.py`
    (confirmed on the full 150-clip corpus in note/README.note.md §61: 84.7% eval/success_rate
    and still climbing vs. baseline's 72% at matched step count, smoothest loss curves in the
    lineage). Kept as-is, including its historical/lookahead observation window
    (`HISTORY_STEPS`/`FUTURE_STEPS`) -- attention needs a multi-step token sequence to attend
    over, so this obs change travels with the architecture change, exactly as it did in the
    original attention experiment (its own env_config() is NOT byte-identical to
    `mlp_wide_discover.py`'s -- it's built on top of `mlp_wide_discover_historical_lookahead.py`).
  - The shape-invariant DOF-space tracking reward/eval-failure criterion from
    `mlp_wide_discover_dofreward.py` (note/README.note.md §65): `mimic_dof_tracking_rewards_factory`
    (dp/dv/contact_match/heading/rh) replaces `mimic_tracking_rewards_factory` (gt/gr/gv/gav/rh),
    and `dp_error` (mean DOF-angle error, radians, threshold=0.35) is logged alongside the
    existing `gt_error` (mean world-space position error, meters, threshold=0.5) in the
    evaluator, rather than replacing it, so both failure curves are visible on the same run.

Meant to run against the canonical (AMASS-only, no HUMOS) corpus, same as
`mlp_wide_discover_dofreward.py` -- since `dof_pos` is exactly shape-invariant, the canonical
single-theta(t)-per-clip corpus is a complete target for all 128 shapes under this reward, no
HUMOS blending needed.

**Known open item, not applied here (see note/README.note.md §62):** the attention architecture
has a positional-encoding gap -- the Transformer has no explicit position embedding, so token
order within the history/future windows isn't distinguished beyond content differences and the
current-state token's first-position pooling anchor. Flagged as a candidate fix before a
full-scale launch, not yet implemented; carried over unchanged from `mlp_wide_discover_attention.py`.

Everything else -- attention-stage sizing (`TOKEN_SIZE=256`, `NUM_HEADS=4`, `FF_SIZE=1024`,
`NUM_ATTN_LAYERS=2`), final-head width/depth (`WIDE_UNITS=2896 x 6` actor, `1024 x 4` critic),
termination (`fall_term_factory` only), optimizers, RSI, `configure_robot_and_simulator` --
byte-identical to `mlp_wide_discover_attention.py`.

    nohup python -u protomotions/train_agent.py \\
    --robot-name smpl_mor --simulator isaacgym \\
    --experiment-path examples/experiments/mimic/mlp_wide_discover_attention_dofreward.py \\
    --experiment-name hhi_wide_150motion_128shape_discover_attention_dofreward \\
    --motion-file /workspace/motion_cache/150_128shape_canonical/150_128shape_canonical_offset.pt \\
    --num-envs 6144 --batch-size 24576 --ngpu 1 \\
    --use-wandb --wandb-project hhi-protomotions --wandb-entity yugoamaryl \\
    --wandb-group hhi_wide_150motion_128shape_discover_attention_dofreward > /tmp/hhi_wide_150motion_128shape_discover_attention_dofreward.log 2>&1 &
"""
from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig
from protomotions.components.terrains.config import TerrainConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.agents.ppo.config import PPOAgentConfig
from protomotions.components.scene_lib import SceneLibConfig
from protomotions.components.motion_lib import MotionLibConfig
import argparse

WIDE_UNITS = 2896  # 1024 * sqrt(8) -- same as mlp_wide.py, kept for architecture parity

# Dilated backward history steps (actor + critic observation). Same schedule already validated
# in examples/experiments/amp/mlp.py, mlp_wide_amp.py, and mlp_wide_discover_historical.py.
HISTORY_STEPS = [1, 2, 3, 4, 8, 16, 32]

# Dilated forward-lookahead steps for the mimic target-pose observation. Same schedule already
# validated in examples/experiments/mimic/mlp_bm_l2c2.py and mlp_wide_discover_lookahead.py.
FUTURE_STEPS = [1, 2, 4, 8]

NUM_HISTORY_STEPS = len(HISTORY_STEPS)  # 7
NUM_FUTURE_STEPS = len(FUTURE_STEPS)  # 4

# Self-attention stage sizing -- first guess, not tuned (see mlp_wide_discover_attention.py).
TOKEN_SIZE = 256
TOKEN_ENCODER_WIDTH = 256
NUM_HEADS = 4
FF_SIZE = 1024
NUM_ATTN_LAYERS = 2


def terrain_config(args: argparse.Namespace):
    return TerrainConfig()


def scene_lib_config(args: argparse.Namespace):
    scene_file = args.scenes_file if hasattr(args, "scenes_file") else None
    return SceneLibConfig(scene_file=scene_file)


def motion_lib_config(args: argparse.Namespace):
    return MotionLibConfig(motion_file=args.motion_file)


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    from protomotions.envs.motion_manager.config import MimicMotionManagerConfig
    from protomotions.envs.control.mimic_control import MimicControlConfig
    from protomotions.envs.component_factories import (
        max_coords_obs_factory,
        historical_max_coords_obs_factory,
        previous_actions_factory,
        mimic_target_poses_max_coords_factory,
        mimic_dof_tracking_rewards_factory,
        fall_term_factory,
        morphology_obs_factory,
    )
    from protomotions.envs.action import make_pd_action_config

    control_components = {
        "mimic": MimicControlConfig(
            bootstrap_on_episode_end=True,
            future_steps=FUTURE_STEPS,
        )
    }

    observation_components = {
        "max_coords_obs": max_coords_obs_factory(),
        # Backward: dilated window of past body-state frames.
        "historical_max_coords_obs": historical_max_coords_obs_factory(
            local_obs=True,
            root_height_obs=True,
            observe_contacts=False,
            history_steps=HISTORY_STEPS,
        ),
        "previous_actions": previous_actions_factory(history_steps=1),
        # Forward: same factory call as discover/lookahead -- resolves to "all available future
        # steps," which now means the FUTURE_STEPS-dilated window instead of a single next-step
        # frame.
        "mimic_target_poses": mimic_target_poses_max_coords_factory(with_velocities=True),
        "morphology_obs": morphology_obs_factory(),
    }

    # Sole termination is a genuine fall/collapse check -- no tracking-error termination at all,
    # so an ugly-but-upright trajectory is never cut short just for drifting from the reference.
    termination_components = {
        "fall": fall_term_factory(termination_height=0.15),
    }

    # DOF-space (shape-invariant) tracking reward, replacing world-space gt/gr/gv/gav -- see
    # mlp_wide_discover_dofreward.py / note/README.note.md Section 65. Weights are a first-guess,
    # combining discover.py's split position+rotation / velocity+angular-velocity weights into
    # one DOF-angle position/velocity term each.
    reward_components = {
        **mimic_dof_tracking_rewards_factory(
            dp_weight=0.8,
            dv_weight=0.3,
            contact_weight=-0.1,
            heading_weight=0.1,
            rh_weight=0.2,
            dp_coef=-40.0,
            dv_coef=-0.5,
            heading_coef=-10.0,
            rh_coef=-100.0,
        ),
    }

    return EnvConfig(
        ref_contact_smooth_window=7,
        max_episode_length=1000,
        num_state_history_steps=max(HISTORY_STEPS),
        control_components=control_components,
        observation_components=observation_components,
        termination_components=termination_components,
        reward_components=reward_components,
        action_config=make_pd_action_config(robot_cfg),
        motion_manager=MimicMotionManagerConfig(
            init_start_prob=0.2,
            resample_on_reset=True,
        ),
    )


def agent_config(
    robot_config: RobotConfig, env_config: EnvConfig, args: argparse.Namespace
) -> PPOAgentConfig:
    from protomotions.agents.common.config import (
        MLPWithConcatConfig,
        MLPLayerConfig,
        ModuleContainerConfig,
        TransformerConfig,
        ModuleOperationReshapeConfig,
        ModuleOperationForwardConfig,
    )
    from protomotions.agents.ppo.config import (
        PPOActorConfig,
        PPOModelConfig,
        AdvantageNormalizationConfig,
    )
    from protomotions.agents.base_agent.config import OptimizerConfig
    from protomotions.agents.evaluators.config import (
        MimicEvaluatorConfig,
        MotionWeightsRulesConfig,
    )
    from protomotions.envs.component_factories import (
        gt_error_factory,
        dp_error_factory,
        gr_error_factory,
        max_joint_error_factory,
    )

    raw_obs_keys = [
        "max_coords_obs",
        "historical_max_coords_obs",
        "mimic_target_poses",
        "previous_actions",
        "morphology_obs",
    ]

    def build_attention_trunk(final_layer_units, final_num_out, final_out_key):
        """Token-encode + self-attend the temporal obs, then feed a flat-MLP head.

        Final head width/depth is caller-supplied and kept identical to the flat-concat
        baseline's trunk -- the token encoders + transformer here are additive preprocessing,
        not a replacement for that capacity.
        """
        return ModuleContainerConfig(
            in_keys=raw_obs_keys,
            out_keys=[final_out_key],
            models=[
                # Per-frame token encoders. Reshape recovers the [batch, steps, dim] sequence
                # from the flat dilated-frame tensor; ModuleOperationForwardConfig then applies
                # the (shared-weight) MLP per-frame automatically for 3D inputs.
                MLPWithConcatConfig(
                    in_keys=["max_coords_obs"],
                    out_keys=["current_state_token"],
                    normalize_obs=True,
                    norm_clamp_value=5,
                    num_out=TOKEN_SIZE,
                    layers=[
                        MLPLayerConfig(units=TOKEN_ENCODER_WIDTH, activation="relu")
                        for _ in range(2)
                    ],
                    module_operations=[
                        ModuleOperationReshapeConfig(new_shape=["batch_size", 1, -1]),
                        ModuleOperationForwardConfig(),
                    ],
                ),
                MLPWithConcatConfig(
                    in_keys=["historical_max_coords_obs"],
                    out_keys=["history_token"],
                    normalize_obs=True,
                    norm_clamp_value=5,
                    num_out=TOKEN_SIZE,
                    layers=[
                        MLPLayerConfig(units=TOKEN_ENCODER_WIDTH, activation="relu")
                        for _ in range(2)
                    ],
                    module_operations=[
                        ModuleOperationReshapeConfig(
                            new_shape=["batch_size", NUM_HISTORY_STEPS, -1]
                        ),
                        ModuleOperationForwardConfig(),
                    ],
                ),
                MLPWithConcatConfig(
                    in_keys=["mimic_target_poses"],
                    out_keys=["future_token"],
                    normalize_obs=True,
                    norm_clamp_value=5,
                    num_out=TOKEN_SIZE,
                    layers=[
                        MLPLayerConfig(units=TOKEN_ENCODER_WIDTH, activation="relu")
                        for _ in range(2)
                    ],
                    module_operations=[
                        ModuleOperationReshapeConfig(
                            new_shape=["batch_size", NUM_FUTURE_STEPS, -1]
                        ),
                        ModuleOperationForwardConfig(),
                    ],
                ),
                # Self-attention over the 12-token sequence (1 current + 7 history + 4 future).
                # current_state_token is listed first -- Transformer pools output[:, 0, :].
                TransformerConfig(
                    in_keys=["current_state_token", "history_token", "future_token"],
                    out_keys=["attn_out"],
                    transformer_token_size=TOKEN_SIZE,
                    latent_dim=TOKEN_SIZE,
                    num_heads=NUM_HEADS,
                    ff_size=FF_SIZE,
                    num_layers=NUM_ATTN_LAYERS,
                    output_activation="relu",
                ),
                # Final head -- width/depth supplied by caller, unchanged from the flat-concat
                # baseline. previous_actions/morphology_obs are single-vector, not sequential,
                # so they bypass attention and are concatenated directly here, same as they're
                # concatenated into the flat trunk today.
                MLPWithConcatConfig(
                    in_keys=["attn_out", "previous_actions", "morphology_obs"],
                    out_keys=[final_out_key],
                    normalize_obs=True,
                    norm_clamp_value=5,
                    num_out=final_num_out,
                    layers=[
                        MLPLayerConfig(units=units, activation="relu")
                        for units in final_layer_units
                    ],
                ),
            ],
        )

    actor_config = PPOActorConfig(
        num_out=robot_config.kinematic_info.num_dofs,
        actor_logstd=-2.9,
        in_keys=raw_obs_keys,
        mu_key="actor_trunk_out",
        mu_model=build_attention_trunk(
            final_layer_units=[WIDE_UNITS] * 6,
            final_num_out=robot_config.number_of_actions,
            final_out_key="actor_trunk_out",
        ),
    )

    critic_config = build_attention_trunk(
        final_layer_units=[1024] * 4,
        final_num_out=1,
        final_out_key="value",
    )

    agent_config: PPOAgentConfig = PPOAgentConfig(
        model=PPOModelConfig(
            in_keys=raw_obs_keys,
            out_keys=["action", "mean_action", "neglogp", "value"],
            actor=actor_config,
            critic=critic_config,
            actor_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=2e-5),
            critic_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=1e-4),
        ),
        batch_size=args.batch_size,
        training_max_steps=args.training_max_steps,
        gradient_clip_val=50.0,
        clip_critic_loss=True,
        evaluator=MimicEvaluatorConfig(
            evaluation_components={
                "gt_error": gt_error_factory(threshold=0.5),
                "dp_error": dp_error_factory(threshold=0.35),
                "gr_error": gr_error_factory(),
                "max_joint_error": max_joint_error_factory(),
            },
            motion_weights_rules=MotionWeightsRulesConfig(
                motion_weights_update_success_discount=0.999,
                motion_weights_update_failure_discount=0,
            ),
        ),
        advantage_normalization=AdvantageNormalizationConfig(
            enabled=True, shift_mean=True, use_ema=True
        ),
    )
    return agent_config


def configure_robot_and_simulator(
    robot_cfg: RobotConfig, simulator_cfg: SimulatorConfig, args: argparse.Namespace
):
    """Configure robot to add contact sensors for foot contact tracking."""
    robot_cfg.update_fields(
        contact_bodies=["all_left_foot_bodies", "all_right_foot_bodies"]
    )


def apply_inference_overrides(
    robot_cfg: RobotConfig,
    simulator_cfg: SimulatorConfig,
    env_cfg,
    agent_cfg,
    terrain_cfg: TerrainConfig,
    motion_lib_cfg: MotionLibConfig,
    scene_lib_cfg: SceneLibConfig,
    args: argparse.Namespace,
):
    """Apply evaluation-specific overrides."""
    if hasattr(env_cfg, "termination_components") and env_cfg.termination_components:
        env_cfg.termination_components = {}

    env_cfg.max_episode_length = 1000000
    env_cfg.motion_manager.resample_on_reset = True
    env_cfg.motion_manager.init_start_prob = 1.0
