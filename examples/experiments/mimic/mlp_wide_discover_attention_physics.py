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
Mimic Environment — "Discover" Stage: Self-Attention + Physics-Feature Morphology Conditioning
(150-clip corpus)

Clone of `mlp_wide_discover_attention.py` (self-attention temporal encoder over dilated
history/lookahead frames — see that file's docstring for the full architecture rationale and the
`TOKEN_SIZE`/`NUM_HEADS`/`FF_SIZE`/`NUM_ATTN_LAYERS` sizing discussion, unchanged here). That file
was subsequently also run on the 150-clip corpus as `hhi_wide_150motion_discover_attention`
(wandb `p6b2la4h`) and, checked against the rest of the 150-motion `discover` lineage at matched
step count, showed both better eval/success_rate (84.7% vs. baseline `discover`'s 72% at step
4799, still climbing rather than plateaued) and the smoothest actor/critic loss curves of the
whole lineage (~25-30% lower step-to-step noise than the next-best run). This file starts from
that same architecture and swaps one thing: the morphology conditioning input.

**Only change: `morphology_obs` (raw SMPL `[gender_id, betas]`, 11-dim, `morphology_obs_factory`)
-> `physics_obs` (z-scored physics features, 15-dim, `physics_obs_factory`)`, everywhere that key
is read (env_config's observation_components, agent_config's raw_obs_keys, and the final head's
`in_keys` in `build_attention_trunk`)**. Both factories already exist in
`protomotions/envs/component_factories.py`; `physics_obs_factory` reads `env_physics_features`
from context, which the isaacgym simulator (`protomotions/simulator/isaacgym/simulator.py:613`,
`_build_physics_features`) already builds automatically for `smpl_mor` multi-shape assets by
loading `protomotions/data/assets/mjcf/smpl_mor/physics_features.pt` — confirmed present and
covering all 128 shapes used by the current corpus (checked via direct load before writing this
file). No new tooling needed; this is the same physics-feature extraction
(`tools/extract_smpl_physics_features.py`: z-scored total mass, bilateral thigh/shin/upper-arm/
forearm lengths, torso height, neck-head height, hip width, shoulder width, leg length, total
height) used in the older `hhi_phy_1024_transfer` single-shape-neutral-pretrain experiment
(`note/README.prelim-report.md` §5), revisited here on the current wide+attention architecture and
the full 150-clip/128-shape corpus rather than the old 1-shape/1024-clip setup.

Motivation for retrying it now: the original `hhi_phy_1024_transfer` vs. `hhi_1024_transfer`
comparison was inconclusive on aggregate metrics (physics features helped 56 clips, hurt 46,
`README.prelim-report.md` §5) but showed a large qualitative gap in a small visual smoke test
(5/8 vs. 0/8 successful rollouts favoring physics features, `README.eval-plan.md`) that was never
followed up on with a wider ablation or the current architecture. Worth a clean re-test now that
there's a stronger base architecture (attention) to combine it with.

Same "first guess, not tuned, isolate one variable" convention as every other lever in this
lineage: nothing else changes relative to `mlp_wide_discover_attention.py` on the 150-clip corpus
(same `HISTORY_STEPS`/`FUTURE_STEPS`, same reward/termination/motion-manager config, same
attention-stage sizing, same final-head width/depth) — this run isolates morphology-conditioning
representation only.

    nohup python -u protomotions/train_agent.py \\
    --robot-name smpl_mor --simulator isaacgym \\
    --experiment-path examples/experiments/mimic/mlp_wide_discover_attention_physics.py \\
    --experiment-name hhi_wide_150motion_discover_attention_physics \\
    --motion-file /workspace/motion_cache/small150_128shape.pt \\
    --num-envs 4096 --batch-size 16384 --ngpu 1 \\
    --use-wandb --wandb-project hhi-protomotions --wandb-entity yugoamaryl \\
    --wandb-group hhi_wide_150motion_discover_attention_physics > /tmp/hhi_wide_150motion_discover_attention_physics.log 2>&1 &
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

# Self-attention stage sizing -- first guess, not tuned (see module docstring).
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
        mimic_tracking_rewards_factory,
        fall_term_factory,
        physics_obs_factory,
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
        # Physics-feature morphology conditioning (z-scored mass/limb-length/height features)
        # instead of raw betas -- the sole change from mlp_wide_discover_attention.py.
        "physics_obs": physics_obs_factory(),
    }

    # Sole termination is a genuine fall/collapse check -- no tracking-error termination at all,
    # so an ugly-but-upright trajectory is never cut short just for drifting from the reference.
    termination_components = {
        "fall": fall_term_factory(termination_height=0.15),
    }

    # Only the tracking-shaping reward remains -- action_smoothness/pow_rew/contact_match_rew
    # (effort, smoothness, contact-timing penalties) are omitted entirely for this stage.
    # Byte-identical to mlp_wide_discover.py's original coefficients -- this experiment isolates
    # the attention-vs-flat-concat encoding lever only.
    reward_components = {
        **mimic_tracking_rewards_factory(
            gt_weight=0.5,
            gr_weight=0.3,
            gv_weight=0.1,
            gav_weight=0.2,
            rh_weight=0.2,
            gt_coef=-25.0,
            gr_coef=-5.0,
            gv_coef=-0.5,
            gav_coef=-0.1,
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
        gr_error_factory,
        max_joint_error_factory,
    )

    raw_obs_keys = [
        "max_coords_obs",
        "historical_max_coords_obs",
        "mimic_target_poses",
        "previous_actions",
        "physics_obs",
    ]

    def build_attention_trunk(final_layer_units, final_num_out, final_out_key):
        """Token-encode + self-attend the temporal obs, then feed a flat-MLP head.

        Final head width/depth is caller-supplied and kept identical to the flat-concat
        baseline's trunk (see module docstring) -- the token encoders + transformer here are
        additive preprocessing, not a replacement for that capacity.
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
                # baseline. previous_actions/physics_obs are single-vector, not sequential,
                # so they bypass attention and are concatenated directly here, same as they're
                # concatenated into the flat trunk today.
                MLPWithConcatConfig(
                    in_keys=["attn_out", "previous_actions", "physics_obs"],
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
