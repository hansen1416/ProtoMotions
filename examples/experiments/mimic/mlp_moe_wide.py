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
Mimic Environment — Capacity-Matched Control for mlp_moe.py (no MoE)
=========================================================================

Same as mlp.py in every respect except the actor trunk is widened from 1024 to 2896
units per layer — no MoE, no gate, no load-balancing loss, just a bigger monolithic
MLPWithConcat trunk, identical in structure to mlp.py.

Exists to isolate "MoE routing helped" from "the network just got bigger" when reading
mlp_moe.py's results: mlp_moe.py's K=8 experts (each 6x1024) have roughly 8x the trunk's
parameter count in that section of the network. Width here is picked to match that:

    w = 1024 * sqrt(K) = 1024 * sqrt(8) ~= 2896        (6*2896^2 ~= 8*6*1024^2)

If K changes for mlp_moe.py, recompute w the same way and update `WIDE_UNITS` below —
see note/README.note.md #18 [Decision] K sizing for why K=8 and where this formula
comes from.

Critic is left unchanged (same 4x1024 as mlp.py/mlp_moe.py) — capacity-matching only
applies to the piece that actually differs between the two runs being compared.

Trained from scratch, no --checkpoint. Use the exact same motion-file/robot-name as the
corresponding mlp_moe.py run for a valid comparison.

    # 1024-motion pilot control:
    python protomotions/train_agent.py \\
        --robot-name smpl_mor --simulator isaacgym \\
        --experiment-path examples/experiments/mimic/mlp_moe_wide.py \\
        --experiment-name hhi_moe_wide_1024_motion \\
        --motion-file <path-to-1024-clip-motion-file> \\
        --num-envs 4096 --batch-size 16384

    # 20,946-motion x {2,4}-shape control (once the filtered motion file is ready):
    python protomotions/train_agent.py \\
        --robot-name smpl_mor --simulator isaacgym \\
        --experiment-path examples/experiments/mimic/mlp_moe_wide.py \\
        --experiment-name hhi_moe_wide_20946_Nshape \\
        --motion-file <path-to-20946-clip-Nshape-motion-file> \\
        --num-envs 4096 --batch-size 16384
"""
from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig
from protomotions.components.terrains.config import TerrainConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.agents.ppo.config import PPOAgentConfig
from protomotions.components.scene_lib import SceneLibConfig
from protomotions.components.motion_lib import MotionLibConfig
import argparse

WIDE_UNITS = 2896  # 1024 * sqrt(8) -- keep in sync with mlp_moe.py's num_experts


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
        previous_actions_factory,
        mimic_target_poses_max_coords_factory,
        action_smoothness_factory,
        mimic_tracking_rewards_factory,
        pow_rew_factory,
        contact_match_rew_factory,
        tracking_error_term_factory,
        morphology_obs_factory,
    )
    from protomotions.envs.action import make_pd_action_config

    control_components = {
        "mimic": MimicControlConfig(
            bootstrap_on_episode_end=True,
        )
    }

    observation_components = {
        "max_coords_obs": max_coords_obs_factory(),
        "previous_actions": previous_actions_factory(history_steps=1),
        "mimic_target_poses": mimic_target_poses_max_coords_factory(with_velocities=True),
        "morphology_obs": morphology_obs_factory(),
    }

    termination_components = {
        "tracking_error": tracking_error_term_factory(threshold=0.5),
    }

    reward_components = {
        "action_smoothness": action_smoothness_factory(weight=-0.02),
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
        "pow_rew": pow_rew_factory(weight=-1e-5, min_value=-0.5),
        "contact_match_rew": contact_match_rew_factory(
            weight=-0.1, zero_during_grace_period=True
        ),
    }

    return EnvConfig(
        ref_contact_smooth_window=7,
        max_episode_length=1000,
        num_state_history_steps=2,
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
    from protomotions.agents.common.config import MLPWithConcatConfig, MLPLayerConfig
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

    actor_config = PPOActorConfig(
        num_out=robot_config.kinematic_info.num_dofs,
        actor_logstd=-2.9,
        in_keys=["max_coords_obs", "mimic_target_poses", "previous_actions", "morphology_obs"],
        mu_key="actor_trunk_out",
        mu_model=MLPWithConcatConfig(
            in_keys=[
                "max_coords_obs",
                "mimic_target_poses",
                "previous_actions",
                "morphology_obs",
            ],
            normalize_obs=True,
            norm_clamp_value=5,
            out_keys=["actor_trunk_out"],
            num_out=robot_config.number_of_actions,
            layers=[MLPLayerConfig(units=WIDE_UNITS, activation="relu") for _ in range(6)],
        ),
    )

    critic_config = MLPWithConcatConfig(
        in_keys=["max_coords_obs", "mimic_target_poses", "previous_actions", "morphology_obs"],
        out_keys=["value"],
        normalize_obs=True,
        norm_clamp_value=5,
        num_out=1,
        layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(4)],
    )

    agent_config: PPOAgentConfig = PPOAgentConfig(
        model=PPOModelConfig(
            in_keys=[
                "max_coords_obs",
                "mimic_target_poses",
                "previous_actions",
                "morphology_obs",
            ],
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
