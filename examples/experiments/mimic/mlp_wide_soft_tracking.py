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
Mimic Environment — Wide Trunk, Soft Tracking (loosened tracking, no mass-scaled gains)
========================================================================================

Same as mlp_wide.py in every respect except termination_components, reward_components, and
mass_scaled_gains (explicitly disabled here). Isolates "does relaxing exact-pose-matching help
multi-shape training" from every other variable already tried (network width, mass-scaled PD
gains, whole-actor or per-segment) -- see note/README.note.md for the full rationale and the run
comparison that motivated this.

mass_scaled_gains is turned OFF for this experiment (smpl_mor.py defaults it on) -- the in-flight
seg-gain-only run (hhi_wide_150motion_128shape_seggain, §44) hadn't shown a promising enough trend
on its own to justify stacking it with soft-tracking in the same run and confounding the two. Not
a revert of §42/§44 -- those stay on by default for every other smpl_mor experiment file.

Motivation: for `smpl_mor` (128 body shapes), the reference motion was retargeted onto each
shape from a single captured/generated trajectory. That retargeted trajectory is not
guaranteed to be exactly achievable for every shape's actual proportions -- a joint
configuration that's fine for one body's limb lengths may be kinematically/dynamically
infeasible for another. mlp_wide.py's termination and reward are both built around exact,
per-body tracking:

- termination_components only had "tracking_error" (compute_tracking_error: MAX per-body
  position error > 0.5m ends the episode). There is no separate fall-detection termination in
  mlp_wide.py -- physical failure and pose-tracking deviation are conflated into one signal.
  A single body's individually-infeasible target (for a given shape) ends the whole episode
  exactly as if the character had collapsed.
- reward_components dominant weights are gt_rew (gt_weight=0.5, exact per-body position) and
  gr_rew (gr_weight=0.3, exact per-body rotation) -- both judge every body against its literal
  reference target, with no coarser "is it generally doing the right thing" signal beyond root
  height (rh_weight=0.2).

Changes here, all additive/reweighting -- no new observation/action/architecture changes:

1. termination_components: added "fall" (fall_term_factory) -- genuine contact+height fall
   detection (already implemented in terminations/base.py as fall_termination, just never wired
   into any experiment config before this). "tracking_error" switched from
   tracking_error_term_factory (MAX per-body, 0.5m) to mean_tracking_error_term_factory (MEAN
   across all bodies, 0.5m) -- tolerates one or two individually-infeasible bodies as long as
   overall tracking stays reasonable, while a real fall still reliably ends the episode via the
   new "fall" component. threshold=0.5 for the mean case is chosen to match agent_config's
   existing gt_error_factory(threshold=0.5) eval-success threshold exactly (mlp_wide.py's
   evaluator success criterion is already mean-based) -- training now terminates exactly when an
   episode would be counted as failing anyway, not earlier (via the old MAX rule) or later.

2. reward_components: gt_weight 0.5->0.3, gr_weight 0.3->0.2, gav_weight 0.2->0.1 (de-emphasize
   exact per-body position/rotation/angular-velocity matching), rh_weight 0.2->0.25 (slightly up
   -- already the coarsest existing signal). Added global_anchor_pos_rew/global_anchor_ori_rew
   (BeyondMimic-style root/anchor-only position+orientation tracking, already implemented, just
   unused in mlp_wide.py) as the new coarse "is it going the right way and facing the right
   direction" signal -- root trajectory matters more than any single limb's exact angle.

Note: this changes what "success" and episode_length mean relative to mlp_wide.py runs --
episodes here can run longer/be marked more successful for behavior that would have ended an
mlp_wide.py episode early. Not directly comparable to prior mlp_wide.py runs on those two
metrics without accounting for this; eval_holdout/success_rate-style comparisons should note the
termination rule changed, not just the policy.

    # Small multi-shape ablation on the massgain fix + soft tracking, combined:
    python protomotions/train_agent.py \\
        --robot-name smpl_mor --simulator isaacgym \\
        --experiment-path examples/experiments/mimic/mlp_wide_soft_tracking.py \\
        --experiment-name <name> \\
        --motion-file <path> \\
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
        global_anchor_pos_rew_factory,
        global_anchor_ori_rew_factory,
        pow_rew_factory,
        contact_match_rew_factory,
        mean_tracking_error_term_factory,
        fall_term_factory,
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
        "fall": fall_term_factory(termination_height=0.15),
        "tracking_error": mean_tracking_error_term_factory(threshold=0.5),
    }

    reward_components = {
        "action_smoothness": action_smoothness_factory(weight=-0.02),
        **mimic_tracking_rewards_factory(
            gt_weight=0.3,
            gr_weight=0.2,
            gv_weight=0.1,
            gav_weight=0.1,
            rh_weight=0.25,
            gt_coef=-25.0,
            gr_coef=-5.0,
            gv_coef=-0.5,
            gav_coef=-0.1,
            rh_coef=-100.0,
        ),
        "global_anchor_pos_rew": global_anchor_pos_rew_factory(weight=0.3, sigma=0.3),
        "global_anchor_ori_rew": global_anchor_ori_rew_factory(weight=0.2, sigma=0.4),
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
    """Configure robot to add contact sensors for foot contact tracking.

    Also disables mass_scaled_gains (§42/§44's whole-actor and per-segment PD gain scaling) for
    this experiment specifically. smpl_mor.py has it on by default, so any smpl_mor run would
    otherwise inherit it -- turned off here to isolate soft-tracking's effect on its own, since
    the in-flight seg-gain-only run (hhi_wide_150motion_128shape_seggain) hasn't shown a
    promising enough trend on its own to justify stacking it with soft-tracking in the same run.
    Not a revert of §42/§44 (those stay on for smpl_mor by default) -- see note/README.note.md.
    """
    robot_cfg.update_fields(
        contact_bodies=["all_left_foot_bodies", "all_right_foot_bodies"]
    )
    robot_cfg.control.mass_scaled_gains = False


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
