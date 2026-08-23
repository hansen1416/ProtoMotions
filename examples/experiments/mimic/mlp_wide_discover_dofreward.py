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
Mimic Environment — "Discover" Stage: Shape-Invariant DOF-Space Reward
=======================================================================

Same "discover" relaxed-termination pilot as `mlp_wide_discover.py`, with one variable changed:
the tracking reward/eval-failure criterion is measured in DOF-angle (local joint-angle) space
instead of world-space rigid-body position/rotation. See note/README.note.md §65 for the full
rationale, verification, and threshold-calibration writeup.

**Why:** every existing tracking reward (gt/gr/gv/gav) compares world-space `rigid_body_pos`/
`rigid_body_rot`, which is shape-dependent by construction — the same joint-angle trajectory lands
a taller body's hand at a different absolute position than a shorter body's. `dof_pos`/`dof_vel`
(local joint angles/velocities) are exactly shape-invariant: verified the 69-dim `dof_pos` layout
excludes the root (a MuJoCo `<freejoint>`, handled separately) and is structurally identical
(joint order/axes/hierarchy) across every shape in the `smpl_mor` family. Under this reward, the
canonical single-θ(t)-per-clip AMASS-retargeted corpus (`data_cache/150_128shape_canonical.pt`,
§64) is a complete, exactly shape-invariant target for all 128 shapes — no HUMOS reference needed.

Identical to `mlp_wide_discover.py` except:
  - `reward_components`: `mimic_tracking_rewards_factory` (gt/gr/gv/gav/rh) replaced with
    `mimic_dof_tracking_rewards_factory` (dp/dv/contact_match/heading/rh) — DOF-angle position +
    velocity tracking (exactly shape-invariant), contact-timing matching (already implemented
    elsewhere, dropped in `discover.py`, re-added here since it's also shape-invariant), and root
    heading tracking (near-invariant world-frame grounding). Root height (`rh`) is kept unchanged.
    Weights are a first-guess, not tuned — same honesty convention as every other lever in this
    lineage: `dp_weight`/`dv_weight` roughly combine `discover.py`'s split gt+gr / gv+gav weights.
  - `evaluator.evaluation_components`: adds `dp_error` (mean DOF-angle error, radians) *alongside*
    the existing `gt_error` (mean world-space position error, meters) rather than replacing it, so
    both failure curves are visible on the same run for direct comparison. `dp_error`'s
    threshold=0.35 (~20° mean per-joint error) is empirically calibrated (FK perturbation sweep
    against the real skeleton, matched to where `gt_error`'s 0.5m threshold is crossed under a
    coherent joint-angle bias) but not yet validated against real rollout data — expect to retune.
  - `termination_components`: unchanged (`fall_term_factory` only) — this experiment isolates the
    reward/eval measurement space, not the termination policy.
  - Everything else — architecture, optimizers, RSI, `configure_robot_and_simulator`,
    `motion_lib_config` — byte-identical to `mlp_wide_discover.py`.

Meant to run against the canonical (AMASS-only, no HUMOS) corpus:

    python protomotions/train_agent.py \\
        --robot-name smpl_mor --simulator isaacgym \\
        --experiment-path examples/experiments/mimic/mlp_wide_discover_dofreward.py \\
        --experiment-name hhi_wide_150motion_128shape_discover_dofreward \\
        --motion-file /workspace/motion_cache/150_128shape_canonical/150_128shape_canonical_offset.pt \\
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

WIDE_UNITS = 2896  # 1024 * sqrt(8) -- same as mlp_wide.py, kept for architecture parity


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
        mimic_dof_tracking_rewards_factory,
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

    # Sole termination is a genuine fall/collapse check -- no tracking-error termination at all,
    # so an ugly-but-upright trajectory is never cut short just for drifting from the reference.
    termination_components = {
        "fall": fall_term_factory(termination_height=0.15),
    }

    # DOF-space (shape-invariant) tracking reward, replacing world-space gt/gr/gv/gav. Weights
    # are a first-guess: dp/dv roughly combine discover.py's split position+rotation /
    # velocity+angular-velocity weights into one DOF-angle position/velocity term each.
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
        dp_error_factory,
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
