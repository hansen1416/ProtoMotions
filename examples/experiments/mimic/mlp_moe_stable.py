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
Mimic Environment — GMT-style MoE Actor with Update-Stability Guard Rails
==========================================================================

Identical to mlp_moe.py (same K=8 MoE actor, same observations/rewards/critic) except
for two PPO update-safety settings that were both already implemented in
protomotions/agents/ppo/{config,agent}.py but left disabled/loose in mlp_moe.py:

1. `adaptive_lr` (KL-based LR scaling, agent.py's `_update_adaptive_lr`) — enabled here.
   Halves actor/critic LR when the post-update KL exceeds 2x `desired_kl`, grows it back
   (x1.5, capped at `max_lr`) when KL is well under target. Was fully OFF in mlp_moe.py.

2. `actor_clip_frac_threshold` tightened from 0.6 -> 0.4 — skip the remaining minibatch
   actor updates for the epoch earlier, before a large fraction of the batch is already
   clipped, rather than only as a last resort.

Both are guard rails that only change behavior once an update is already destabilizing —
they don't alter nominal training dynamics on well-behaved epochs, so this is meant as a
direct, single-purpose comparison against mlp_moe.py's hhi_moe_20946_neutral run, not a
new architecture or exploration change. (Entropy/learnable_std was considered as a third
change at the same time but deliberately held back for a separate follow-up run, since it
would confound "did the guard rails help" with "did more exploration help" — see
note/README.note.md #18.)

Meant to be launched as a WARM START from hhi_moe_20946_neutral's checkpoint, not from
scratch: pass --checkpoint pointing at that run's last.ckpt together with a NEW
--experiment-name (train_agent.py's detect_checkpoint_mode() treats this as "warm_start",
not "resume" -- it executes this experiment file fresh, so the settings above actually
take effect, and only loads the checkpoint's weights as initialization; a same-name
"resume" would instead reload the pickled config and ignore all of this). Architecture is
identical to mlp_moe.py (same MoEMLPConfig, same critic, same action config), so the
checkpoint loads cleanly with no shape mismatches. Chosen over training from scratch
because it directly tests the question we actually care about -- does this prevent/dampen
a repeat of the epoch-7540-style dip from the policy's current ~90%+ state -- for a
fraction of the compute, at the cost of not also re-testing the early-training clip_frac
instability (epochs 247-931 in hhi_moe_20946_neutral's log). See note/README.note.md #18.

    # Warm start from hhi_moe_20946_neutral's checkpoint (6x A40, matched envs/batch):
    python protomotions/train_agent.py \\
        --robot-name smpl_mor_neutral --simulator isaacgym \\
        --experiment-path examples/experiments/mimic/mlp_moe_stable.py \\
        --experiment-name hhi_moe_20946_neutral_stable \\
        --checkpoint results/hhi_moe_20946_neutral/last.ckpt \\
        --motion-file /workspace/20946_neutral_offset/humanml3d_neutral_20946_slurmrank.pt \\
        --num-envs 6144 --batch-size 24576 --ngpu 6 \\
        --use-wandb --wandb-project hhi-protomotions --wandb-entity yugoamaryl \\
        --wandb-group hhi_moe_20946_neutral_stable
"""
from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig
from protomotions.components.terrains.config import TerrainConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.agents.ppo.config import PPOAgentConfig
from protomotions.components.scene_lib import SceneLibConfig
from protomotions.components.motion_lib import MotionLibConfig
import argparse


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
    from protomotions.agents.common.moe_mlp import MoEMLPConfig
    from protomotions.agents.ppo.config import (
        PPOActorConfig,
        PPOModelConfig,
        AdvantageNormalizationConfig,
        MoELoadBalanceConfig,
        AdaptiveLRConfig,
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

    actor_in_keys = ["max_coords_obs", "mimic_target_poses", "previous_actions", "morphology_obs"]

    actor_config = PPOActorConfig(
        num_out=robot_config.kinematic_info.num_dofs,
        actor_logstd=-2.9,
        in_keys=actor_in_keys,
        mu_key="actor_trunk_out",
        mu_model=MoEMLPConfig(
            in_keys=actor_in_keys,
            normalize_obs=True,
            norm_clamp_value=5,
            out_keys=["actor_trunk_out"],
            num_out=robot_config.number_of_actions,
            num_experts=8,
            expert_layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(6)],
            gate_hidden_units=[256, 256],
            gate_mode="learned",
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
        # Tightened from mlp_moe.py's 0.6: skip the remaining actor minibatch updates
        # for the epoch earlier, before a large fraction of the batch is already clipped.
        actor_clip_frac_threshold=0.4,
        # KL-based LR scaling (was disabled in mlp_moe.py): halves actor/critic LR when
        # post-update KL > 2x desired_kl, grows it back (x1.5) when KL is well under target.
        adaptive_lr=AdaptiveLRConfig(enabled=True, desired_kl=0.01),
        moe_load_balance=MoELoadBalanceConfig(enabled=True, lambda_lb=0.01),
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
