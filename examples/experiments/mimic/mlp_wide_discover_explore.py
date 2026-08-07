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
Mimic Environment — "Discover" Stage: PPO Exploration Lever (`learnable_std`)

note/README.note.md §52 confirmed `discover`'s 75-78% plateau is not actuator saturation but an
RL optimization/precision problem. The first attempted fix, reward-gradient sharpening
(`mlp_wide_discover_sharpen.py`, `gt_coef` -25.0 -> -9.0), made things worse -- a plain
coefficient change reshapes the entire reward curve, not just the local gradient near the
near-miss zone, and reduced overall precision pressure (`eval/success_rate` declined to ~10pp
below `discover` by step 8399 instead of improving). Do not touch reward coefficients again for
this problem.

This is the second attempted fix, targeting PPO exploration instead. `discover`'s action
distribution has always had a FIXED std (`actor_logstd=-2.9`, `std = exp(-2.9) ~= 0.055`,
`PPOActorConfig.learnable_std` defaults `False`) -- and `entropy_coef` (default 0.005) is a
complete no-op whenever `learnable_std=False` (`PPO.actor_step`,
`protomotions/agents/ppo/agent.py:457-462`, entropy loss hardcoded to 0.0 in that branch). So
there has never actually been any entropy-driven exploration shaping this policy -- "entropy
collapse" couldn't have happened, because there was no entropy in play to begin with.

Pulling `discover`'s own `actor/clip_frac`/`actor/update_skipped` wandb history (no rerun needed)
shows no instability signal anywhere in the run: mean clip_frac 0.244, range 0.068-0.44, never
once triggers `actor_clip_frac_threshold=0.6` across 500 logged points. Per this project's own
established diagnostic heuristic (used identically for the MoE/wide-20946 lineage, note.md
§29-30): a flat curve with no clip_frac spikes points at an exploration problem, not a step-size
problem -- i.e. at `learnable_std`, not at `adaptive_lr`/tighter `actor_clip_frac_threshold`/
`e_clip` annealing (there's no instability here to guard against).

**Risk precedent, read before reusing this result**: `learnable_std=True` was already tried once
elsewhere in this project (`mlp_wide_explore.py`) and failed as a straight regression --
`actor/grad_norm_before_clip` spiked ~50x within ~200 epochs of enabling it, destroying action
smoothness (jerk roughly doubled) even though `eval/success_rate` looked superficially fine. That
attempt was a WARM START -- flipping `learnable_std=True` mid-training on an already-converged
checkpoint, suddenly unfreezing a parameter the rest of the network had never had to account for
moving, causing an acute non-stationarity shock. This experiment trains FROM SCRATCH instead
(matching every other lever in this 150-clip lineage) -- the std is free to move from epoch 0,
on equal footing with the rest of the network's optimization, never shocking an already-settled
policy. Not a guaranteed fix for the same failure mode, but it removes the specific mechanism
implicated in the prior regression. If launched, check `actor/std_mean` and
`actor/grad_norm_before_clip` in the first ~500-1000 epochs for the same spike signature before
trusting `eval/success_rate` alone -- it can mask this failure mode.

Identical to `mlp_wide_discover.py` except:
  - `PPOActorConfig(..., learnable_std=True, ...)`: the sole change. `actor_logstd=-2.9` stays as
    the *initial* value, now trainable instead of fixed. `entropy_coef` is left at its
    `PPOAgentConfig` default (0.005) -- matching this project's own established convention
    (`mlp_moe_stable.py`/`mlp_wide_explore.py`) of isolating `learnable_std` alone first, to avoid
    confounding which change is responsible for any result.
  - Everything else -- termination/reward components, architecture, optimizer LR values,
    `gradient_clip_val`, `advantage_normalization`, evaluator, `mass_scaled_gains` -- byte-
    identical to `mlp_wide_discover.py`. Isolate one variable, same as every prior lever.

Meant to be trained on the same 150-clip/128-shape ablation set as `discover` and every other
lever, for a direct same-data comparison:

    python protomotions/train_agent.py \\
        --robot-name smpl_mor --simulator isaacgym \\
        --experiment-path examples/experiments/mimic/mlp_wide_discover_explore.py \\
        --experiment-name hhi_wide_150motion_128shape_discover_explore \\
        --motion-file /workspace/motion_cache/small150_128shape.pt \\
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
        mimic_tracking_rewards_factory,
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

    # Only the tracking-shaping reward remains -- action_smoothness/pow_rew/contact_match_rew
    # (effort, smoothness, contact-timing penalties) are omitted entirely for this stage.
    # Byte-identical to mlp_wide_discover.py -- the sharpen experiment already showed touching
    # these coefficients backfires; this experiment isolates the PPO exploration lever instead.
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
        # Sole change from mlp_wide_discover.py: the action std is now a trainable parameter
        # instead of fixed at exp(-2.9)~=0.055 for the whole run. entropy_coef is left at its
        # PPOAgentConfig default (0.005) to isolate this one variable (see module docstring).
        learnable_std=True,
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
