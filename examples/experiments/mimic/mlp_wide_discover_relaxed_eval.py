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
Mimic Environment — "Discover" Stage: Relaxed Evaluator Success Threshold

Five straight levers on the `discover` baseline -- `discover_sharpen` (reward-coefficient
reshape, `gt_coef` -25.0 -> -9.0), `discover_explore` (PPO exploration), `discover_historical`
(backward observation context), `discover_lookahead` (forward observation context),
`discover_historical_lookahead` (both combined) -- have all landed on the same `eval/success_rate`
plateau (~75-80%). `discover_window_match` (bounded local-window reference matching) is currently
training on the same question from a different angle. Every one of those levers changed how the
policy is *trained*; this is the first lever to instead change how success is *measured*.

Why this is different from every prior lever: `agent_config()`'s evaluator has used
`gt_error_factory(threshold=0.5)` -- fail an episode if mean body-position error exceeds 0.5m --
unchanged in literally every experiment file in this entire lineage (`mlp.py` through every
`discover_*` sibling, MoE, fusion_stage2, lora_stage2). It has never once been varied. This
experiment asks the opposite question from the others: instead of "can we make the policy track
more precisely," it asks "is 0.5m an appropriate bar for what should count as success at all."

Mechanism and consequence (checked directly against `base_evaluator.py`/`mimic_evaluator.py`
before writing this): `_motion_failed` -- the tensor `eval/success_rate` is computed from -- is a
per-episode logical OR over every evaluation component that has a `threshold` set
(`combine_evaluation()` in `base_env/utils.py`). `gt_error` is the *only* thresholded component in
`discover`'s evaluator (`gr_error`/`max_joint_error` are logged metrics only, no threshold). That
same `_motion_failed` also drives `motion_weights_rules` (`mimic_evaluator.py::
_update_motion_sampling_weights`): failing clips get their curriculum sampling weight reset high,
passing clips decay. So loosening this threshold is not purely cosmetic re-labeling -- it also
changes which clips get aggressively re-sampled during training, since clips that now clear the
bar stop being treated as "still hard."

This means `eval/success_rate` from this run is **not directly comparable** to the 75-80% plateau
number from every prior lever -- the yardstick itself moved. That's the deliberate point of the
experiment, not an oversight: it's a genuinely untried axis, run in parallel with
`discover_relaxed_rh.py` (which loosens reward-side tracking pressure instead, keeping the
evaluator untouched and therefore directly comparable to the plateau history). Together the two
runs separate "does the policy actually track better under looser reward pressure" from "is the
bar we're grading it against arbitrarily strict" -- two different questions this pair keeps
cleanly isolated from each other.

Identical to `mlp_wide_discover.py` except:
  - `agent_config()`'s evaluator: `gt_error_factory(threshold=0.5)` -> `threshold=0.75`. This is
    the *only* change. `gr_error_factory()`/`max_joint_error_factory()` stay unthresholded (logged
    only, as before).
  - Everything else -- `reward_components` (all `mimic_tracking_rewards_factory` weights/coefs
    unchanged, including `rh_coef=-100.0`), `termination_components` (`fall_term_factory` only),
    `control_components` (no window matching), architecture, optimizers, RSI, `mass_scaled_gains`
    -- byte-identical to `mlp_wide_discover.py`.

0.75 is a first guess (1.5x the original bar), not derived from measured failure-residual data --
flagged the same way `discover_window_match.py`'s `4/1` window size was. Not swept.

Meant to be trained on the same 150-clip/128-shape ablation set as every other lever in this
lineage, for a direct same-data comparison:

    python protomotions/train_agent.py \\
        --robot-name smpl_mor --simulator isaacgym \\
        --experiment-path examples/experiments/mimic/mlp_wide_discover_relaxed_eval.py \\
        --experiment-name hhi_wide_150motion_128shape_discover_relaxed_eval \\
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
    # Unchanged from mlp_wide_discover.py -- this experiment only touches the evaluator's
    # success threshold, not the reward the policy is actually trained against.
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
                # Only change vs. mlp_wide_discover.py: threshold 0.5 -> 0.75. This is the sole
                # thresholded component, so it alone determines both eval/success_rate and the
                # curriculum's failed/passed split (see module docstring).
                "gt_error": gt_error_factory(threshold=0.75),
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
