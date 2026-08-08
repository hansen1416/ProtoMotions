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
Mimic Environment — "Discover" Stage: Forward-Lookahead Lever (multi-step future target window)

Joining `discover`'s 55-epoch failure log against motion text descriptions and kinematic features
(note.md, this session) found its hardest clips are dominated by turning/heading-change locomotion
and dynamic transitions (kicks, jumps, quick accel/decel) -- a different profile than the old
crawl/kneel/squat/single-leg-balance cluster. Root-angular-velocity and DOF-velocity features
correlate with per-clip failure rate at r~0.2-0.3 -- weak-moderate, but far stronger than shape
extremity's r~0.003-0.09 across every prior check -- real signal that "busier"/more-rotational
motions are harder, independent of body shape.

`discover` never overrides `MimicControlConfig.future_steps`, which defaults to `1` -- so the
policy's target-pose observation only ever contains the single immediately-next reference frame
(33ms ahead), essentially no anticipation. For a clip that starts turning or kicking, the policy
gets almost no warning before it must react, and PD-control lag then compounds the problem -- a
plausible direct mechanism for the newly-identified failure category. This is the mirror image of
`mlp_wide_discover_historical.py` (a backward-looking dilated window over past body state): this
lever instead widens what the policy can see of the upcoming *target*.

Not a new mechanism -- `examples/experiments/mimic/mlp_bm_l2c2.py` already uses
`MimicControlConfig(bootstrap_on_episode_end=True, future_steps=[1, 2, 4, 8])` in production.
`mimic_control.py`'s `populate_context()` already handles a list-valued `future_steps`, and already
clamps future queries past clip end via `torch.minimum(future_times, motion_lengths)` (holds at the
final frame), so looking ahead near a clip's end is already-handled behavior, not new risk.
`mimic_target_poses_max_coords_factory(with_velocities=True)` (`discover`'s existing observation
call, unchanged here) already selects "all available future steps" when its own `future_steps` arg
is left `None` -- so widening `MimicControlConfig.future_steps` automatically widens the
target-pose observation tensor with zero other code changes needed. Reusing `mlp_bm_l2c2.py`'s
exact validated `[1, 2, 4, 8]` schedule rather than inventing a new one -- lowest-risk choice for a
first attempt.

Identical to `mlp_wide_discover.py` except:
  - `FUTURE_STEPS = [1, 2, 4, 8]` (same dilation schedule already validated in `mlp_bm_l2c2.py`)
    passed to the `mimic` control component's `future_steps`.
  - `mimic_target_poses_max_coords_factory(with_velocities=True)` call itself is unchanged --
    it already resolves to "all steps in the buffer," which now means the 4 dilated future frames
    instead of 1. `max_coords_obs`, `previous_actions`, `morphology_obs`,
    `num_state_history_steps=2` (this lever doesn't touch backward history) all stay exactly as-is.
  - No `in_keys` changes needed anywhere (actor, `mu_model`, critic, `PPOModelConfig`) -- the
    `"mimic_target_poses"` key already exists in every `in_keys` list; only its underlying tensor
    widens. `nn.LazyLinear` infers the new input dimension automatically.
  - Everything else -- termination, reward coefficients (discover's original -25.0/etc.),
    architecture width, optimizer LRs, `gradient_clip_val`, `advantage_normalization`, evaluator,
    `mass_scaled_gains`, `learnable_std` (stays `False`, default) -- byte-identical to
    `mlp_wide_discover.py`. Isolates one variable, not combined with `_historical.py` or the
    pending `_explore.py` lever.

Meant to be trained on the same 150-clip/128-shape ablation set as `discover` and every other
lever, for a direct same-data comparison:

    python protomotions/train_agent.py \\
        --robot-name smpl_mor --simulator isaacgym \\
        --experiment-path examples/experiments/mimic/mlp_wide_discover_lookahead.py \\
        --experiment-name hhi_wide_150motion_128shape_discover_lookahead \\
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

# Dilated forward-lookahead schedule for the mimic target-pose observation. Same schedule already
# validated in examples/experiments/mimic/mlp_bm_l2c2.py.
FUTURE_STEPS = [1, 2, 4, 8]


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
            future_steps=FUTURE_STEPS,
        )
    }

    observation_components = {
        "max_coords_obs": max_coords_obs_factory(),
        "previous_actions": previous_actions_factory(history_steps=1),
        # Same factory call as discover -- resolves to "all available future steps," which now
        # means the FUTURE_STEPS-dilated window from the mimic control component above instead of
        # a single next-step frame.
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
    # Byte-identical to mlp_wide_discover.py's original coefficients -- this experiment isolates
    # the forward-lookahead lever only.
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
