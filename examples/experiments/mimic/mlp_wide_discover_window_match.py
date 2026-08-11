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
Mimic Environment — "Discover" Stage: Bounded Local-Window Reward Matching (Phase-Slip
Forgiveness)
========================================================================================

Five straight levers on `discover` (mlp_wide_discover_sharpen.py: reward coefficient reshape;
mlp_wide_discover_explore.py: PPO exploration; mlp_wide_discover_historical.py: backward
observation context; mlp_wide_discover_lookahead.py: forward observation context;
mlp_wide_discover_historical_lookahead.py: both combined) all landed on the same eval/success_rate
plateau (~75-80%, see note/README.note.md §54). Actuator saturation was already ruled out via an
IsaacGym torque-replay diagnostic (max torque/effort-limit ratio 0.46 across 40 episodes -- plenty
of headroom). Failure-mode analysis (video review + kinematic correlation) points at turning/
heading-change and dynamic-transition clips where the policy's motion looks qualitatively right but
"close but not exact" -- consistent with the policy being a few frames out of *phase* with the
reference (control lag on fast transitions), not performing the wrong motion. Root-angular-velocity/
DOF-velocity features correlate with per-clip failure rate at r~=0.2-0.3 -- weak but 3-30x stronger
than every shape-extremity check (~0). Every context-widening lever tried so far gave the policy
more *information* without changing what the reward actually measures at each instant (still
frame-exact wall-clock comparison against the reference). This is the first lever that changes the
matching mechanism itself.

Mechanism: instead of comparing the policy's current pose to the reference at a single deterministic
time index, search a small window of nearby reference times
[motion_times - window_back_steps*dt, motion_times + window_fwd_steps*dt] and reward against
whichever frame in that window the policy's current pose actually matches best (lowest whole-pose
mean position error) -- forgiving small phase misalignment instead of penalizing it as if the pose
were simply wrong. Implemented in MimicControl.populate_context() (protomotions/envs/control/
mimic_control.py): the window is stateless, recomputed fresh every step around the ever-advancing
nominal motion_times (not a persistent warping cursor that could get "stuck"), which intrinsically
bounds drift to at most the window size at any single step.

Critically, this ONLY changes the REWARD signal. termination_components (this file has none tracking
error at all -- see mlp_wide_discover.py) and the evaluator's evaluation_components (gt_error/
gr_error/max_joint_error below) still read the exact, frame-matched EnvContext.mimic.ref_state,
completely unaffected -- see EnvContext.mimic.reward_ref_state in context_views.py, and the 5
repointed reward factories (gt/gr/gv/gav/rh_rew_factory) in component_factories.py. This keeps
eval/success_rate comparable to the entire 75-80% plateau history: if eval/termination also became
lenient, an improved number would just mean the yardstick got easier, not that the policy improved.

window_back_steps=4 (133ms), window_fwd_steps=1 (33ms) at this robot/simulator's real 30Hz control
rate (smpl_mor + IsaacGym: fps=60, decimation=2). Asymmetric and back-biased because the hypothesized
failure mode is control lag (policy trailing the reference during fast transitions) -- a lagging
policy's current pose matches the reference's *past*, not its future. Kept deliberately small (6
candidate frames total) to bound the reward-hacking surface, since nothing in the diagnostics
suggests the policy runs ahead of schedule. THIS IS A FIRST-GUESS STARTING POINT, NOT A TUNED
NUMBER -- two new diagnostics, env/mimic/window_offset_steps_mean and env/mimic/window_at_bound_mean
(logged automatically whenever the window is active), tell us post-hoc whether 4 was too small
(offsets pinned at the backward edge) or comfortably oversized (offsets cluster near 0).

Identical to `mlp_wide_discover.py` except:
  - `control_components`: `MimicControlConfig` adds `window_back_steps=4, window_fwd_steps=1`.
    Everything else -- observation_components, termination_components, reward_components (still the
    same mimic_tracking_rewards_factory weights/coefs -- only what they READ changed, not the
    weights/coefficients themselves), agent_config() including the evaluator, architecture,
    optimizers -- is byte-identical. Isolates one variable (reward matching mechanism), not combined
    with `_historical.py` or `_lookahead.py`'s observation-context levers, same as every prior lever
    in this lineage.

Meant to be trained on the same 150-clip/128-shape ablation set every other lever already uses, for
a direct same-data comparison:

    python protomotions/train_agent.py \\
        --robot-name smpl_mor --simulator isaacgym \\
        --experiment-path examples/experiments/mimic/mlp_wide_discover_window_match.py \\
        --experiment-name hhi_wide_150motion_128shape_discover_window_match \\
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
            window_back_steps=4,
            window_fwd_steps=1,
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
    # (effort, smoothness, contact-timing penalties) are omitted entirely for this stage. Weights/
    # coefficients unchanged from mlp_wide_discover.py -- only the reference these factories read
    # (reward_ref_state vs ref_state) differs, via MimicControlConfig.window_back_steps/fwd_steps.
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
