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
Mimic Environment — "Discover" Stage: Temporal Context Lever (dilated history window)

`discover`'s policy is effectively memoryless: of its four observation inputs (`max_coords_obs`,
`mimic_target_poses`, `previous_actions`, `morphology_obs`), all are single-frame except
`previous_actions`, which only exposes the immediately-prior action.
`num_state_history_steps=2` in `mlp_wide_discover.py` is a red herring -- it only allocates
`StateHistoryBuffer` capacity; nothing in that config's `observation_components` actually reads
more than the one previous action out of it. If the "close but not exact" failures (note.md §51-52)
are timing/momentum-related -- plausible for contact-rich motion -- the policy has no way to see
far enough back to diagnose or correct a mistimed motion.

Two heavier options were considered and set aside for a first attempt:
  - True recurrence (LSTM/GRU with hidden state carried across PPO rollout steps): PPO's rollout
    buffer (`ExperienceBuffer`/`DictDataset`) flattens `[num_steps, num_envs, ...]` and globally
    shuffles across BOTH env and time before every mini-epoch -- there is no sequence-preserving
    minibatch path, no hidden-state storage, and no reset-on-done logic anywhere in this codebase.
    Adding true recurrence would need substantial new infrastructure with no existing precedent.
  - `Transformer`/`TransformerConfig` (`protomotions/agents/common/transformer.py`) already exists
    and does genuine temporal self-attention over a token window, but has only ever been used once
    (MaskedMimic's VAE prior) -- never in a PPO actor/critic -- and needs hand-built token encoders
    (its docstring's "positional encoding" isn't actually implemented).

This experiment instead uses dilated frame-stacking -- windowed history concatenated into one
wider flat observation -- which is already production-tested for actor+critic in this exact
codebase: `examples/experiments/amp/mlp.py` and `examples/experiments/mimic/mlp_wide_amp.py`
already feed `historical_max_coords_obs_factory(history_steps=[1,2,3,4,8,16,32])` directly into
actor/critic `in_keys` via the same plain `MLPWithConcatConfig` trunk used here. No PPO/buffer
changes are needed -- it's just a wider flat observation computed fresh every step from the env's
own rolling `StateHistoryBuffer`, fully compatible with the existing flat/shuffled minibatch
pipeline. Cheapest, lowest-risk way to test whether temporal context helps at all before
committing to Transformer-based encoding or true recurrence.

Identical to `mlp_wide_discover.py` except:
  - `HISTORY_STEPS = [1, 2, 3, 4, 8, 16, 32]` (same dilation schedule as `amp/mlp.py`/
    `mlp_wide_amp.py` -- reused, not invented blind) and a new `historical_max_coords_obs`
    observation component wired into both actor and critic `in_keys`.
  - `num_state_history_steps=32` (was 2) -- buffer needs to actually hold enough frames for the
    dilation schedule now that something reads beyond index 1.
  - Everything else -- termination, reward coefficients (discover's original -25.0/-5.0/etc, not
    the failed `sharpen` values), architecture width, optimizer LR, `gradient_clip_val`,
    `advantage_normalization`, evaluator, `mass_scaled_gains`, `learnable_std` (stays `False` --
    isolate this variable from the still-pending `discover_explore` lever) -- byte-identical to
    `mlp_wide_discover.py`.

Meant to be trained on the same 150-clip/128-shape ablation set as `discover` and every other
lever, for a direct same-data comparison:

    python protomotions/train_agent.py \\
        --robot-name smpl_mor --simulator isaacgym \\
        --experiment-path examples/experiments/mimic/mlp_wide_discover_historical.py \\
        --experiment-name hhi_wide_150motion_128shape_discover_historical \\
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

# Dilated history steps for temporal context (actor + critic). Same schedule already validated
# in examples/experiments/amp/mlp.py and examples/experiments/mimic/mlp_wide_amp.py.
HISTORY_STEPS = [1, 2, 3, 4, 8, 16, 32]


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
        # New: dilated window of past body-state frames, giving the policy real temporal
        # context beyond the single previous action `previous_actions` already provides.
        "historical_max_coords_obs": historical_max_coords_obs_factory(
            local_obs=True,
            root_height_obs=True,
            observe_contacts=False,
            history_steps=HISTORY_STEPS,
        ),
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
    # Byte-identical to mlp_wide_discover.py's original coefficients (not the failed sharpen
    # values) -- this experiment isolates the temporal-context lever only.
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

    actor_in_keys = [
        "max_coords_obs",
        "historical_max_coords_obs",
        "mimic_target_poses",
        "previous_actions",
        "morphology_obs",
    ]

    actor_config = PPOActorConfig(
        num_out=robot_config.kinematic_info.num_dofs,
        actor_logstd=-2.9,
        in_keys=actor_in_keys,
        mu_key="actor_trunk_out",
        mu_model=MLPWithConcatConfig(
            in_keys=actor_in_keys,
            normalize_obs=True,
            norm_clamp_value=5,
            out_keys=["actor_trunk_out"],
            num_out=robot_config.number_of_actions,
            layers=[MLPLayerConfig(units=WIDE_UNITS, activation="relu") for _ in range(6)],
        ),
    )

    critic_config = MLPWithConcatConfig(
        in_keys=actor_in_keys,
        out_keys=["value"],
        normalize_obs=True,
        norm_clamp_value=5,
        num_out=1,
        layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(4)],
    )

    agent_config: PPOAgentConfig = PPOAgentConfig(
        model=PPOModelConfig(
            in_keys=actor_in_keys,
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
