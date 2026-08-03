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
Mimic Environment — Wide Trunk, AMP-Style Discriminator (style reward, blended with tracking)
===============================================================================================

Same env/reward/termination as mlp_wide.py, plus an AMP discriminator (protomotions.agents.
amp.agent.AMP) that judges the agent's recent trajectory against real reference clips, and
blends a style reward in alongside the existing tracking reward.

Not built on MimicADD (protomotions/agents/mimic/agent_add.py) -- that class's discriminator
input (mimic_target_poses_diff, expert=0) is frame-locked to the current reference index, which
is really an adversarial reformulation of the tracking loss, not "looks natural regardless of
exact phase". This file wires the discriminator the way examples/experiments/amp/mlp.py does
instead: historical_max_coords_obs, sampled at random unaligned times from the whole motion
library, so the "real" class is a general style prior rather than a specific per-frame target.

The discriminator is also conditioned on morphology_obs. This is grounded, not a spurious label:
motion_lib.py stores motion_betas/motion_gender_ids per clip, and mimic_motion_manager.py's
sample_motions() only ever assigns an env a clip retargeted for that env's own body shape -- so
each expert sample's attached shape is the real shape that clip was generated for, not an
unrelated borrowed label. See note/README.note.md for the full reasoning (including a first-pass
version of this argument that was wrong, and what corrected it).

task_reward_w and discriminator_reward_w are independent, pre-existing knobs (protomotions/
agents/ppo/agent.py, protomotions/agents/amp/agent.py) that sum at the advantage level -- both
add/mlp.py and amp/mlp.py set task_reward_w=0.0 (pure style, no explicit task reward), but this
file keeps both nonzero (anchored near a 0.5/0.5 split), since the goal here is still to
reproduce a *specific* retargeted clip per env, not just "some natural human motion".

configure_robot_and_simulator() is unmodified from mlp_wide.py -- in particular it does NOT
disable mass_scaled_gains, so this run inherits smpl_mor's default per-segment PD gain scaling
(note.md #44), same as the in-flight hhi_wide_150motion_128shape_seggain run. That makes seggain
the direct same-everything-else comparator for this experiment.

Known inherent behavior change from using the AMP agent class (not a design choice made here --
see AMP.post_env_step_modifications): episodes can now also end when the discriminator judges
`discriminator_max_cumulative_bad_transitions` (default 10) consecutive transitions as below
`discriminator_reward_threshold`, on top of mlp_wide.py's existing tracking-error termination.

    python protomotions/train_agent.py \\
        --robot-name smpl_mor --simulator isaacgym \\
        --experiment-path examples/experiments/mimic/mlp_wide_amp.py \\
        --experiment-name hhi_wide_150motion_128shape_amp \\
        --motion-file /workspace/motion_cache/small150_128shape.pt \\
        --num-envs 4096 --batch-size 16384
"""
from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig
from protomotions.components.terrains.config import TerrainConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.agents.amp.config import AMPAgentConfig
from protomotions.components.scene_lib import SceneLibConfig
from protomotions.components.motion_lib import MotionLibConfig
import argparse

WIDE_UNITS = 2896  # keep in sync with mlp_wide.py

# Dilated history steps for the discriminator's pose-window input -- matches amp/mlp.py's dilation.
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
        previous_actions_factory,
        mimic_target_poses_max_coords_factory,
        historical_max_coords_obs_factory,
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
        # Discriminator-only input (not consumed by actor/critic) -- agent's own recent
        # trajectory window, judged against reference_obs_components' expert samples below.
        "historical_max_coords_obs": historical_max_coords_obs_factory(
            local_obs=True,
            root_height_obs=True,
            observe_contacts=False,
            history_steps=HISTORY_STEPS,
        ),
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
        # Bumped from mlp_wide.py's 2 -- needs to hold the discriminator's dilated history
        # window (max(HISTORY_STEPS) = 32). Real memory cost, same order as amp/mlp.py's own
        # already-working config.
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
) -> AMPAgentConfig:
    from protomotions.agents.common.config import (
        MLPWithConcatConfig,
        MLPLayerConfig,
        ModuleContainerConfig,
    )
    from protomotions.agents.ppo.config import PPOActorConfig
    from protomotions.agents.amp.config import (
        AMPModelConfig,
        DiscriminatorConfig,
        AMPParametersConfig,
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
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.obs import (
        compute_historical_max_coords_from_motion_lib,
        compute_morphology_from_motion_lib,
    )

    actor_in_keys = [
        "max_coords_obs",
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

    disc_in_keys = ["historical_max_coords_obs", "morphology_obs"]

    discriminator_config = DiscriminatorConfig(
        in_keys=disc_in_keys,
        out_keys=["disc_logits"],
        models=[
            MLPWithConcatConfig(
                in_keys=disc_in_keys,
                out_keys=["disc_logits"],
                normalize_obs=True,
                norm_clamp_value=5,
                num_out=1,
                layers=[
                    MLPLayerConfig(units=1024, activation="relu"),
                    MLPLayerConfig(units=512, activation="relu"),
                ],
            )
        ],
    )

    disc_critic_in_keys = ["max_coords_obs", "historical_max_coords_obs"]

    disc_critic_config = ModuleContainerConfig(
        in_keys=disc_critic_in_keys,
        out_keys=["disc_value"],
        models=[
            MLPWithConcatConfig(
                in_keys=disc_critic_in_keys,
                out_keys=["disc_value"],
                normalize_obs=True,
                norm_clamp_value=5,
                num_out=1,
                layers=[
                    MLPLayerConfig(units=512, activation="relu"),
                    MLPLayerConfig(units=256, activation="relu"),
                ],
            )
        ],
    )

    # Reference observation components for discriminator expert data. Agent injects
    # motion_lib/motion_ids/motion_times/dt at runtime (not available in EnvContext).
    # NOTE: compute_historical_max_coords_from_motion_lib requires num_state_history_steps
    # as a plain argument with no default -- must be passed explicitly here even though
    # history_steps is a list (the branch that would use it is skipped, but Python still
    # requires the argument). amp/mlp.py's own reference_obs_components omits this and
    # would raise a TypeError if ever actually called -- not copied here.
    reference_obs_components = {
        "historical_max_coords_obs": MdpComponent(
            compute_func=compute_historical_max_coords_from_motion_lib,
            dynamic_vars={},
            static_params={
                "history_steps": HISTORY_STEPS,
                "num_state_history_steps": max(HISTORY_STEPS),
                "local_obs": True,
                "root_height_obs": True,
            },
        ),
        "morphology_obs": MdpComponent(
            compute_func=compute_morphology_from_motion_lib,
            dynamic_vars={},
            static_params={},
        ),
    }

    agent_config: AMPAgentConfig = AMPAgentConfig(
        model=AMPModelConfig(
            in_keys=actor_in_keys + disc_in_keys,
            out_keys=["action", "mean_action", "neglogp", "value", "disc_logits", "disc_value"],
            actor=actor_config,
            critic=critic_config,
            discriminator=discriminator_config,
            disc_critic=disc_critic_config,
            actor_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=2e-5),
            critic_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=1e-4),
            discriminator_optimizer=OptimizerConfig(
                _target_="torch.optim.Adam", lr=1e-4
            ),
            disc_critic_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=1e-4),
        ),
        reference_obs_components=reference_obs_components,
        batch_size=args.batch_size,
        # PHC-anchored 0.5/0.5 blend: keep explicit tracking reward alongside the
        # discriminator's style reward, unlike add/mlp.py and amp/mlp.py's task_reward_w=0.0
        # (pure style) -- this project still needs to reproduce a *specific* retargeted
        # clip per env, not just "some natural human motion".
        task_reward_w=0.5,
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
        amp_parameters=AMPParametersConfig(
            discriminator_reward_w=0.5,
            discriminator_reward_threshold=0.03,
        ),
    )
    return agent_config


def configure_robot_and_simulator(
    robot_cfg: RobotConfig, simulator_cfg: SimulatorConfig, args: argparse.Namespace
):
    """Configure robot to add contact sensors for foot contact tracking.

    Unmodified from mlp_wide.py -- deliberately does NOT disable mass_scaled_gains, so this
    run inherits smpl_mor's default per-segment PD gain scaling (note.md #44), same as the
    in-flight hhi_wide_150motion_128shape_seggain run. See module docstring.
    """
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
    """Apply evaluation-specific overrides.

    Reuses mlp_wide.py's (disable termination, resample_on_reset, init_start_prob=1.0) and
    amp/mlp.py's (zero discriminator_reward_threshold so eval isn't early-terminated by the
    discriminator) overrides, same two-call pattern as examples/experiments/add/mlp.py.
    """
    from protomotions.utils.config_utils import (
        import_experiment_relative_eval_overrides,
    )

    apply_inference_overrides_fn = import_experiment_relative_eval_overrides(
        "mlp_wide.py"
    )
    apply_inference_overrides_fn(
        robot_cfg, simulator_cfg, env_cfg, agent_cfg, terrain_cfg,
        motion_lib_cfg, scene_lib_cfg, args,
    )

    apply_inference_overrides_fn = import_experiment_relative_eval_overrides(
        "../amp/mlp.py"
    )
    apply_inference_overrides_fn(
        robot_cfg, simulator_cfg, env_cfg, agent_cfg, terrain_cfg,
        motion_lib_cfg, scene_lib_cfg, args,
    )
