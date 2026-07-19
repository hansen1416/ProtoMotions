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
Stage 2 — Shape Transfer via Frozen Backbone + Residual Adapter
===========================================================================

Same as mlp_wide.py in every respect except the actor's mu_model is
ResidualAdapterMLPWithConcatConfig instead of MLPWithConcatConfig: the 6x2896 trunk is
warm-started from a Stage 1 checkpoint (hhi_wide_20946_neutral) and frozen, and a small
trainable MLP adds a residual correction on top. Design doc, rationale, and code-level notes:
note/README.note.md #32, #34, #35.

As of 2026-07-19 (#35) the adapter's input defaults to `morphology_obs` ONLY (the 11-dim
`[gender_id, betas/3.0]`), not the trunk's full pose/motion-target/previous-action input --
see `protomotions/agents/common/residual_adapter_mlp.py`'s module docstring and
`ResidualAdapterMLPWithConcatConfig.adapter_in_keys`. This makes `delta` constant within an
episode (morphology_obs doesn't change mid-episode), so the adapter cannot itself introduce
frame-to-frame jerk -- diagnosed as a real problem in the prior (full-obs-input) variant,
which plateaued at ~78% success_rate under wandb run `hhi_wide_residual_stage2` / `3skv3b2g`
with the adapter contributing ~40% of the trunk's output magnitude.

Critic is unchanged (same 4x1024 as mlp_wide.py) and fully fine-tuned, unfrozen, no adapter --
no Stage-1 prior to protect there, matches the existing "critic stays flat-concat" precedent
from #18.

Requires --checkpoint pointing at a Stage 1 checkpoint that has been run through
tools/reset_morphology_normalizer.py first (obs-normalizer saturation fix, orthogonal to the
architecture change here -- still needed). Loads via allow_partial_checkpoint_load=True since
the new adapter_mlp keys don't exist in that checkpoint (see
base_agent/config.py's allow_partial_checkpoint_load).

    # 1. One-time: reset the morphology obs-normalizer dims on the Stage 1 checkpoint
    #    (already done previously -- reuse the existing last_morph_reset.ckpt if present,
    #    don't regenerate it).
    python tools/reset_morphology_normalizer.py \\
        --checkpoint results/hhi_wide_20946_neutral/last.ckpt \\
        --output results/hhi_wide_20946_neutral/last_morph_reset.ckpt

    # 2. Full Stage 2 run, streaming the 328 shards shard-by-shard from R2 (~1.1 TB total,
    #    too big to fit in RAM/VRAM at once -- see note/README.stage2-streaming-loader-plan.md
    #    and note/README.note.md #35 for why a single non-rotating load isn't feasible here).
    #    Each rank rotates through its own slice of the remote file list; --motion-file is
    #    not used in this mode. New experiment name each time (see resume-mode gotcha in
    #    CLAUDE.md) -- this one tests the shape-only adapter input (#35):
    python protomotions/train_agent.py \\
        --robot-name smpl_mor --simulator isaacgym \\
        --experiment-path examples/experiments/mimic/mlp_wide_lora_stage2.py \\
        --experiment-name hhi_wide_residual_stage2_shapeonly \\
        --checkpoint results/hhi_wide_20946_neutral/last_morph_reset.ckpt \\
        --r2-motion-source r2:proto-data/hhi_stage2/ \\
        --motion-cache-dir /workspace/motion_cache \\
        --epochs-per-shard 64 \\
        --num-envs 6144 --batch-size 24576 --ngpu 6

Verification checklist before trusting a new run's numbers (note/README.note.md #32, #35):
    1. CPU dummy-forward: frozen params get zero grad, adapter params get nonzero grad,
       delta == 0 at init, different morphology_obs -> different delta; delta must NOT change
       if pose/motion-target/previous-action inputs change with morphology_obs held fixed
       (confirms the adapter is actually shape-only post-#35).
    2. Load an hhi_wide_20946_neutral checkpoint with strict=False; missing_keys should be
       exactly {adapter_mlp.*}, unexpected_keys empty; output should match mlp_wide.py's
       output bit-for-bit at that checkpoint (delta == 0 at init).
"""
from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig
from protomotions.components.terrains.config import TerrainConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.agents.ppo.config import PPOAgentConfig
from protomotions.components.scene_lib import SceneLibConfig
from protomotions.components.motion_lib import MotionLibConfig
import argparse

WIDE_UNITS = 2896  # must match the Stage 1 checkpoint's trunk width (mlp_wide.py)
ADAPTER_UNITS = 512  # width of the small trainable residual-adapter MLP's hidden layers
ADAPTER_NUM_LAYERS = 2


def additional_experiment_arguments(parser: argparse.ArgumentParser):
    """Add Stage 2 streaming-data CLI arguments (note/README.stage2-streaming-loader-plan.md).

    When --r2-motion-source is set, motion_lib_config() below returns a
    StreamingMotionLibConfig that streams shards from R2 instead of loading --motion-file.
    This keeps the smoke-test command (2a, small hhi_stage1_merged6 data, --motion-file) working
    unchanged -- streaming only turns on for the full Stage 2 run (2b).
    """
    parser.add_argument(
        "--r2-motion-source",
        type=str,
        default=None,
        help=(
            "rclone remote directory of per-shard Stage 2 motion files "
            "(e.g. r2:proto-data/hhi_stage2/). When set, streams shards shard-by-shard "
            "instead of using --motion-file."
        ),
    )
    parser.add_argument(
        "--motion-cache-dir",
        type=str,
        default="/workspace/motion_cache",
        help="Local directory to cache downloaded Stage 2 shards (2 files per rank kept).",
    )
    parser.add_argument(
        "--epochs-per-shard",
        type=int,
        default=64,
        help="Number of training epochs to run on each streamed shard before rotating.",
    )
    parser.add_argument(
        "--shard-shuffle-seed",
        type=int,
        default=42,
        help="Seed for the deterministic shard shuffle/rank-partition.",
    )


def terrain_config(args: argparse.Namespace):
    return TerrainConfig()


def scene_lib_config(args: argparse.Namespace):
    scene_file = args.scenes_file if hasattr(args, "scenes_file") else None
    return SceneLibConfig(scene_file=scene_file)


def motion_lib_config(args: argparse.Namespace):
    if getattr(args, "r2_motion_source", None):
        from protomotions.components.motion_lib_pool import StreamingMotionLibConfig

        return StreamingMotionLibConfig(
            r2_source=args.r2_motion_source,
            local_cache_dir=args.motion_cache_dir,
            epochs_per_shard=args.epochs_per_shard,
            shard_shuffle_seed=args.shard_shuffle_seed,
        )
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
    from protomotions.agents.common.residual_adapter_mlp import ResidualAdapterMLPWithConcatConfig
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

    actor_in_keys = ["max_coords_obs", "mimic_target_poses", "previous_actions", "morphology_obs"]

    actor_config = PPOActorConfig(
        num_out=robot_config.kinematic_info.num_dofs,
        actor_logstd=-2.9,
        in_keys=actor_in_keys,
        mu_key="actor_trunk_out",
        mu_model=ResidualAdapterMLPWithConcatConfig(
            in_keys=actor_in_keys,
            normalize_obs=True,
            norm_clamp_value=5,
            out_keys=["actor_trunk_out"],
            num_out=robot_config.number_of_actions,
            layers=[MLPLayerConfig(units=WIDE_UNITS, activation="relu") for _ in range(6)],
            adapter_layers=[
                MLPLayerConfig(units=ADAPTER_UNITS, activation="relu") for _ in range(ADAPTER_NUM_LAYERS)
            ],
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
            # 5-10x lower than mlp_wide.py's 2e-5 -- only adapter_mlp is trainable on the
            # actor side, so this LR governs a much smaller optimization problem.
            actor_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=4e-6),
            # Critic has no frozen prior to protect; unchanged from mlp_wide.py.
            critic_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=1e-4),
        ),
        batch_size=args.batch_size,
        training_max_steps=args.training_max_steps,
        gradient_clip_val=50.0,
        clip_critic_loss=True,
        # Stage 1 checkpoint has no adapter_down/hypernet keys -- see note/README.note.md #32.
        allow_partial_checkpoint_load=True,
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
