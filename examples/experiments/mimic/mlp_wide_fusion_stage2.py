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
Stage 2 v4 — Shape Transfer via Frozen Backbone + Concat-Fusion Adapter
===========================================================================

Same as mlp_wide_lora_stage2.py (v3, shape-only residual adapter) in every respect except the
actor's mu_model is FusionAdapterMLPWithConcatConfig instead of ResidualAdapterMLPWithConcatConfig:

    beta_embed = beta_encoder(morphology_obs)                  # small MLP, betas ONLY
    trunk_out  = frozen_trunk(obs)                              # unchanged Stage 1 weights
    delta      = fusion_mlp(concat(trunk_out, beta_embed))      # small trainable MLP, TRAINABLE
    output     = trunk_out + delta

v3 (`hhi_wide_residual_stage2`, wandb `48j007hg`) restricted the adapter to `morphology_obs`
only, which made `delta` provably constant within an episode (jerk-free by construction) but
can only represent a per-body constant offset -- confirmed underperforming the earlier
full-obs-input variant (`3skv3b2g`, plateaued ~78%) in early training (44.5% vs 65.9% success
at epoch 13799). This v4 lets the fusion head see the frozen trunk's own output (which is
pose-dependent, changes every timestep) so it can learn pose-shape interaction corrections
that v3 structurally cannot -- at the cost of reopening the frame-to-frame jerk risk that
motivated v3's restriction in the first place. See
`protomotions/agents/common/fusion_adapter_mlp.py`'s module docstring and
note/README.note.md #35 ("v4 candidate") for full rationale.

To control jerk without relying on the architecture alone, `action_smoothness` reward weight
is bumped from -0.02 (v2/v3) to -0.1 here -- the single other change from mlp_wide_lora_stage2.py.

Critic is unchanged (same 4x1024 as mlp_wide.py) and fully fine-tuned, unfrozen, no adapter.

Requires --checkpoint pointing at a Stage 1 checkpoint that has been run through
tools/reset_morphology_normalizer.py first (reuse the existing last_morph_reset.ckpt). Loads
via allow_partial_checkpoint_load=True since the new beta_encoder/fusion_mlp keys don't exist
in that checkpoint.

    python protomotions/train_agent.py \\
        --robot-name smpl_mor --simulator isaacgym \\
        --experiment-path examples/experiments/mimic/mlp_wide_fusion_stage2.py \\
        --experiment-name hhi_wide_fusion_stage2 \\
        --checkpoint results/hhi_wide_20946_neutral/last_morph_reset.ckpt \\
        --r2-motion-source r2:proto-data/hhi_stage2/ \\
        --motion-cache-dir /workspace/motion_cache \\
        --epochs-per-shard 64 \\
        --num-envs 6144 --batch-size 24576 --ngpu 6

Verification checklist before trusting a new run's numbers (mirrors #32/#35's checklist):
    1. CPU dummy-forward: frozen params (norm, mlp) get zero grad, adapter params
       (beta_encoder, fusion_mlp) get nonzero grad, delta == 0 at init.
    2. Load an hhi_wide_20946_neutral checkpoint with strict=False; missing_keys should be
       exactly {beta_encoder.*, fusion_mlp.*}, unexpected_keys empty; output should match
       mlp_wide.py's output bit-for-bit at that checkpoint (delta == 0 at init).
    3. With morphology_obs held fixed, delta should still change when pose/motion-target/
       previous-action inputs change (confirms fusion_mlp is actually pose-dependent, unlike
       v3) -- this is the whole point of v4, opposite of v3's invariance check.
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
BETA_ENCODER_UNITS = 128  # width of the beta-only embedding MLP
FUSION_UNITS = 512  # width of the fusion head's hidden layers
FUSION_NUM_LAYERS = 2


def additional_experiment_arguments(parser: argparse.ArgumentParser):
    """Add Stage 2 streaming-data CLI arguments.

    Two mutually exclusive alternative motion sources, checked in this order by
    motion_lib_config() below:

    1. --global-clip-pool-source (note/README.stage2-global-clip-sampling-plan.md): a persistent,
       globally-weighted resident pool of clips (GlobalClipPoolConfig) -- the current mechanism,
       tracks clip difficulty for the life of the run and survives resumes.
    2. --r2-motion-source (note/README.stage2-streaming-loader-plan.md): the older shard-by-shard
       rotation (StreamingMotionLibConfig), which resets its difficulty weights every rotation --
       kept only for in-flight runs/resumes already using it, not for new runs.

    Neither set -> falls back to --motion-file. This keeps the smoke-test command (small
    hhi_stage1_merged6 data, --motion-file) working unchanged.
    """
    parser.add_argument(
        "--global-clip-pool-source",
        type=str,
        default=None,
        help=(
            "rclone remote directory of per-clip Stage 2 motion files + clip_manifest.jsonl "
            "(e.g. r2:proto-data/hhi_stage2_per_clip/). When set, uses the persistent global "
            "clip-priority pool instead of --r2-motion-source/--motion-file."
        ),
    )
    parser.add_argument(
        "--global-clip-pool-size", type=int, default=256,
        help="Number of clips resident per rank (K). Matches the validated hhi_1024_motion "
        "pilot's per-rank footprint (1024 clips / 4 ranks).",
    )
    parser.add_argument(
        "--global-clip-pool-cache-dir", type=str, default="/workspace/motion_cache",
        help="Local directory to cache downloaded per-clip files.",
    )
    parser.add_argument(
        "--global-clip-pool-cache-multiplier", type=float, default=3.0,
        help="Local disk cache sized at this multiple of K, for download hysteresis.",
    )
    parser.add_argument(
        "--global-clip-pool-rebuild-every", type=int, default=64,
        help="Epochs between resident-pool rebuilds. Deliberately decoupled from "
        "--eval-metrics-every (weight updates still need real eval rollouts and stay on that "
        "cadence); rebuilds are cheap and default to matching today's shard rotation cadence "
        "(--epochs-per-shard) so new/unproven clips get pulled in at least as often.",
    )
    parser.add_argument(
        "--global-clip-pool-shuffle-seed", type=int, default=42,
        help="Seed for the deterministic clip shuffle/rank-partition. Must stay fixed across "
        "a run (including resume).",
    )
    parser.add_argument(
        "--global-clip-pool-exploration-coefficient", type=float, default=1.0,
        help="Scale of the UCB-style exploration bonus added to a clip's selection priority.",
    )
    parser.add_argument(
        "--r2-motion-source",
        type=str,
        default=None,
        help=(
            "rclone remote directory of per-shard Stage 2 motion files "
            "(e.g. r2:proto-data/hhi_stage2/). When set (and --global-clip-pool-source is not), "
            "streams shards shard-by-shard instead of using --motion-file. Superseded by "
            "--global-clip-pool-source for new runs -- kept for in-flight runs/resumes."
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
    if getattr(args, "global_clip_pool_source", None):
        from protomotions.components.global_clip_pool import GlobalClipPoolConfig

        return GlobalClipPoolConfig(
            r2_source=args.global_clip_pool_source,
            local_cache_dir=args.global_clip_pool_cache_dir,
            resident_pool_size=args.global_clip_pool_size,
            cache_size_multiplier=args.global_clip_pool_cache_multiplier,
            pool_rebuild_every=args.global_clip_pool_rebuild_every,
            clip_partition_shuffle_seed=args.global_clip_pool_shuffle_seed,
            exploration_bonus_coefficient=args.global_clip_pool_exploration_coefficient,
        )
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
        # Bumped from -0.02 (v2/v3) to -0.1: v4's fusion head sees the (pose-dependent,
        # per-timestep) trunk output, reopening the jerk risk v3's betas-only restriction
        # avoided by construction -- see module docstring.
        "action_smoothness": action_smoothness_factory(weight=-0.1),
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
    from protomotions.agents.common.fusion_adapter_mlp import FusionAdapterMLPWithConcatConfig
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
        mu_model=FusionAdapterMLPWithConcatConfig(
            in_keys=actor_in_keys,
            normalize_obs=True,
            norm_clamp_value=5,
            out_keys=["actor_trunk_out"],
            num_out=robot_config.number_of_actions,
            layers=[MLPLayerConfig(units=WIDE_UNITS, activation="relu") for _ in range(6)],
            beta_encoder_layers=[
                MLPLayerConfig(units=BETA_ENCODER_UNITS, activation="relu"),
            ],
            fusion_layers=[
                MLPLayerConfig(units=FUSION_UNITS, activation="relu") for _ in range(FUSION_NUM_LAYERS)
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
            # 5-10x lower than mlp_wide.py's 2e-5 -- only beta_encoder/fusion_mlp are
            # trainable on the actor side, so this LR governs a much smaller optimization
            # problem. Same value as v2/v3 -- v2's own diagnosis showed clip_frac was healthy
            # at this LR once the adapter had a direct gradient path (which this design keeps).
            actor_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=4e-6),
            # Critic has no frozen prior to protect; unchanged from mlp_wide.py.
            critic_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=1e-4),
        ),
        batch_size=args.batch_size,
        training_max_steps=args.training_max_steps,
        gradient_clip_val=50.0,
        clip_critic_loss=True,
        # Stage 1 checkpoint has no beta_encoder/fusion_mlp keys -- see note/README.note.md #32.
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
