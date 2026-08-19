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
Mimic Environment — Full-Scale Stage 2: Self-Attention Temporal Encoder + Discover Reward/Term
(20,946 clips x 128 shapes, GlobalClipPool)

Full-dataset counterpart to `mlp_wide_discover_attention.py`, which was validated on the 150-clip
corpus (`hhi_wide_150motion_discover_attention`, wandb `p6b2la4h`: 84.7% eval/success_rate vs.
baseline `discover`'s 72% at matched steps, still climbing, smoothest actor/critic losses of the
7-run `discover` lineage -- see `note/README.note.md` §61) and reviewed in detail in §62 (obs
shapes, reward, architecture, PPO update mechanics, and one flagged gap: the shared `Transformer`
class has no positional encoding despite its docstring implying one -- deliberately NOT fixed here,
per the discussion in this session: unlikely to move the success-rate ceiling since token *type*
already carries the coarse temporal signal and the observed plateau looks like an intrinsic
hard-motion-cluster capacity limit, not a temporal-precision one).

**Two changes from `mlp_wide_discover_attention.py`, both about data source, not architecture:**

1. `env_config()`, `agent_config()`'s attention trunk, `configure_robot_and_simulator()`, and
   `apply_inference_overrides()` are byte-identical -- same `HISTORY_STEPS`/`FUTURE_STEPS`, same
   attention-stage sizing, same discover-lineage reward/termination (fall-only termination, no
   action_smoothness/pow_rew/contact_match_rew -- see `mlp_wide_discover.py`'s docstring for why:
   CMU's getting-up-policies two-phase curriculum, relax every constraint that isn't "did you get
   anywhere near the reference" so the policy's only job is finding ANY successful trajectory
   through the hardest clips). That relaxation was validated on the 150-clip and 13-clip subsets
   only, never before at full 20,946-clip scale -- carried forward as-is by explicit decision
   (2026-08-19 session) rather than reverting to `mlp_wide.py`'s stricter config, since the point
   of scaling up is carrying the whole validated recipe forward, not cherry-picking pieces of it.
2. `motion_lib_config()` and `additional_experiment_arguments()` are new, copied from
   `mlp_wide_stage2_scratch.py`'s `GlobalClipPool` wiring (same pattern used successfully for
   `hhi_wide_stage2_scratch`, wandb `hl2g9b0m`) -- streams per-clip motion files from R2 instead of
   loading one static `.pt` file, with a resident training pool, a fixed eval holdout immune to
   resident-set churn, a weight floor so solved clips keep nonzero rehearsal priority, and a random
   rehearsal fraction. Falls back to the plain static `MotionLibConfig` if
   `--global-clip-pool-source` isn't passed, so this file still works unchanged for small-scale
   testing (e.g. the smoke test recommended below).
   `evaluator.eval_metrics_every` is bumped from `mlp_wide_discover_attention.py`'s implicit
   default of 200 (tuned for static-data eval cadence) to 256, matching
   `--global-clip-pool-rebuild-every`'s default -- same reasoning as `mlp_wide_stage2_scratch.py`.

**Bundled-change caveat, worth remembering when reading this run's results against `hl2g9b0m`:**
this file differs from that full-scale reference point in FOUR ways at once -- historical/lookahead
observations, self-attention encoding, inherited `mass_scaled_gains=True` (a `smpl_mor` default
that didn't exist when `hl2g9b0m` launched), and the discover-lineage relaxed reward/termination
(also untested at full scale before now). If this run beats `hl2g9b0m`, the gain can't be cleanly
attributed to attention alone.

**Untested combination, worth a smoke test before the full 6-GPU launch:** no prior experiment
combines `GlobalClipPool` (streamed per-clip source) with the dilated historical/lookahead
observation window. Recommend a short run first (small `--global-clip-pool-size`, few envs) to
confirm a freshly-streamed clip has enough buffered past/future frames before committing the full
run below.

Full-scale launch, matching `hl2g9b0m`'s GlobalClipPool settings and env/batch sizing:

    nohup python -u protomotions/train_agent.py \\
    --robot-name smpl_mor --simulator isaacgym \\
    --experiment-path examples/experiments/mimic/mlp_wide_stage2_discover_attention.py \\
    --experiment-name hhi_wide_stage2_discover_attention \\
    --global-clip-pool-source r2:proto-data/hhi_stage2_per_clip/ \\
    --global-clip-pool-cache-dir /workspace/motion_cache \\
    --global-clip-pool-size 256 --global-clip-pool-rebuild-every 256 \\
    --global-clip-pool-eval-holdout-size 128 --global-clip-pool-weight-floor 0.05 \\
    --global-clip-pool-random-fraction 0.2 \\
    --num-envs 6144 --batch-size 24576 --ngpu 6 \\
    --use-wandb --wandb-project hhi-protomotions --wandb-entity yugoamaryl \\
    --wandb-group hhi_wide_stage2_discover_attention > /tmp/hhi_wide_stage2_discover_attention.log 2>&1 &
"""
from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig
from protomotions.components.terrains.config import TerrainConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.agents.ppo.config import PPOAgentConfig
from protomotions.components.scene_lib import SceneLibConfig
import argparse

WIDE_UNITS = 2896  # 1024 * sqrt(8) -- same as mlp_wide.py, kept for architecture parity

# Dilated backward history steps (actor + critic observation). Same schedule already validated
# in examples/experiments/amp/mlp.py, mlp_wide_amp.py, and mlp_wide_discover_historical.py.
HISTORY_STEPS = [1, 2, 3, 4, 8, 16, 32]

# Dilated forward-lookahead steps for the mimic target-pose observation. Same schedule already
# validated in examples/experiments/mimic/mlp_bm_l2c2.py and mlp_wide_discover_lookahead.py.
FUTURE_STEPS = [1, 2, 4, 8]

NUM_HISTORY_STEPS = len(HISTORY_STEPS)  # 7
NUM_FUTURE_STEPS = len(FUTURE_STEPS)  # 4

# Self-attention stage sizing -- first guess, not tuned (see mlp_wide_discover_attention.py's
# module docstring for the full discussion). Unchanged here -- this file only changes data source.
TOKEN_SIZE = 256
TOKEN_ENCODER_WIDTH = 256
NUM_HEADS = 4
FF_SIZE = 1024
NUM_ATTN_LAYERS = 2


def terrain_config(args: argparse.Namespace):
    return TerrainConfig()


def scene_lib_config(args: argparse.Namespace):
    scene_file = args.scenes_file if hasattr(args, "scenes_file") else None
    return SceneLibConfig(scene_file=scene_file)


def additional_experiment_arguments(parser: argparse.ArgumentParser):
    """GlobalClipPool CLI arguments -- same block as `mlp_wide_stage2_scratch.py`'s. All 3 fix
    flags (eval-holdout-size, weight-floor, random-fraction) default to exactly the old
    GlobalClipPool behavior when omitted (0, 0.0, 0.0 respectively), matching that file's
    already-validated defaults for this exact dataset.
    """
    parser.add_argument(
        "--global-clip-pool-source",
        type=str,
        default=None,
        help=(
            "rclone remote directory of per-clip Stage 2 motion files + clip_manifest.jsonl "
            "(e.g. r2:proto-data/hhi_stage2_per_clip/)."
        ),
    )
    parser.add_argument(
        "--global-clip-pool-size", type=int, default=256,
        help="Number of clips resident per rank (K) for training.",
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
        "--global-clip-pool-rebuild-every", type=int, default=256,
        help="Epochs between resident-pool rebuilds.",
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
        "--global-clip-pool-selection-temperature", type=float, default=1.0,
        help="Softmax temperature over priority rank used to sample the resident set.",
    )
    parser.add_argument(
        "--global-clip-pool-weight-ema-alpha", type=float, default=0.1,
        help="EMA blend factor for global_clip_weights updates.",
    )
    parser.add_argument(
        "--global-clip-pool-difficulty-scores-path", type=str,
        default="data/preprocessing/valid_ids_sorted_by_difficulty.txt",
        help="Path to a 'clip_id: score' file, ascending easy->hard, used to seed "
        "global_clip_weights instead of a flat 1.0.",
    )
    parser.add_argument(
        "--global-clip-pool-eval-holdout-size", type=int, default=128,
        help="Clips/rank permanently excluded from training and reserved for a fixed "
        "generalization probe, evaluated separately each eval tick under eval_holdout/* metric "
        "keys. 0 disables the holdout (old behavior).",
    )
    parser.add_argument(
        "--global-clip-pool-weight-floor", type=float, default=0.05,
        help="Minimum value global_clip_weights can EMA-decay to on repeated success -- a "
        "'solved' clip keeps a nonzero selection priority forever. 0.0 = old unbounded-decay "
        "behavior.",
    )
    parser.add_argument(
        "--global-clip-pool-random-fraction", type=float, default=0.2,
        help="Fraction of each rebuild's K resident slots filled by pure uniform random "
        "selection instead of priority. 0.0 = old fully-priority-based behavior.",
    )


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
            selection_temperature=args.global_clip_pool_selection_temperature,
            weight_ema_alpha=args.global_clip_pool_weight_ema_alpha,
            difficulty_scores_path=args.global_clip_pool_difficulty_scores_path,
            eval_holdout_size=args.global_clip_pool_eval_holdout_size,
            weight_floor=args.global_clip_pool_weight_floor,
            random_fraction=args.global_clip_pool_random_fraction,
        )
    from protomotions.components.motion_lib import MotionLibConfig

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
            future_steps=FUTURE_STEPS,
        )
    }

    observation_components = {
        "max_coords_obs": max_coords_obs_factory(),
        # Backward: dilated window of past body-state frames.
        "historical_max_coords_obs": historical_max_coords_obs_factory(
            local_obs=True,
            root_height_obs=True,
            observe_contacts=False,
            history_steps=HISTORY_STEPS,
        ),
        "previous_actions": previous_actions_factory(history_steps=1),
        # Forward: same factory call as discover/lookahead -- resolves to "all available future
        # steps," which now means the FUTURE_STEPS-dilated window instead of a single next-step
        # frame.
        "mimic_target_poses": mimic_target_poses_max_coords_factory(with_velocities=True),
        "morphology_obs": morphology_obs_factory(),
    }

    # Sole termination is a genuine fall/collapse check -- no tracking-error termination at all,
    # so an ugly-but-upright trajectory is never cut short just for drifting from the reference.
    # Discover-lineage relaxation (mlp_wide_discover.py), carried forward to full scale as-is.
    termination_components = {
        "fall": fall_term_factory(termination_height=0.15),
    }

    # Only the tracking-shaping reward remains -- action_smoothness/pow_rew/contact_match_rew
    # (effort, smoothness, contact-timing penalties) are omitted entirely, same discover-lineage
    # relaxation as termination above.
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
    from protomotions.agents.common.config import (
        MLPWithConcatConfig,
        MLPLayerConfig,
        ModuleContainerConfig,
        TransformerConfig,
        ModuleOperationReshapeConfig,
        ModuleOperationForwardConfig,
    )
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

    raw_obs_keys = [
        "max_coords_obs",
        "historical_max_coords_obs",
        "mimic_target_poses",
        "previous_actions",
        "morphology_obs",
    ]

    def build_attention_trunk(final_layer_units, final_num_out, final_out_key):
        """Token-encode + self-attend the temporal obs, then feed a flat-MLP head.

        Final head width/depth is caller-supplied and kept identical to the flat-concat
        baseline's trunk -- the token encoders + transformer here are additive preprocessing,
        not a replacement for that capacity. Byte-identical to
        mlp_wide_discover_attention.py's build_attention_trunk.
        """
        return ModuleContainerConfig(
            in_keys=raw_obs_keys,
            out_keys=[final_out_key],
            models=[
                # Per-frame token encoders. Reshape recovers the [batch, steps, dim] sequence
                # from the flat dilated-frame tensor; ModuleOperationForwardConfig then applies
                # the (shared-weight) MLP per-frame automatically for 3D inputs.
                MLPWithConcatConfig(
                    in_keys=["max_coords_obs"],
                    out_keys=["current_state_token"],
                    normalize_obs=True,
                    norm_clamp_value=5,
                    num_out=TOKEN_SIZE,
                    layers=[
                        MLPLayerConfig(units=TOKEN_ENCODER_WIDTH, activation="relu")
                        for _ in range(2)
                    ],
                    module_operations=[
                        ModuleOperationReshapeConfig(new_shape=["batch_size", 1, -1]),
                        ModuleOperationForwardConfig(),
                    ],
                ),
                MLPWithConcatConfig(
                    in_keys=["historical_max_coords_obs"],
                    out_keys=["history_token"],
                    normalize_obs=True,
                    norm_clamp_value=5,
                    num_out=TOKEN_SIZE,
                    layers=[
                        MLPLayerConfig(units=TOKEN_ENCODER_WIDTH, activation="relu")
                        for _ in range(2)
                    ],
                    module_operations=[
                        ModuleOperationReshapeConfig(
                            new_shape=["batch_size", NUM_HISTORY_STEPS, -1]
                        ),
                        ModuleOperationForwardConfig(),
                    ],
                ),
                MLPWithConcatConfig(
                    in_keys=["mimic_target_poses"],
                    out_keys=["future_token"],
                    normalize_obs=True,
                    norm_clamp_value=5,
                    num_out=TOKEN_SIZE,
                    layers=[
                        MLPLayerConfig(units=TOKEN_ENCODER_WIDTH, activation="relu")
                        for _ in range(2)
                    ],
                    module_operations=[
                        ModuleOperationReshapeConfig(
                            new_shape=["batch_size", NUM_FUTURE_STEPS, -1]
                        ),
                        ModuleOperationForwardConfig(),
                    ],
                ),
                # Self-attention over the 12-token sequence (1 current + 7 history + 4 future).
                # current_state_token is listed first -- Transformer pools output[:, 0, :].
                TransformerConfig(
                    in_keys=["current_state_token", "history_token", "future_token"],
                    out_keys=["attn_out"],
                    transformer_token_size=TOKEN_SIZE,
                    latent_dim=TOKEN_SIZE,
                    num_heads=NUM_HEADS,
                    ff_size=FF_SIZE,
                    num_layers=NUM_ATTN_LAYERS,
                    output_activation="relu",
                ),
                # Final head -- width/depth supplied by caller, unchanged from the flat-concat
                # baseline. previous_actions/morphology_obs are single-vector, not sequential,
                # so they bypass attention and are concatenated directly here, same as they're
                # concatenated into the flat trunk today.
                MLPWithConcatConfig(
                    in_keys=["attn_out", "previous_actions", "morphology_obs"],
                    out_keys=[final_out_key],
                    normalize_obs=True,
                    norm_clamp_value=5,
                    num_out=final_num_out,
                    layers=[
                        MLPLayerConfig(units=units, activation="relu")
                        for units in final_layer_units
                    ],
                ),
            ],
        )

    actor_config = PPOActorConfig(
        num_out=robot_config.kinematic_info.num_dofs,
        actor_logstd=-2.9,
        in_keys=raw_obs_keys,
        mu_key="actor_trunk_out",
        mu_model=build_attention_trunk(
            final_layer_units=[WIDE_UNITS] * 6,
            final_num_out=robot_config.number_of_actions,
            final_out_key="actor_trunk_out",
        ),
    )

    critic_config = build_attention_trunk(
        final_layer_units=[1024] * 4,
        final_num_out=1,
        final_out_key="value",
    )

    agent_config: PPOAgentConfig = PPOAgentConfig(
        model=PPOModelConfig(
            in_keys=raw_obs_keys,
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
            # Bumped from the 200 default (tuned for static-data training) to match
            # --global-clip-pool-rebuild-every's default of 256 -- same reasoning as
            # mlp_wide_stage2_scratch.py.
            eval_metrics_every=256,
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
    motion_lib_cfg,
    scene_lib_cfg: SceneLibConfig,
    args: argparse.Namespace,
):
    """Apply evaluation-specific overrides."""
    if hasattr(env_cfg, "termination_components") and env_cfg.termination_components:
        env_cfg.termination_components = {}

    env_cfg.max_episode_length = 1000000
    env_cfg.motion_manager.resample_on_reset = True
    env_cfg.motion_manager.init_start_prob = 1.0
