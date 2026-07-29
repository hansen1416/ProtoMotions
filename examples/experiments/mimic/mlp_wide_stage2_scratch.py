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
Stage 2 from scratch — plain wide trunk, no adapter, full 128-shape data
===========================================================================

Abandons the Stage 2 adapter/warm-start lineage (v1-v6, see `mlp_wide_fusion_stage2*.py` and
note/README.note.md #32-40): every architecture tried there — frozen backbone, unfrozen backbone,
full-concat adapter, shape-only adapter, fusion adapter, +1 deepened layer — plateaued at
~78-82% `eval/success_rate`, while the single-shape `hhi_wide_20946_neutral` trunk (same width,
same obs, no adapter) reaches 95-97%. This experiment trains that same plain architecture
(`mlp_wide.py`'s `MLPWithConcatConfig`, 6x2896, `morphology_obs` already flat-concatenated into
`actor_in_keys`) from scratch directly on the full 128-shape Stage 2 data, instead of warm-starting
an adapter on top of a single-shape-converged trunk. No `--checkpoint`, no capacity change —
`morphology_obs` is only an 11-dim (gender + 10 SMPL betas) continuous input, so 128 shapes are 128
samples along a smooth low-dimensional manifold, not 128 unrelated tasks the way 20,946 largely-
unrelated motion clips were (the diversity axis that originally justified widening 1024->2896).

Before trusting this run's numbers, two real problems in the shared data/eval pipeline
(`protomotions/components/global_clip_pool.py`) were fixed — every v1-v6 run shared these
unfixed, so none of their `eval/success_rate` numbers were fully trustworthy either:

1. **Resident-pool sizing was never re-derived for this dataset.** `--global-clip-pool-size 256`
   was carried over unchanged from the `hhi_1024_motion` pilot, where 256 clips/rank was the
   *entire* per-rank vocabulary (100% resident). At Stage 2's real per-rank vocabulary (~3,492
   clips), 256 is only ~7.3% resident at once. Real scoreboard data pulled from a live v5 run
   (107 rebuild cycles) showed 61% of clips visited exactly twice ever vs. 38 clips visited 100+
   times — a ~50x exposure disparity, since the priority scoreboard deprioritizes a clip the
   moment it succeeds and only a slow log-scale UCB bonus brings it back. Two new knobs address
   this without needing a bigger resident pool: `--global-clip-pool-weight-floor` (a solved clip's
   weight never fully vanishes) and `--global-clip-pool-random-fraction` (a bounded fraction of
   each rebuild's slots is pure rehearsal, independent of priority). `--global-clip-pool-rebuild-
   every` is also bumped 64->256 (4x longer residency per rebuild) as an easy first ablation step.

2. **`eval/success_rate` was measured on the resident pool itself, not a fixed holdout.**
   `MimicEvaluator.motion_lib` is the live `GlobalClipPool`, and `eval_one_shape_per_motion` covers
   whatever's *currently resident* — which is priority-biased toward hard/unproven clips by
   construction, making the metric a moving, harder-than-average subsample rather than a stable
   estimate of true full-dataset performance. Fixed via `--global-clip-pool-eval-holdout-size`: a
   fixed set of clips permanently excluded from the trainable vocabulary (never selected, never
   trained on, never influences the curriculum), evaluated as a second, read-only pass each eval
   tick (`MimicEvaluator._evaluate_holdout`) and logged under separate `eval_holdout/*` keys.
   **`eval_holdout/success_rate` is the metric to trust for this run, not `eval/success_rate`**
   (the latter is kept for curriculum-health visibility only, same meaning as before).

Full design/discussion: this session, will be written up in note/README.note.md.

    python protomotions/train_agent.py \\
        --robot-name smpl_mor --simulator isaacgym \\
        --experiment-path examples/experiments/mimic/mlp_wide_stage2_scratch.py \\
        --experiment-name hhi_wide_stage2_scratch \\
        --global-clip-pool-source r2:proto-data/hhi_stage2_per_clip/ \\
        --global-clip-pool-cache-dir /workspace/motion_cache \\
        --global-clip-pool-size 256 \\
        --global-clip-pool-rebuild-every 256 \\
        --global-clip-pool-eval-holdout-size 128 \\
        --global-clip-pool-weight-floor 0.05 \\
        --global-clip-pool-random-fraction 0.2 \\
        --num-envs 6144 --batch-size 24576 --ngpu 6
"""
from protomotions.robot_configs.base import RobotConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.agents.ppo.config import PPOAgentConfig
import argparse

# Re-exported for compatibility with the config-builder's optional-hook lookups and for anyone
# importing this file's env_config directly -- not used to build the trunk here (see agent_config).
from examples.experiments.mimic.mlp_wide import (
    WIDE_UNITS,
    env_config,
)
from examples.experiments.mimic.mlp_wide_fusion_stage2 import (
    terrain_config,
    scene_lib_config,
    configure_robot_and_simulator,
    apply_inference_overrides,
)

__all__ = [
    "additional_experiment_arguments",
    "terrain_config",
    "scene_lib_config",
    "motion_lib_config",
    "env_config",
    "configure_robot_and_simulator",
    "apply_inference_overrides",
    "agent_config",
]


def additional_experiment_arguments(parser: argparse.ArgumentParser):
    """GlobalClipPool CLI arguments -- same block as `mlp_wide_fusion_stage2.py`'s (kept as an
    independent copy there rather than edited in place, to keep the v1-v6 adapter lineage
    reproducible), extended with 3 new opt-in flags for the data/eval pipeline fixes described in
    this file's module docstring. All 3 default to exactly the old GlobalClipPool behavior when
    omitted (`eval_holdout_size=0`, `weight_floor=0.0`, `random_fraction=0.0`).
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
        help="Epochs between resident-pool rebuilds. Bumped from the old default of 64 -- 4x "
        "longer residency per rebuild, a first easy-to-defend ablation step given the weight "
        "floor / random-rehearsal fixes below also now guarantee rehearsal independent of cadence.",
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
        "keys. 0 disables the holdout (old behavior). See module docstring, fix #2.",
    )
    parser.add_argument(
        "--global-clip-pool-weight-floor", type=float, default=0.05,
        help="Minimum value global_clip_weights can EMA-decay to on repeated success -- a "
        "'solved' clip keeps a nonzero selection priority forever. 0.0 = old unbounded-decay "
        "behavior. See module docstring, fix #1.",
    )
    parser.add_argument(
        "--global-clip-pool-random-fraction", type=float, default=0.2,
        help="Fraction of each rebuild's K resident slots filled by pure uniform random "
        "selection instead of priority. 0.0 = old fully-priority-based behavior. See module "
        "docstring, fix #1.",
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


def agent_config(
    robot_config: RobotConfig, env_config: EnvConfig, args: argparse.Namespace
) -> PPOAgentConfig:
    """Identical to `mlp_wide.py`'s `agent_config` -- same plain 6x2896 `MLPWithConcatConfig`
    trunk, no adapter, actor lr `2e-5` (real from-scratch training, not the `4e-6` used for
    adapter fine-tuning in the v1-v6 lineage). Only `evaluator.eval_metrics_every` differs, bumped
    to 256 to match v5/v6's already-validated pool-training cadence (mlp_wide.py's own default of
    200 was tuned for static-data training, not GlobalClipPool rebuilds).
    """
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

    actor_in_keys = ["max_coords_obs", "mimic_target_poses", "previous_actions", "morphology_obs"]

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
