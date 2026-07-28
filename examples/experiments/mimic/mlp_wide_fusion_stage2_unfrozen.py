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
Stage 2 v5 — Unfreeze the v4 Fusion-Adapter Backbone
===========================================================================

Same as mlp_wide_fusion_stage2.py (v4) in every respect except one config flag:
`freeze_backbone=False` on the actor's `FusionAdapterMLPWithConcatConfig`.

Why: v4 (`hhi_wide_fusion_stage2_clippool`, wandb `8reyx4ci`/`c9xvetac`) plateaus below 70%
success_rate, clearly worse than v2's ~78% (`3skv3b2g`) despite v4's fusion head having similar
adapter capacity (2x512, same as v2's adapter_mlp). Root cause read as an information
bottleneck, not a capacity or curriculum issue: `fusion_mlp` only ever sees `trunk_out`
(dimension = num_actions, a few dozen) concatenated with a 128-dim beta embedding -- it never
sees the raw pose or motion-target directly, only the frozen trunk's already-collapsed action
decision. Widening the adapter doesn't fix that; it still can't see what it needs to see.

The deeper issue: `morphology_obs` was already part of the trunk's own input the whole time
(`actor_in_keys` includes it), but Stage 1 only ever trained on a single body shape, so the
trunk's weights never had a reason to route information from that input into the policy.
Freezing the trunk for all of Stage 2 permanently locks in "ignores body shape" at the one place
in the network with the capacity to use it well. No adapter design bolted on afterward can fix
that from the outside.

This experiment unfreezes the trunk (`norm`/`mlp`) so the whole network -- trunk + beta_encoder
+ fusion_mlp -- fine-tunes jointly, warm-started from v4's own checkpoint (not from Stage 1
directly) to keep the GPU-hours already spent: the fusion head's partially-learned correction
carries over as a warm start, and the trunk now gets the chance to absorb shape-conditioning
into its own weights instead of leaving all of it to a downstream patch.

Two things change together with `freeze_backbone=False` (see `fusion_adapter_mlp.py`):
    1. `norm`/`mlp` get `requires_grad=True` instead of being frozen after the first forward.
    2. `norm`/`mlp` are no longer pinned to `.eval()` during training, so the obs-normalizer's
       running stats (calibrated on Stage 1's single body shape) can adapt to the full
       morphology distribution instead of staying stale.
The actor optimizer already holds every actor parameter (`list(model._actor.parameters())` at
construction, not filtered by `requires_grad`), so no optimizer-construction change is needed --
flipping `requires_grad` is sufficient for previously-frozen params to start receiving updates.

**Watch closely at the start**: `actor_optimizer.lr` is left at v2/v3/v4's 4e-6, which was
deliberately small because it used to govern only the tiny adapter. It now governs the entire
6x2896 trunk too. Watch `actor/clip_frac` and `actor/grad_norm_before_clip` over the first
~1000-2000 epochs -- if updates are barely moving the policy, the LR may need raising; if
`eval/success_rate` or `eval/normalized_jerk_mean` regress sharply from v4's plateau, the LR is
too aggressive and is overwriting the already-converged motion-tracking skill (the exact failure
mode freezing was originally meant to prevent).

Requires --checkpoint pointing at v4's own latest checkpoint (e.g.
results/hhi_wide_fusion_stage2_clippool/last.ckpt on the training pod), NOT the Stage 1
last_morph_reset.ckpt -- we want to keep the fusion head's already-learned correction, not
restart it from zero-init. Loads via allow_partial_checkpoint_load=True (harmless here since
keys should match exactly, kept for consistency with v4's launch).

    python protomotions/train_agent.py \\
        --robot-name smpl_mor --simulator isaacgym \\
        --experiment-path examples/experiments/mimic/mlp_wide_fusion_stage2_unfrozen.py \\
        --experiment-name hhi_wide_fusion_stage2_unfrozen \\
        --checkpoint results/hhi_wide_fusion_stage2_clippool/last.ckpt \\
        --global-clip-pool-source r2:proto-data/hhi_stage2_per_clip/ \\
        --global-clip-pool-cache-dir /workspace/motion_cache \\
        --num-envs 6144 --batch-size 24576 --ngpu 6

Verification checklist before trusting a new run's numbers:
    1. CPU dummy-forward: ALL params (norm, mlp, beta_encoder, fusion_mlp) get nonzero grad --
       opposite of v4's check, where norm/mlp were expected to get zero grad.
    2. Load v4's checkpoint with strict=True (keys should match exactly, no missing/unexpected).
    3. `norm`'s running mean/var should change after a few training steps with varied
       `morphology_obs` in the batch (confirms the eval()-pin removal is actually in effect).
"""
from examples.experiments.mimic.mlp_wide_fusion_stage2 import (
    additional_experiment_arguments,
    terrain_config,
    scene_lib_config,
    motion_lib_config,
    env_config,
    configure_robot_and_simulator,
    apply_inference_overrides,
    WIDE_UNITS,
    BETA_ENCODER_UNITS,
    FUSION_UNITS,
    FUSION_NUM_LAYERS,
)
from protomotions.robot_configs.base import RobotConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.agents.ppo.config import PPOAgentConfig
import argparse

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
            freeze_backbone=False,
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
            # Left at v2/v3/v4's 4e-6 -- deliberately conservative starting point now that this
            # LR governs the whole 6x2896 trunk, not just the small adapter. See module
            # docstring's "watch closely at the start" note.
            actor_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=4e-6),
            critic_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=1e-4),
        ),
        batch_size=args.batch_size,
        training_max_steps=args.training_max_steps,
        gradient_clip_val=50.0,
        clip_critic_loss=True,
        # v4's checkpoint already has every key this config expects (norm/mlp/beta_encoder/
        # fusion_mlp) -- unlike v4's own warm-start from bare Stage 1. Kept True anyway for
        # robustness against a Stage-1 checkpoint being passed in by mistake.
        allow_partial_checkpoint_load=True,
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
