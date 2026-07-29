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
Stage 2 v6 — Deepen the Unfrozen Trunk (Net2Net Identity Insertion)
===========================================================================

Same as mlp_wide_fusion_stage2_unfrozen.py (v5) in every respect except one: the actor trunk
has 7 hidden layers instead of 6.

Why: v5 (`hhi_wide_fusion_stage2_unfrozen`, wandb `07c4zjgs`) genuinely plateaued -- fast climb
right after unfreezing (52%->77% success_rate in ~1,500 epochs), then flat 76-82% for the next
~5,100 epochs. Every Stage 2 architecture tried so far (frozen fusion, unfrozen fusion, v2's
full-concat adapter, v3's shape-only adapter) clusters around this same ~78-82% band despite very
different information flow, while the *single-shape* 6-layer trunk (hhi_wide_20946_neutral)
reaches 95-97%. That's ambiguous: either 6 layers has enough depth to track motion but not enough
to also absorb 128-shape conditioning, or the ceiling is task-difficulty (the same
crawl/kneel/squat/single-leg-balance classes that were already the dominant failure mode even in
single-shape training), not depth. This experiment tests depth directly and cheaply, continuing
from v5's own plateaued checkpoint instead of paying for a full restart.

Mechanism: `tools/deepen_fusion_trunk.py` inserts a new `Linear(2896,2896)` into the trunk,
initialized to the identity matrix with zero bias, right after the last existing ReLU (before the
final num_actions projection). Since that ReLU's output is already >= 0, ReLU(Identity(x)) == x
exactly -- the network computes bit-for-bit the same thing the moment the layer is inserted, so
none of v5's ~80%-plateau progress is lost; the new layer is simply free to start learning a
nontrivial transformation from that point on. See that script's module docstring for the full
mechanism and why `actor_optimizer` state is left stale rather than deleted -- it fails to load
with a `ValueError` (param-group size mismatch) once the trunk has an extra layer's worth of
params, which `allow_partial_checkpoint_load=True` below catches and skips, same as v5's own
warm-start from v4.

Deliberately +1 layer, not +2: cheaper, isolates one variable. If `eval/success_rate` doesn't
move within the same ~1-2k epoch window the unfreeze itself took to resolve, that's evidence the
ceiling isn't depth-related -- add a second layer the same way, or abandon this line for the
from-scratch/no-adapter alternative, rather than adding more layers blindly.

`actor_optimizer` lr stays at v5's 4e-6, unchanged -- Net2Net's own justification for identity
init is that no LR bump is needed, since the loss landscape at the insertion point is smooth
(output identical to before insertion, so the immediate gradient signal into the new layer starts
small and grows as training proceeds).

Requires --checkpoint pointing at the *deepened* checkpoint produced by
tools/deepen_fusion_trunk.py, NOT v5's raw last.ckpt directly (that has only 6 mlp layers --
loading it here would leave the 7th layer's LazyLinear un-materialized-to-identity, i.e. randomly
initialized, which defeats the whole point):

    python tools/deepen_fusion_trunk.py \\
        --checkpoint results/hhi_wide_fusion_stage2_unfrozen/last.ckpt \\
        --output results/hhi_wide_fusion_stage2_unfrozen/last_deepened1.ckpt \\
        --num-new-layers 1 --verify

    python protomotions/train_agent.py \\
        --robot-name smpl_mor --simulator isaacgym \\
        --experiment-path examples/experiments/mimic/mlp_wide_fusion_stage2_deepened.py \\
        --experiment-name hhi_wide_fusion_stage2_deepened1 \\
        --checkpoint results/hhi_wide_fusion_stage2_unfrozen/last_deepened1.ckpt \\
        --global-clip-pool-source r2:proto-data/hhi_stage2_per_clip/ \\
        --global-clip-pool-cache-dir /workspace/motion_cache \\
        --num-envs 6144 --batch-size 24576 --ngpu 6

Verification checklist before trusting a new run's numbers:
    1. tools/deepen_fusion_trunk.py --verify: standalone old-vs-new forward pass matches within
       1e-5, before ever touching the real training stack.
    2. CPU dummy-forward through the full FusionAdapterMLPWithConcat actor: output at t=0 matches
       v5's pre-surgery checkpoint on the same input; the new layer's params get nonzero grad
       after backward (confirms it isn't stuck at identity).
    3. First eval point after launch should land close to v5's plateau (~79-80% success, not a
       sharp regression) -- that's the correctness gate, separate from whether depth actually
       helps.
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

NUM_NEW_LAYERS = 1  # must match tools/deepen_fusion_trunk.py's --num-new-layers


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
            layers=[
                MLPLayerConfig(units=WIDE_UNITS, activation="relu")
                for _ in range(6 + NUM_NEW_LAYERS)
            ],
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
            # Unchanged from v5 -- identity-init means no LR bump is needed. See module docstring.
            actor_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=4e-6),
            critic_optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=1e-4),
        ),
        batch_size=args.batch_size,
        training_max_steps=args.training_max_steps,
        gradient_clip_val=50.0,
        clip_critic_loss=True,
        # Required, not just a safety net this time: tools/deepen_fusion_trunk.py writes mlp
        # keys matching this config's layout exactly, but leaves actor_optimizer stale (see that
        # script's docstring) -- its ValueError on load is only caught when this is True.
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
