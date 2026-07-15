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
"""Frozen-backbone + zero-init LoRA-style residual adapter, conditioned on body morphology.

Design doc: note/README.note.md #32. Built for Stage 2 (shape-transfer) fine-tuning: the
trunk is a frozen, unmodified copy of a Stage 1 `MLPWithConcat` (e.g. `mlp_wide.py`'s 6x2896
trunk), and a small trainable adapter injects a per-env residual conditioned on
`morphology_obs`:

    base_out   = frozen_trunk(obs)                  # unchanged Stage 1 weights
    bottleneck = adapter_down(obs)                   # shared LazyLinear, in -> r, TRAINABLE
    W_up_e     = hypernet(morphology_obs)            # per-env (num_out -> r), TRAINABLE
    delta      = einsum(bottleneck, W_up_e) * scale
    output     = base_out + delta

`hypernet`'s last layer is zero-initialized, so `delta == 0` from the first real forward
pass onward (the network is materialized once during the harness's own dummy warm-up forward
in `BaseAgent.setup()`, before any real rollout/inference happens) — Stage 2 starts as an
exact continuation of the Stage 1 policy, not a fresh regression.

`LoRAResidualMLPWithConcat` subclasses `MLPWithConcat` directly (rather than wrapping it) so
`norm.*`/`mlp.*` parameter names are unchanged from the Stage 1 checkpoint — no key remapping
needed, just `strict=False` loading (`allow_partial_checkpoint_load` on the agent config) to
tolerate the new `adapter_down.*`/`hypernet.*` keys that don't exist in that checkpoint.

Key Classes:
    LoRAResidualMLPWithConcatConfig — configuration dataclass
    LoRAResidualMLPWithConcat       — TensorDictModuleBase implementation
"""

import torch
from torch import nn
from typing import Optional
from dataclasses import dataclass, field
from tensordict import TensorDict

from protomotions.agents.common.mlp import MLPWithConcat
from protomotions.agents.common.config import MLPWithConcatConfig
from protomotions.agents.utils.training import get_activation_func


@dataclass
class LoRAResidualMLPWithConcatConfig(MLPWithConcatConfig):
    """Configuration for the frozen-backbone + LoRA-style residual adapter MLP.

    Drop-in replacement for MLPWithConcatConfig as an actor trunk (`mu_model`). All
    `MLPWithConcatConfig` fields (layers, in_keys, out_keys, normalize_obs, ...) define the
    frozen backbone, identical to the Stage 1 config it's meant to warm-start from.
    """

    _target_: str = "protomotions.agents.common.lora_residual_mlp.LoRAResidualMLPWithConcat"

    adapter_rank: int = field(
        default=16,
        metadata={"help": "Rank r of the low-rank residual delta.", "min": 1},
    )
    lora_alpha: Optional[int] = field(
        default=None,
        metadata={
            "help": "LoRA scale numerator; delta is scaled by lora_alpha/adapter_rank. "
            "Defaults to adapter_rank (scale=1.0) if not set."
        },
    )
    hyper_hidden_units: int = field(
        default=64,
        metadata={"help": "Hidden width of the small hypernetwork that generates the per-env up-projection.", "min": 1},
    )
    hyper_activation: str = field(
        default="silu",
        metadata={"help": "Activation function used inside the hypernetwork."},
    )
    morphology_key: str = field(
        default="morphology_obs",
        metadata={"help": "TensorDict key holding the per-env morphology/body-shape observation."},
    )

    def __post_init__(self):
        super().__post_init__()
        assert self.adapter_rank >= 1, "LoRAResidualMLPWithConcatConfig: adapter_rank must be >= 1"
        if self.lora_alpha is None:
            self.lora_alpha = self.adapter_rank


class LoRAResidualMLPWithConcat(MLPWithConcat):
    """Frozen MLPWithConcat trunk + a small trainable, morphology-conditioned residual adapter.

    Subclasses MLPWithConcat (not composition) so `norm.*`/`mlp.*` state_dict keys are
    identical to a plain Stage 1 checkpoint's — only `adapter_down.*`/`hypernet.*` are new.

    Freezing the base and zero-initializing the hypernetwork's last layer both happen lazily,
    right after the first forward pass (when LazyLinear layers materialize into real
    `nn.Linear` params) — see `_finalize_after_first_forward`. This mirrors the repo-wide
    convention (`BaseAgent.setup()`) of a dummy forward pass to materialize lazy modules
    before any real use, so no changes to the training loop's setup/load ordering are needed.
    """

    config: LoRAResidualMLPWithConcatConfig

    def __init__(self, config: LoRAResidualMLPWithConcatConfig):
        super().__init__(config)

        self._lora_scale = config.lora_alpha / config.adapter_rank
        self._adapter_finalized = False

        self.adapter_down = nn.LazyLinear(config.adapter_rank)
        self.hypernet = nn.Sequential(
            nn.LazyLinear(config.hyper_hidden_units),
            get_activation_func(config.hyper_activation),
            nn.LazyLinear(config.num_out * config.adapter_rank),
        )

    def _adapter_input(self, tensordict: TensorDict) -> torch.Tensor:
        """Reuse the frozen trunk's normalized concatenated obs if available, else recompute."""
        norm_key = f"norm_{self.config.in_keys[0]}"
        if self.config.normalize_obs and norm_key in tensordict.keys():
            return tensordict[norm_key]
        return torch.cat([tensordict[key] for key in self.config.in_keys], dim=-1)

    def _finalize_after_first_forward(self):
        """Freeze everything except the adapter, and zero-init the hypernetwork's last layer.

        Must run after the first forward pass, once LazyLinear layers have materialized into
        real nn.Linear params (UninitializedParameter can't be frozen/zero-initialized).
        """
        adapter_param_ids = {id(p) for p in self.adapter_down.parameters()}
        adapter_param_ids |= {id(p) for p in self.hypernet.parameters()}
        for p in self.parameters():
            if id(p) not in adapter_param_ids:
                p.requires_grad = False

        last_linear = self.hypernet[-1]
        nn.init.zeros_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)

        self._adapter_finalized = True

    def train(self, mode: bool = True):
        """Keep the frozen trunk's obs-normalizer pinned to eval regardless of the outer
        actor's train()/eval() calls.

        `requires_grad=False` only stops gradient updates to `norm`/`mlp`'s *parameters* --
        it does nothing to `RunningMeanStd`'s running mean/var buffers, which
        `NormObsBase.forward` keeps updating purely based on `self.training` (see
        `agents/common/common.py`). `BaseAgent.train()` calls `self.model.train()` before
        every optimization step, which by default recurses into every submodule -- so
        without this override the "frozen" trunk's input normalization would keep drifting
        under its frozen weights throughout Stage 2, silently undermining the point of
        freezing it. Matches the freeze convention used for `expert_model` in
        `agents/masked_mimic/agent.py` (`requires_grad=False` + pinned `.eval()`).
        """
        super().train(mode)
        if self._adapter_finalized:
            self.norm.eval()
            self.mlp.eval()
        return self

    def forward(self, tensordict: TensorDict) -> TensorDict:
        tensordict = super().forward(tensordict)
        base_out = tensordict[self.config.out_keys[0]]

        adapter_input = self._adapter_input(tensordict)
        bottleneck = self.adapter_down(adapter_input)  # [B, r]

        morphology = tensordict[self.config.morphology_key]
        hyper_out = self.hypernet(morphology)  # [B, num_out * r]

        if not self._adapter_finalized:
            self._finalize_after_first_forward()

        w_up = hyper_out.view(-1, self.config.num_out, self.config.adapter_rank)  # [B, num_out, r]
        delta = torch.einsum("br,bor->bo", bottleneck, w_up) * self._lora_scale

        tensordict[self.config.out_keys[0]] = base_out + delta
        return tensordict
