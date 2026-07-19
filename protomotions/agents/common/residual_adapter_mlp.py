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
"""Frozen-backbone + zero-init residual-adapter MLP, conditioned on body morphology.

Design doc: note/README.note.md #32, #35. Built for Stage 2 (shape-transfer) fine-tuning: the
trunk is a frozen, unmodified copy of a Stage 1 `MLPWithConcat` (e.g. `mlp_wide.py`'s 6x2896
trunk), and a small trainable MLP adds a per-env residual correction:

    base_out = frozen_trunk(obs)               # unchanged Stage 1 weights
    delta    = adapter_mlp(adapter_in_keys)     # small trainable MLP, TRAINABLE
    output   = base_out + delta

`adapter_mlp` reads only `config.adapter_in_keys` (default: `["morphology_obs"]`, the 11-dim
`[gender_id, betas/3.0]`), **not** the trunk's full pose/motion-target/previous-action input.
`morphology_obs` is constant for the whole episode (only changes on env reset to a new body),
so restricting the adapter to it forces `delta` to be constant within an episode too -- it
cannot introduce frame-to-frame jerk, unlike an earlier variant that fed it the full
1014-dim concatenated obs (diagnosed 2026-07-19: adapter output was ~40% of the trunk's
magnitude and tracked the same high-frequency pose signal the trunk did). `adapter_in_keys`
is configurable if a future experiment wants to add pose-context back in.

`adapter_mlp`'s last layer is zero-initialized, so `delta == 0` from the first real forward
pass onward (the network is materialized once during the harness's own dummy warm-up forward
in `BaseAgent.setup()`, before any real rollout/inference happens) -- Stage 2 starts as an
exact continuation of the Stage 1 policy, not a fresh regression.

`ResidualAdapterMLPWithConcat` subclasses `MLPWithConcat` directly (rather than wrapping it)
so `norm.*`/`mlp.*` parameter names are unchanged from the Stage 1 checkpoint -- no key
remapping needed, just `strict=False` loading (`allow_partial_checkpoint_load` on the agent
config) to tolerate the new `adapter_mlp.*` keys that don't exist in that checkpoint.

Key Classes:
    ResidualAdapterMLPWithConcatConfig — configuration dataclass
    ResidualAdapterMLPWithConcat       — TensorDictModuleBase implementation
"""

import torch
from torch import nn
from typing import List
from dataclasses import dataclass, field
from tensordict import TensorDict

from protomotions.agents.common.mlp import MLPWithConcat
from protomotions.agents.common.config import MLPWithConcatConfig, MLPLayerConfig
from protomotions.agents.utils.training import get_activation_func


@dataclass
class ResidualAdapterMLPWithConcatConfig(MLPWithConcatConfig):
    """Configuration for the frozen-backbone + residual-adapter MLP.

    Drop-in replacement for MLPWithConcatConfig as an actor trunk (`mu_model`). All
    `MLPWithConcatConfig` fields (layers, in_keys, out_keys, normalize_obs, ...) define the
    frozen backbone, identical to the Stage 1 config it's meant to warm-start from.
    """

    _target_: str = "protomotions.agents.common.residual_adapter_mlp.ResidualAdapterMLPWithConcat"

    adapter_layers: List[MLPLayerConfig] = field(
        default_factory=lambda: [
            MLPLayerConfig(units=512, activation="relu"),
            MLPLayerConfig(units=512, activation="relu"),
        ],
        metadata={"help": "Hidden layers of the small trainable residual-adapter MLP."},
    )

    adapter_in_keys: List[str] = field(
        default_factory=lambda: ["morphology_obs"],
        metadata={
            "help": (
                "Keys the adapter reads (concatenated), separate from the frozen trunk's "
                "in_keys. Defaults to body-shape only so delta stays constant within an "
                "episode -- see module docstring."
            )
        },
    )

    def __post_init__(self):
        super().__post_init__()
        assert self.adapter_layers, "ResidualAdapterMLPWithConcatConfig: adapter_layers must be non-empty"
        assert self.adapter_in_keys, "ResidualAdapterMLPWithConcatConfig: adapter_in_keys must be non-empty"


class ResidualAdapterMLPWithConcat(MLPWithConcat):
    """Frozen MLPWithConcat trunk + a small trainable, morphology-aware residual MLP.

    Subclasses MLPWithConcat (not composition) so `norm.*`/`mlp.*` state_dict keys are
    identical to a plain Stage 1 checkpoint's -- only `adapter_mlp.*` is new.

    Freezing the base and zero-initializing the adapter's last layer both happen lazily,
    right after the first forward pass (when LazyLinear layers materialize into real
    `nn.Linear` params) -- see `_finalize_after_first_forward`. This mirrors the repo-wide
    convention (`BaseAgent.setup()`) of a dummy forward pass to materialize lazy modules
    before any real use, so no changes to the training loop's setup/load ordering are needed.
    """

    config: ResidualAdapterMLPWithConcatConfig

    def __init__(self, config: ResidualAdapterMLPWithConcatConfig):
        super().__init__(config)

        self._adapter_finalized = False

        adapter_modules = []
        for layer in config.adapter_layers:
            adapter_modules.append(nn.LazyLinear(layer.units))
            adapter_modules.append(get_activation_func(layer.activation))
        adapter_modules.append(nn.LazyLinear(config.num_out))
        self.adapter_mlp = nn.Sequential(*adapter_modules)

    def _adapter_input(self, tensordict: TensorDict):
        """Concatenate the adapter's own input keys (default: morphology_obs only).

        Deliberately independent of the trunk's `in_keys`/normalization -- see module
        docstring for why the adapter is scoped to body-shape by default.
        """
        return torch.cat([tensordict[key] for key in self.config.adapter_in_keys], dim=-1)

    def _finalize_after_first_forward(self):
        """Freeze everything except the adapter, and zero-init the adapter's last layer.

        Must run after the first forward pass, once LazyLinear layers have materialized into
        real nn.Linear params (UninitializedParameter can't be frozen/zero-initialized).
        """
        adapter_param_ids = {id(p) for p in self.adapter_mlp.parameters()}
        for p in self.parameters():
            if id(p) not in adapter_param_ids:
                p.requires_grad = False

        last_linear = self.adapter_mlp[-1]
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
        delta = self.adapter_mlp(adapter_input)

        if not self._adapter_finalized:
            self._finalize_after_first_forward()

        tensordict[self.config.out_keys[0]] = base_out + delta
        return tensordict
