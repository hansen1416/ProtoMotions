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
"""Frozen-backbone + concat-fusion adapter MLP, conditioned on body morphology.

Design doc: note/README.note.md #32, #35 ("v4 candidate"). Stage 2 (shape-transfer) v4: unlike
`residual_adapter_mlp.py`'s `ResidualAdapterMLPWithConcat` (v3), whose adapter only ever sees
`morphology_obs`, this variant lets the correction depend on what the frozen trunk actually
computed for the current pose -- trading v3's "provably jerk-free by construction" guarantee
for the ability to learn pose-shape interaction corrections (e.g. a different ankle correction
while crouching vs. standing), which v3 cannot represent (it can only learn a per-body constant
offset).

    beta_embed = beta_encoder(morphology_obs)                  # small MLP, betas ONLY
    trunk_out  = frozen_trunk(obs)                              # unchanged Stage 1 weights
    delta      = fusion_mlp(concat(trunk_out, beta_embed))      # small trainable MLP, TRAINABLE
    output     = trunk_out + delta

`fusion_mlp`'s last layer is zero-initialized (same convention as v3), so `delta == 0` from the
first real forward pass onward -- Stage 2 still starts as an exact continuation of the Stage 1
policy, not a fresh regression, even though the fusion head's *input* includes the trunk's
output.

Known tradeoff (deliberately accepted, see module docstring above and #35's "v4 candidate"
note): `trunk_out` changes every timestep (unlike `morphology_obs`, which is constant per
episode), so `delta` is no longer constant within an episode and can in principle reintroduce
the frame-to-frame jerk that motivated v3's betas-only restriction. Pair with a stronger
`action_smoothness` reward weight (see experiment file) rather than relying on architecture
alone to control jerk.

Key Classes:
    FusionAdapterMLPWithConcatConfig — configuration dataclass
    FusionAdapterMLPWithConcat       — TensorDictModuleBase implementation
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
class FusionAdapterMLPWithConcatConfig(MLPWithConcatConfig):
    """Configuration for the frozen-backbone + concat-fusion adapter MLP.

    Drop-in replacement for MLPWithConcatConfig as an actor trunk (`mu_model`). All
    `MLPWithConcatConfig` fields (layers, in_keys, out_keys, normalize_obs, ...) define the
    frozen backbone, identical to the Stage 1 config it's meant to warm-start from.
    """

    _target_: str = "protomotions.agents.common.fusion_adapter_mlp.FusionAdapterMLPWithConcat"

    adapter_in_keys: List[str] = field(
        default_factory=lambda: ["morphology_obs"],
        metadata={
            "help": (
                "Keys the beta encoder reads (concatenated), separate from the frozen "
                "trunk's in_keys. Defaults to body-shape only -- see module docstring."
            )
        },
    )

    beta_encoder_layers: List[MLPLayerConfig] = field(
        default_factory=lambda: [MLPLayerConfig(units=128, activation="relu")],
        metadata={"help": "Layers of the small MLP that embeds morphology_obs (betas only)."},
    )

    fusion_layers: List[MLPLayerConfig] = field(
        default_factory=lambda: [
            MLPLayerConfig(units=512, activation="relu"),
            MLPLayerConfig(units=512, activation="relu"),
        ],
        metadata={
            "help": (
                "Hidden layers of the fusion head, which reads concat(trunk_out, beta_embed). "
                "A final zero-initialized LazyLinear(num_out) is appended automatically."
            )
        },
    )

    def __post_init__(self):
        super().__post_init__()
        assert self.adapter_in_keys, "FusionAdapterMLPWithConcatConfig: adapter_in_keys must be non-empty"
        assert self.beta_encoder_layers, "FusionAdapterMLPWithConcatConfig: beta_encoder_layers must be non-empty"
        assert self.fusion_layers, "FusionAdapterMLPWithConcatConfig: fusion_layers must be non-empty"


class FusionAdapterMLPWithConcat(MLPWithConcat):
    """Frozen MLPWithConcat trunk + a small trainable concat-fusion head.

    Subclasses MLPWithConcat (not composition) so `norm.*`/`mlp.*` state_dict keys are
    identical to a plain Stage 1 checkpoint's -- only `beta_encoder.*`/`fusion_mlp.*` are new.

    Freezing the base and zero-initializing the fusion head's last layer both happen lazily,
    right after the first forward pass (when LazyLinear layers materialize into real
    `nn.Linear` params) -- see `_finalize_after_first_forward`, same convention as
    `ResidualAdapterMLPWithConcat`.
    """

    config: FusionAdapterMLPWithConcatConfig

    def __init__(self, config: FusionAdapterMLPWithConcatConfig):
        super().__init__(config)

        self._adapter_finalized = False

        beta_encoder_modules = []
        for layer in config.beta_encoder_layers:
            beta_encoder_modules.append(nn.LazyLinear(layer.units))
            beta_encoder_modules.append(get_activation_func(layer.activation))
        self.beta_encoder = nn.Sequential(*beta_encoder_modules)

        fusion_modules = []
        for layer in config.fusion_layers:
            fusion_modules.append(nn.LazyLinear(layer.units))
            fusion_modules.append(get_activation_func(layer.activation))
        fusion_modules.append(nn.LazyLinear(config.num_out))
        self.fusion_mlp = nn.Sequential(*fusion_modules)

    def _beta_input(self, tensordict: TensorDict):
        """Concatenate the beta encoder's own input keys (default: morphology_obs only)."""
        return torch.cat([tensordict[key] for key in self.config.adapter_in_keys], dim=-1)

    def _finalize_after_first_forward(self):
        """Freeze everything except the adapter (beta_encoder + fusion_mlp), and zero-init
        the fusion head's last layer.

        Must run after the first forward pass, once LazyLinear layers have materialized into
        real nn.Linear params (UninitializedParameter can't be frozen/zero-initialized).
        """
        adapter_param_ids = {id(p) for p in self.beta_encoder.parameters()} | {
            id(p) for p in self.fusion_mlp.parameters()
        }
        for p in self.parameters():
            if id(p) not in adapter_param_ids:
                p.requires_grad = False

        last_linear = self.fusion_mlp[-1]
        nn.init.zeros_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)

        self._adapter_finalized = True

    def train(self, mode: bool = True):
        """Keep the frozen trunk's obs-normalizer pinned to eval regardless of the outer
        actor's train()/eval() calls. Same rationale as `ResidualAdapterMLPWithConcat.train`.
        """
        super().train(mode)
        if self._adapter_finalized:
            self.norm.eval()
            self.mlp.eval()
        return self

    def forward(self, tensordict: TensorDict) -> TensorDict:
        tensordict = super().forward(tensordict)
        trunk_out = tensordict[self.config.out_keys[0]]

        beta_embed = self.beta_encoder(self._beta_input(tensordict))
        fusion_input = torch.cat([trunk_out, beta_embed], dim=-1)
        delta = self.fusion_mlp(fusion_input)

        if not self._adapter_finalized:
            self._finalize_after_first_forward()

        tensordict[self.config.out_keys[0]] = trunk_out + delta
        return tensordict
