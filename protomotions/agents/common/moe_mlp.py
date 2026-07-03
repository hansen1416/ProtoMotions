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
"""GMT-style Mixture-of-Experts MLP — motion axis only.

K parallel expert trunks + a gate, blended as a convex combination:

    a = sum_i p_i * a_i(x)

Gate is either learned (softmax over a small gate net, the GMT default) or hard-routed
from a precomputed per-env motion->expert assignment already present in the TensorDict
(no gate net involved, one expert per sample, no collapse risk). Both modes write
`gate_probs`/`expert_selection` so a load-balancing loss can be computed elsewhere
(see PPO.calculate_extra_actor_loss) and so routing can be inspected post-hoc.

Body-shape conditioning (HyperDistill) is layered on separately once this module is
validated standalone — see note/README.note.md #18. This module treats morphology_obs
(if present in in_keys) as a plain concatenated input, same as the flat-concat baseline.

Key Classes:
    MoEMLPConfig — configuration dataclass
    MoEMLP       — TensorDictModuleBase implementation
"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Optional
from dataclasses import dataclass, field
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase
from protomotions.agents.common.common import NormObsBase
from protomotions.agents.common.config import NormObsBaseConfig, MLPLayerConfig
from protomotions.agents.utils.training import get_activation_func


@dataclass
class MoEMLPConfig(NormObsBaseConfig):
    """Configuration for the GMT-style Mixture-of-Experts MLP.

    Drop-in replacement for MLPWithConcatConfig as an actor/critic trunk. `expert_layers`
    defines one expert's hidden layers; `num_experts` copies of that stack are built,
    each independently weighted, and blended via the gate.
    """

    _target_: str = "protomotions.agents.common.moe_mlp.MoEMLP"

    in_keys: List[str] = field(
        default_factory=list,
        metadata={"help": "Input tensor keys to read and concatenate from TensorDict."},
    )
    out_keys: List[str] = field(
        default_factory=list,
        metadata={"help": "Output tensor key (exactly one)."},
    )
    num_out: int = field(
        default=None,
        metadata={"help": "Output dimension of each expert / the blended output."},
    )
    num_experts: int = field(
        default=2,
        metadata={"help": "Number of parallel expert trunks (K)."},
    )
    expert_layers: List[MLPLayerConfig] = field(
        default_factory=list,
        metadata={"help": "Hidden layer configs for one expert. Same shape used for all K experts."},
    )
    gate_hidden_units: List[int] = field(
        default_factory=lambda: [256, 256],
        metadata={"help": "Hidden unit counts for the gate MLP (ignored when gate_mode='hard')."},
    )
    gate_activation: str = field(
        default="silu",
        metadata={"help": "Activation function used inside the gate MLP."},
    )
    gate_mode: str = field(
        default="learned",
        metadata={"help": "'learned' = softmax gate net. 'hard' = lookup from hard_routing_key.",
                  "options": ["learned", "hard"]},
    )
    hard_routing_key: Optional[str] = field(
        default=None,
        metadata={"help": "TensorDict key holding a per-env int64 expert index. Required when gate_mode='hard'."},
    )
    gate_probs_key: str = field(
        default="gate_probs",
        metadata={"help": "TensorDict key to write the [B, num_experts] gate distribution to."},
    )
    expert_selection_key: str = field(
        default="expert_selection",
        metadata={"help": "TensorDict key to write the argmax expert index [B] to."},
    )
    output_activation: Optional[str] = field(
        default=None,
        metadata={"help": "Optional activation applied to the blended output."},
    )

    def __post_init__(self):
        assert self.num_out is not None, "MoEMLPConfig: num_out must be provided"
        assert self.expert_layers, "MoEMLPConfig: expert_layers must be provided"
        assert len(self.out_keys) == 1, "MoEMLPConfig requires exactly one out_key"
        assert self.num_experts >= 1, "MoEMLPConfig: num_experts must be >= 1"
        assert self.gate_mode in ("learned", "hard"), f"Unknown gate_mode: {self.gate_mode}"
        if self.gate_mode == "hard":
            assert self.hard_routing_key is not None, (
                "MoEMLPConfig: hard_routing_key is required when gate_mode='hard'"
            )


def _build_expert(layers: List[MLPLayerConfig], num_out: int) -> nn.Sequential:
    """Build one expert trunk: LazyLinear+activation per hidden layer, then a LazyLinear head."""
    blocks = []
    for layer_cfg in layers:
        blocks.append(nn.LazyLinear(layer_cfg.units))
        blocks.append(get_activation_func(layer_cfg.activation))
    blocks.append(nn.LazyLinear(num_out))
    return nn.Sequential(*blocks)


class MoEMLP(TensorDictModuleBase):
    """GMT-style Mixture-of-Experts MLP, usable as actor trunk or critic.

    Architecture
    ------------
        cat(in_keys) -> [normalize] -> x
        expert_i(x) for i in 1..K            -> stacked [B, K, num_out]
        gate(x) (or hard lookup)             -> gate_probs [B, K]
        output = sum_i gate_probs_i * expert_i(x)

    All K experts are evaluated for every sample, even in 'hard' gate_mode (the blend
    collapses to a one-hot selection). Skipping the un-selected experts' compute in
    hard mode is a possible follow-up optimization, not required for correctness.
    """

    config: MoEMLPConfig

    def __init__(self, config: MoEMLPConfig):
        TensorDictModuleBase.__init__(self)
        self.config = config

        self.norm = NormObsBase(config)

        self.experts = nn.ModuleList(
            [_build_expert(config.expert_layers, config.num_out) for _ in range(config.num_experts)]
        )

        self.gate = None
        if config.gate_mode == "learned":
            gate_blocks = []
            for units in config.gate_hidden_units:
                gate_blocks.append(nn.LazyLinear(units))
                gate_blocks.append(get_activation_func(config.gate_activation))
            gate_blocks.append(nn.LazyLinear(config.num_experts))
            self.gate = nn.Sequential(*gate_blocks)

        self.output_activation = None
        if config.output_activation is not None:
            self.output_activation = get_activation_func(config.output_activation)

        self.in_keys = list(config.in_keys)
        if config.gate_mode == "hard":
            self.in_keys = self.in_keys + [config.hard_routing_key]
        self.out_keys = config.out_keys

    def forward(self, tensordict: TensorDict) -> TensorDict:
        x = torch.cat([tensordict[key] for key in self.config.in_keys], dim=-1)
        x = self.norm(x)

        # [B, K, num_out]
        expert_outs = torch.stack([expert(x) for expert in self.experts], dim=1)

        if self.config.gate_mode == "hard":
            expert_id = tensordict[self.config.hard_routing_key].long().view(-1)
            gate_probs = F.one_hot(expert_id, num_classes=self.config.num_experts).to(x.dtype)
        else:
            gate_logits = self.gate(x)
            gate_probs = torch.softmax(gate_logits, dim=-1)

        blended = torch.einsum("bk,bko->bo", gate_probs, expert_outs)

        if self.output_activation is not None:
            blended = self.output_activation(blended)

        tensordict[self.config.out_keys[0]] = blended
        tensordict[self.config.gate_probs_key] = gate_probs
        tensordict[self.config.expert_selection_key] = gate_probs.argmax(dim=-1)

        return tensordict
