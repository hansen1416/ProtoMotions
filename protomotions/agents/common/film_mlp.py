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
"""FiLM-conditioned MLP for morphology-aware policy and value networks.

FiLM (Feature-wise Linear Modulation) separates the morphology signal
(gender + shape betas) from the main state/task observations. Instead of
concatenating everything into one flat vector, the morphology code is routed
through a small conditioner network whose output modulates the main trunk's
hidden activations layer-by-layer:

    h_l = h_l * gamma_l(morphology) + beta_l(morphology)

This keeps the trunk motion/task-centric while body shape influences
representations multiplicatively at every hidden layer.

Proved effective in the predecessor HHI project; this module provides a
drop-in replacement for MLPWithConcat for direct comparison.

Key Classes:
    FiLMMLPConfig    — configuration dataclass
    FiLMMLPWithCond  — TensorDictModuleBase implementation
"""

import torch
from torch import nn
from typing import List, Optional
from dataclasses import dataclass, field
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase
from protomotions.agents.common.common import NormObsBase
from protomotions.agents.common.config import NormObsBaseConfig, MLPLayerConfig
from protomotions.agents.utils.training import get_activation_func


@dataclass
class FiLMMLPConfig(NormObsBaseConfig):
    """Configuration for FiLM-conditioned MLP.

    Drop-in replacement for MLPWithConcatConfig. The cond_keys (default:
    ["morphology_obs"]) are routed through a separate conditioner network
    whose output modulates the main trunk at each hidden layer via FiLM.

    The morphology tensor is expected to be [gender_id, beta_0, ..., beta_9]
    where gender_id is in {0, 1} and betas are in [-3, 3]. Betas are
    normalized by dividing by beta_norm_scale before being fed to the
    conditioner.
    """

    _target_: str = "protomotions.agents.common.film_mlp.FiLMMLPWithCond"

    in_keys: List[str] = field(
        default_factory=list,
        metadata={"help": "Main state/task observation keys (not morphology)."},
    )
    cond_keys: List[str] = field(
        default_factory=lambda: ["morphology_obs"],
        metadata={"help": "Conditioning keys — morphology [gender_id, betas]."},
    )
    out_keys: List[str] = field(
        default_factory=list,
        metadata={"help": "Output tensor key (exactly one)."},
    )
    num_out: int = field(
        default=None,
        metadata={"help": "Output dimension of the final linear head."},
    )
    layers: List[MLPLayerConfig] = field(
        default_factory=list,
        metadata={"help": "Hidden layer configs for the main trunk. One FiLM pair per layer."},
    )
    cond_hidden_units: List[int] = field(
        default_factory=lambda: [64, 64],
        metadata={"help": "Hidden unit counts for the morphology conditioner MLP."},
    )
    cond_activation: str = field(
        default="silu",
        metadata={"help": "Activation function used inside the conditioner MLP."},
    )
    beta_norm_scale: float = field(
        default=3.0,
        metadata={"help": "Divide beta dims by this value before conditioning (betas live in [-3, 3])."},
    )
    output_activation: Optional[str] = field(
        default=None,
        metadata={"help": "Optional activation applied to the final output."},
    )

    def __post_init__(self):
        assert self.num_out is not None, "FiLMMLPConfig: num_out must be provided"
        assert self.layers, "FiLMMLPConfig: layers must be provided"
        assert len(self.out_keys) == 1, "FiLMMLPConfig requires exactly one out_key"


class FiLMMLPWithCond(TensorDictModuleBase):
    """FiLM-conditioned MLP, usable as actor trunk, critic, or discriminator.

    Architecture
    ------------
    Main branch (state/task obs):
        cat(in_keys) → [normalize] → block_0 → FiLM_0 → block_1 → FiLM_1 → ... → head

    Conditioner branch (morphology):
        cat(cond_keys) → normalize_morphology → cond_mlp → cond_linear
            → split → [(gamma_0, beta_0), (gamma_1, beta_1), ...]

    FiLM at hidden layer l:
        h_l  = block_l(h_{l-1})
        h_l  = h_l * gamma_l + beta_l

    Supports both 2-D [B, D] and 3-D [T, N, D] tensors via ellipsis indexing
    in _split_film_params, making it usable for rollout-time AMP reward
    computation as well as mini-batch training.
    """

    config: FiLMMLPConfig

    def __init__(self, config: FiLMMLPConfig):
        TensorDictModuleBase.__init__(self)
        self.config = config

        # Running-mean/std normalizer for the main observations (lazy init).
        self.norm = NormObsBase(config)

        # Main trunk: one nn.Sequential block per hidden layer.
        # Kept as a ModuleList so forward can zip blocks with FiLM params.
        trunk_blocks = []
        for i, layer_cfg in enumerate(config.layers):
            block = [nn.LazyLinear(layer_cfg.units)]
            # Layer norm only supported on the first hidden layer (same convention as MLPWithConcat).
            if layer_cfg.use_layer_norm and i == 0:
                block.append(nn.LayerNorm(layer_cfg.units))
            block.append(get_activation_func(layer_cfg.activation))
            trunk_blocks.append(nn.Sequential(*block))
        self.trunk_blocks = nn.ModuleList(trunk_blocks)

        # Output head (no activation, no FiLM).
        self.head = nn.LazyLinear(config.num_out)

        # Morphology conditioner: maps morphology code → flat FiLM parameter vector.
        cond_layers = []
        for units in config.cond_hidden_units:
            cond_layers.append(nn.LazyLinear(units))
            cond_layers.append(get_activation_func(config.cond_activation))
        self.cond_mlp = nn.Sequential(*cond_layers)

        # film_out_size = 2 * sum(hidden units) — one gamma and one beta per unit per layer.
        film_out_size = sum(2 * layer_cfg.units for layer_cfg in config.layers)
        self.cond_linear = nn.LazyLinear(film_out_size)

        self.output_activation = None
        if config.output_activation is not None:
            self.output_activation = get_activation_func(config.output_activation)

        # TensorDict key declarations: union of main and conditioning keys.
        cond_only = [k for k in config.cond_keys if k not in config.in_keys]
        self.in_keys = config.in_keys + cond_only
        self.out_keys = config.out_keys

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _normalize_morphology(self, cond: torch.Tensor) -> torch.Tensor:
        """Normalize morphology: gender stays in {0,1}; betas divided by beta_norm_scale."""
        gender = cond[..., :1]
        betas = cond[..., 1:] / self.config.beta_norm_scale
        return torch.cat([gender, betas], dim=-1)

    def _split_film_params(self, cond_out: torch.Tensor):
        """Split flat conditioner output into per-layer (gamma, beta) pairs.

        Uses ellipsis indexing so it works for both [B, D] and [T, N, D].
        """
        params = []
        pos = 0
        for layer_cfg in self.config.layers:
            h = layer_cfg.units
            gamma = cond_out[..., pos:pos + h]
            beta = cond_out[..., pos + h:pos + 2 * h]
            params.append((gamma, beta))
            pos += 2 * h
        return params

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, tensordict: TensorDict) -> TensorDict:
        # --- Main branch: concatenate and optionally normalize main obs ---
        x = torch.cat([tensordict[key] for key in self.config.in_keys], dim=-1)
        x = self.norm(x)

        # --- Conditioning branch: normalize morphology and compute FiLM params ---
        cond = torch.cat([tensordict[key] for key in self.config.cond_keys], dim=-1)
        cond = self._normalize_morphology(cond)
        cond_out = self.cond_linear(self.cond_mlp(cond))
        film_params = self._split_film_params(cond_out)

        # --- FiLM-modulated trunk forward ---
        for block, (gamma, beta) in zip(self.trunk_blocks, film_params):
            x = block(x)
            x = x * gamma + beta

        x = self.head(x)

        if self.output_activation is not None:
            x = self.output_activation(x)

        tensordict[self.config.out_keys[0]] = x
        return tensordict
