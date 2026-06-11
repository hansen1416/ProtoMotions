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
"""Shape-embedding MLP for morphology-aware policy and value networks.

Replaces FiLM conditioning with a simpler design: morphology (gender + shape betas)
is passed through a small nonlinear encoder to produce a compact shape embedding,
which is then concatenated with the normalized main observations before the trunk.

    shape_embed = encoder(normalize(morphology_obs))
    trunk_input = cat(normalize(main_obs), shape_embed)
    output      = trunk(trunk_input)

This avoids FiLM's multiplicative instability and fanout bottleneck while still
giving the network a learned, nonlinear shape representation rather than raw betas.

See note/README.shape-embedding.md for design rationale and ablation plan.

Key Classes:
    ShapeEmbedMLPConfig  — configuration dataclass
    ShapeEmbedMLP        — TensorDictModuleBase implementation
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
class ShapeEmbedMLPConfig(NormObsBaseConfig):
    """Configuration for shape-embedding MLP.

    Drop-in replacement for FiLMMLPConfig. The cond_keys (default:
    ["morphology_obs"]) are encoded by a small MLP into a shape embedding
    that is concatenated with the normalized main observations before the trunk.

    The embedding dimension is implicitly cond_hidden_units[-1].

    The morphology tensor is expected to be [gender_id, beta_0, ..., beta_9]
    where gender_id is in {0, 1} and betas are in [-3, 3]. Betas are
    normalized by dividing by beta_norm_scale before being fed to the encoder.
    """

    _target_: str = "protomotions.agents.common.shape_embed_mlp.ShapeEmbedMLP"

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
        metadata={"help": "Hidden layer configs for the main trunk."},
    )
    cond_hidden_units: List[int] = field(
        default_factory=lambda: [64],
        metadata={
            "help": "Hidden unit counts for the shape encoder. "
            "The last value is the embedding dimension fed into the trunk."
        },
    )
    cond_activation: str = field(
        default="silu",
        metadata={"help": "Activation function used inside the shape encoder."},
    )
    beta_norm_scale: float = field(
        default=3.0,
        metadata={"help": "Divide beta dims by this before encoding (betas live in [-3, 3])."},
    )
    output_activation: Optional[str] = field(
        default=None,
        metadata={"help": "Optional activation applied to the final output."},
    )

    def __post_init__(self):
        assert self.num_out is not None, "ShapeEmbedMLPConfig: num_out must be provided"
        assert self.layers, "ShapeEmbedMLPConfig: layers must be provided"
        assert self.cond_hidden_units, "ShapeEmbedMLPConfig: cond_hidden_units must be non-empty"
        assert len(self.out_keys) == 1, "ShapeEmbedMLPConfig requires exactly one out_key"


class ShapeEmbedMLP(TensorDictModuleBase):
    """Shape-embedding MLP, usable as actor trunk, critic, or discriminator.

    Architecture
    ------------
    Shape encoder (morphology):
        cat(cond_keys) → normalize_morphology → encoder → shape_embed

    Main branch (state/task obs):
        cat(in_keys) → normalize → cat(shape_embed) → trunk → head

    The encoder and trunk both use LazyLinear, so input shapes are inferred
    on the first forward pass.
    """

    config: ShapeEmbedMLPConfig

    def __init__(self, config: ShapeEmbedMLPConfig):
        TensorDictModuleBase.__init__(self)
        self.config = config

        # Running-mean/std normalizer for the main observations (lazy init).
        self.norm = NormObsBase(config)

        # Shape encoder: small MLP that projects morphology → shape embedding.
        encoder_layers = []
        for units in config.cond_hidden_units:
            encoder_layers.append(nn.LazyLinear(units))
            encoder_layers.append(get_activation_func(config.cond_activation))
        self.shape_encoder = nn.Sequential(*encoder_layers)

        # Main trunk + output head: standard MLP with LazyLinear.
        trunk_layers = []
        for i, layer_cfg in enumerate(config.layers):
            trunk_layers.append(nn.LazyLinear(layer_cfg.units))
            if layer_cfg.use_layer_norm and i == 0:
                trunk_layers.append(nn.LayerNorm(layer_cfg.units))
            trunk_layers.append(get_activation_func(layer_cfg.activation))
        trunk_layers.append(nn.LazyLinear(config.num_out))
        self.trunk = nn.Sequential(*trunk_layers)

        self.output_activation = None
        if config.output_activation is not None:
            self.output_activation = get_activation_func(config.output_activation)

        # TensorDict key declarations: union of main and conditioning keys.
        cond_only = [k for k in config.cond_keys if k not in config.in_keys]
        self.in_keys = config.in_keys + cond_only
        self.out_keys = config.out_keys

    def _normalize_morphology(self, cond: torch.Tensor) -> torch.Tensor:
        """Normalize morphology: gender stays in {0,1}; betas divided by beta_norm_scale."""
        gender = cond[..., :1]
        betas = cond[..., 1:] / self.config.beta_norm_scale
        return torch.cat([gender, betas], dim=-1)

    def forward(self, tensordict: TensorDict) -> TensorDict:
        # Main obs: concatenate and normalize.
        x = torch.cat([tensordict[key] for key in self.config.in_keys], dim=-1)
        x = self.norm(x)

        # Morphology: normalize and encode to compact shape embedding.
        cond = torch.cat([tensordict[key] for key in self.config.cond_keys], dim=-1)
        cond = self._normalize_morphology(cond)
        shape_embed = self.shape_encoder(cond)

        # Concatenate embedding with main obs and run through trunk.
        x = torch.cat([x, shape_embed], dim=-1)
        x = self.trunk(x)

        if self.output_activation is not None:
            x = self.output_activation(x)

        tensordict[self.config.out_keys[0]] = x
        return tensordict
