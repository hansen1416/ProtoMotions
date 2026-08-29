# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""Actor transformer with morphology-conditioned adaptive LayerNorm-Zero blocks."""

from dataclasses import dataclass, field

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase
from torch import nn

from protomotions.agents.common.config import TransformerConfig
from protomotions.agents.utils.training import get_activation_func


@dataclass
class MorphologyAdaLNZeroTransformerConfig(TransformerConfig):
    """Configuration for a transformer conditioned by ``[gender_id, SMPL betas]``."""

    _target_: str = (
        "protomotions.agents.common.morphology_transformer."
        "MorphologyAdaLNZeroTransformer"
    )
    condition_key: str = field(
        default="morphology_obs",
        metadata={"help": "TensorDict key containing [gender_id, ten SMPL betas]."},
    )
    condition_hidden_dim: int = field(
        default=128,
        metadata={"help": "Hidden width of the morphology encoder.", "min": 1},
    )
    beta_norm_scale: float = field(
        default=3.0,
        metadata={"help": "Fixed divisor for SMPL betas before conditioning."},
    )

    def __post_init__(self):
        super().__post_init__()
        assert self.condition_key in self.in_keys, (
            f"condition_key '{self.condition_key}' must be present in in_keys"
        )
        assert self.beta_norm_scale > 0, "beta_norm_scale must be positive"


class AdaLNZeroEncoderLayer(nn.Module):
    """Pre-norm transformer block with zero-initialized conditional residual gates."""

    def __init__(self, config: MorphologyAdaLNZeroTransformerConfig):
        super().__init__()
        dim = config.latent_dim
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(dim, config.ff_size),
            get_activation_func(config.activation),
            nn.Dropout(config.dropout),
            nn.Linear(config.ff_size, dim),
        )
        self.attention_dropout = nn.Dropout(config.dropout)
        self.ffn_dropout = nn.Dropout(config.dropout)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim),
        )
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    @staticmethod
    def _modulate(
        x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        (
            shift_attention,
            scale_attention,
            gate_attention,
            shift_ffn,
            scale_ffn,
            gate_ffn,
        ) = self.modulation(condition).chunk(6, dim=-1)

        attention_input = self._modulate(
            self.norm1(x), shift_attention, scale_attention
        )
        attention_output = self.attention(
            attention_input,
            attention_input,
            attention_input,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        x = x + gate_attention.unsqueeze(1) * self.attention_dropout(attention_output)

        ffn_input = self._modulate(self.norm2(x), shift_ffn, scale_ffn)
        x = x + gate_ffn.unsqueeze(1) * self.ffn_dropout(self.ffn(ffn_input))
        return x


class MorphologyAdaLNZeroTransformer(TensorDictModuleBase):
    """Temporal transformer whose actor blocks are conditioned by body morphology."""

    config: MorphologyAdaLNZeroTransformerConfig

    def __init__(self, config: MorphologyAdaLNZeroTransformerConfig):
        super().__init__()
        self.config = config
        self.in_keys = config.in_keys
        self.out_keys = config.out_keys

        mask_keys = (
            set(config.input_and_mask_mapping.values())
            if config.input_and_mask_mapping
            else set()
        )
        self._token_input_keys = [
            key
            for key in config.in_keys
            if key != config.condition_key and key not in mask_keys
        ]
        assert self._token_input_keys, "At least one temporal token input is required"

        self.slot_embeddings = None
        if config.use_learned_slot_embeddings:
            self.slot_embeddings = nn.Parameter(
                torch.empty(1, config.max_sequence_length, config.latent_dim)
            )
            nn.init.normal_(self.slot_embeddings, mean=0.0, std=0.02)

        self.token_type_embeddings = None
        if config.use_learned_token_type_embeddings:
            self.token_type_embeddings = nn.Parameter(
                torch.empty(len(self._token_input_keys), config.latent_dim)
            )
            nn.init.normal_(self.token_type_embeddings, mean=0.0, std=0.02)

        self.condition_encoder = nn.Sequential(
            nn.Linear(11, config.condition_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.condition_hidden_dim, config.latent_dim),
            nn.SiLU(),
        )
        self.layers = nn.ModuleList(
            [AdaLNZeroEncoderLayer(config) for _ in range(config.num_layers)]
        )
        self.output_activation = (
            get_activation_func(config.output_activation)
            if config.output_activation is not None
            else None
        )

    def _normalize_morphology(self, morphology: torch.Tensor) -> torch.Tensor:
        if morphology.shape[-1] != 11:
            raise ValueError(
                f"Expected morphology [gender_id, 10 betas] with size 11, got "
                f"{morphology.shape[-1]}"
            )
        gender = 2.0 * morphology[..., :1] - 1.0
        betas = (morphology[..., 1:] / self.config.beta_norm_scale).clamp(-1.0, 1.0)
        return torch.cat((gender, betas), dim=-1)

    def _build_tokens(self, tensordict: TensorDict) -> torch.Tensor:
        token_groups = []
        for token_type, key in enumerate(self._token_input_keys):
            tokens = tensordict[key]
            if tokens.dim() == 2:
                tokens = tokens.unsqueeze(1)
            if tokens.shape[-1] != self.config.latent_dim:
                raise ValueError(
                    f"Transformer input '{key}' has token size {tokens.shape[-1]}, "
                    f"expected {self.config.latent_dim}"
                )
            if self.token_type_embeddings is not None:
                tokens = tokens + self.token_type_embeddings[token_type].view(1, 1, -1)
            token_groups.append(tokens)

        tokens = torch.cat(token_groups, dim=1)
        if self.slot_embeddings is not None:
            if tokens.shape[1] > self.slot_embeddings.shape[1]:
                raise ValueError(
                    f"Received {tokens.shape[1]} tokens, but slot embeddings support "
                    f"at most {self.slot_embeddings.shape[1]}"
                )
            tokens = tokens + self.slot_embeddings[:, : tokens.shape[1]]
        return tokens

    def _build_mask(self, tensordict: TensorDict) -> torch.Tensor:
        masks = []
        mapping = self.config.input_and_mask_mapping or {}
        for key in self._token_input_keys:
            tokens = tensordict[key]
            if key in mapping:
                mask = tensordict[mapping[key]].logical_not()
                masks.append(mask.unsqueeze(1) if mask.dim() == 1 else mask)
            else:
                sequence_length = 1 if tokens.dim() == 2 else tokens.shape[1]
                masks.append(
                    torch.zeros(
                        tokens.shape[0],
                        sequence_length,
                        dtype=torch.bool,
                        device=tokens.device,
                    )
                )
        return torch.cat(masks, dim=1)

    def forward(self, tensordict: TensorDict) -> TensorDict:
        tokens = self._build_tokens(tensordict)
        mask = self._build_mask(tensordict)
        condition = self.condition_encoder(
            self._normalize_morphology(tensordict[self.config.condition_key])
        )
        for layer in self.layers:
            tokens = layer(tokens, condition, mask)

        output = tokens[:, 0]
        if self.output_activation is not None:
            output = self.output_activation(output)
        tensordict[self.out_keys[0]] = output
        return tensordict
