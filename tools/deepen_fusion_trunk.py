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
"""Insert function-preserving identity layers into a FusionAdapterMLPWithConcat trunk.

`_actor.mu.mlp` is a plain Linear->ReLU stack, uniform width at every hidden layer, no
LayerNorm/residual in the way. That makes a Net2Net-style "deepen" possible: insert a new
`Linear(hidden, hidden)` initialized to the identity matrix (bias=0) right after an existing
ReLU. Since that ReLU's output is already >= 0, ReLU(Identity(x)) == x exactly -- the network
computes bit-for-bit the same thing the moment the layer is inserted, and the new layer is
free to actually learn from there instead of disrupting a checkpoint that already has real
progress in it. See note/README.note.md and `examples/experiments/mimic/
mlp_wide_fusion_stage2_deepened.py` for why (hhi_wide_fusion_stage2_unfrozen plateaued ~78-82%
success_rate; this tests whether depth was the limiting factor, cheaply, without a full restart).

Only `_actor.mu.mlp.*` is touched. `_actor.mu.norm.*`, `_actor.mu.beta_encoder.*`,
`_actor.mu.fusion_mlp.*`, and all `_critic.*` keys pass through unchanged.

`actor_optimizer` state is left untouched (stale) rather than deleted: the new actor has more
params than the old optimizer state was saved for, so `Adam.load_state_dict` raises `ValueError`
("parameter group that doesn't match the size of optimizer's group") when the checkpoint is
loaded into the deepened experiment. That's caught by the existing `allow_partial_checkpoint_load`
path in `protomotions/agents/ppo/agent.py:load_parameters` (`except ValueError: ... Skipping
actor_optimizer state load`), same as every prior warm-start in this codebase whose actor gained
new params. Deleting the key instead (an earlier version of this script did this) causes a plain
`KeyError` on `state_dict["actor_optimizer"]` itself, which happens *before* `load_state_dict`
runs and is not caught -- don't do that. `critic_optimizer` is real and valid since the critic
isn't modified.

Usage:
    python tools/deepen_fusion_trunk.py \\
        --checkpoint results/hhi_wide_fusion_stage2_unfrozen/last.ckpt \\
        --output results/hhi_wide_fusion_stage2_unfrozen/last_deepened1.ckpt \\
        --num-new-layers 1 --verify
"""

import argparse
import torch
from torch import nn

MLP_PREFIX = "_actor.mu.mlp."


def _hidden_and_final_indices(mlp_sd: dict):
    indices = sorted({int(k.split(".")[0]) for k in mlp_sd})
    return indices[:-1], indices[-1]


def deepen_mlp_state_dict(mlp_sd: dict, num_new_layers: int) -> dict:
    """Return a new `_actor.mu.mlp.*`-style state dict with `num_new_layers` identity
    layers inserted right before the final projection layer."""
    hidden_indices, final_idx = _hidden_and_final_indices(mlp_sd)
    hidden_dim = mlp_sd[f"{hidden_indices[-1]}.bias"].shape[0]

    new_sd = {}
    for idx in hidden_indices:
        new_sd[f"{idx}.weight"] = mlp_sd[f"{idx}.weight"].clone()
        new_sd[f"{idx}.bias"] = mlp_sd[f"{idx}.bias"].clone()

    next_idx = final_idx
    for _ in range(num_new_layers):
        new_sd[f"{next_idx}.weight"] = torch.eye(hidden_dim)
        new_sd[f"{next_idx}.bias"] = torch.zeros(hidden_dim)
        next_idx += 2

    new_sd[f"{next_idx}.weight"] = mlp_sd[f"{final_idx}.weight"].clone()
    new_sd[f"{next_idx}.bias"] = mlp_sd[f"{final_idx}.bias"].clone()
    return new_sd


def _build_sequential(mlp_sd: dict) -> nn.Sequential:
    """Rebuild the Linear->ReLU->...->Linear stack described by an `_actor.mu.mlp.*`
    state dict, for standalone (no-IsaacGym) verification."""
    indices = sorted({int(k.split(".")[0]) for k in mlp_sd})
    modules = []
    for i, idx in enumerate(indices):
        w = mlp_sd[f"{idx}.weight"]
        linear = nn.Linear(w.shape[1], w.shape[0])
        with torch.no_grad():
            linear.weight.copy_(w)
            linear.bias.copy_(mlp_sd[f"{idx}.bias"])
        modules.append(linear)
        if i < len(indices) - 1:
            modules.append(nn.ReLU())
    return nn.Sequential(*modules)


def verify(old_mlp_sd: dict, new_mlp_sd: dict):
    old_net = _build_sequential(old_mlp_sd)
    new_net = _build_sequential(new_mlp_sd)

    in_dim = old_mlp_sd[f"{_hidden_and_final_indices(old_mlp_sd)[0][0]}.weight"].shape[1]
    x = torch.randn(8, in_dim)
    with torch.no_grad():
        old_out = old_net(x)
        new_out = new_net(x)

    max_diff = (old_out - new_out).abs().max().item()
    assert max_diff < 1e-5, f"Deepened trunk output diverged from original: max_diff={max_diff}"

    old_params = sum(p.numel() for p in old_net.parameters())
    new_params = sum(p.numel() for p in new_net.parameters())
    print(f"[verify] output match: max_diff={max_diff:.2e} (< 1e-5)")
    print(f"[verify] param count: old={old_params:,} new={new_params:,} (+{new_params - old_params:,})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Path to a FusionAdapterMLPWithConcat checkpoint")
    parser.add_argument("--output", default=None, help="Output path (default: <checkpoint>_deepened<N>.ckpt)")
    parser.add_argument("--num-new-layers", type=int, default=1, help="Number of identity layers to insert")
    parser.add_argument("--verify", action="store_true", help="Run the standalone output-match check before saving")
    parser.add_argument("--dry-run", action="store_true", help="Verify only, do not save")
    args = parser.parse_args()

    print(f"Loading: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    model_sd = ckpt["model"]

    old_mlp_sd = {
        k[len(MLP_PREFIX):]: v for k, v in model_sd.items() if k.startswith(MLP_PREFIX)
    }
    assert old_mlp_sd, f"No keys found under {MLP_PREFIX!r} -- is this a FusionAdapterMLPWithConcat checkpoint?"

    new_mlp_sd = deepen_mlp_state_dict(old_mlp_sd, args.num_new_layers)

    if args.verify or args.dry_run:
        verify(old_mlp_sd, new_mlp_sd)

    if args.dry_run:
        print("\n[dry-run] No changes saved.")
        return

    for k in list(model_sd.keys()):
        if k.startswith(MLP_PREFIX):
            del model_sd[k]
    for k, v in new_mlp_sd.items():
        model_sd[f"{MLP_PREFIX}{k}"] = v

    if "actor_optimizer" in ckpt:
        print(
            "Leaving actor_optimizer state as-is (stale) -- it will fail to load with a "
            "ValueError (param-group size mismatch) once the trunk is deepened, which "
            "allow_partial_checkpoint_load=True catches and skips. Do not delete this key: "
            "that turns into an uncaught KeyError instead. See module docstring."
        )

    out_path = args.output
    if out_path is None:
        out_path = args.checkpoint.replace(".ckpt", f"_deepened{args.num_new_layers}.ckpt")

    torch.save(ckpt, out_path)
    print(f"\nSaved to: {out_path}")
    print(f"Inserted {args.num_new_layers} identity layer(s) into {MLP_PREFIX}*")


if __name__ == "__main__":
    main()
