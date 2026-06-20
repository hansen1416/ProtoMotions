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
"""Reset the morphology_obs slice of the running obs normalizer in a Stage 1 checkpoint.

After Stage 1 (neutral, beta=0), the last 10 beta dims of the obs normalizer have
var ≈ 0 because the input was always a constant (zero betas). When Stage 2 introduces
real betas, those dims get divided by sqrt(~0 + 1e-5) ≈ 0.003 and immediately saturate
at the clamp value (5), stalling adaptation.

Fix: reset var[-N_MORPHOLOGY:] = 1.0 so Stage 2 betas land in a reasonable normalized
range from step 1. mean[-N_MORPHOLOGY:] is also zeroed for symmetry (it should already
be near-zero after neutral training). count is left untouched so locomotion dim stats
continue updating at the same momentum.

Usage:
    python tools/reset_morphology_normalizer.py \\
        --checkpoint results/hhi_stage1_neutral/last.ckpt \\
        --output    results/hhi_stage1_neutral/last_morph_reset.ckpt

    # Dry-run: print stats without saving
    python tools/reset_morphology_normalizer.py \\
        --checkpoint results/hhi_stage1_neutral/last.ckpt \\
        --dry-run
"""

import argparse
import torch


# morphology_obs = [gender_id (1), betas/3.0 (10)] = 11 dims, always last in the obs concat
N_MORPHOLOGY = 11

NORM_KEYS = [
    "_actor.mu.norm.running_obs_norm",
    "_critic.norm.running_obs_norm",
]


def print_stats(model_sd: dict, label: str):
    for base in NORM_KEYS:
        mean = model_sd[f"{base}.mean"]
        var  = model_sd[f"{base}.var"]
        count = model_sd[f"{base}.count"]
        print(f"\n  [{label}] {base}")
        print(f"    obs_dim={mean.shape[0]}  count={count.item():.2e}")
        print(f"    mean[-{N_MORPHOLOGY}:] = {mean[-N_MORPHOLOGY:].tolist()}")
        print(f"    var [-{N_MORPHOLOGY}:] = {var[-N_MORPHOLOGY:].tolist()}")


def reset_morphology_norm(model_sd: dict):
    for base in NORM_KEYS:
        mean = model_sd[f"{base}.mean"]
        var  = model_sd[f"{base}.var"]
        mean[-N_MORPHOLOGY:] = 0.0
        var[-N_MORPHOLOGY:]  = 1.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to Stage 1 last.ckpt")
    parser.add_argument("--output", default=None, help="Output path (default: <checkpoint>_morph_reset.ckpt)")
    parser.add_argument("--dry-run", action="store_true", help="Print stats only, do not save")
    args = parser.parse_args()

    print(f"Loading: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    model_sd = ckpt["model"]

    print("\n=== BEFORE reset ===")
    print_stats(model_sd, "before")

    if args.dry_run:
        print("\n[dry-run] No changes saved.")
        return

    reset_morphology_norm(model_sd)

    print("\n=== AFTER reset ===")
    print_stats(model_sd, "after")

    out_path = args.output
    if out_path is None:
        out_path = args.checkpoint.replace(".ckpt", "_morph_reset.ckpt")

    torch.save(ckpt, out_path)
    print(f"\nSaved to: {out_path}")
    print("Use this checkpoint as --checkpoint for Stage 2 training.")


if __name__ == "__main__":
    main()
