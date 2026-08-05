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

"""
Verify that info/episode_length (~115-123, flat since ~epoch 1000) on
hhi_wide_150motion_128shape_discover is explained by clip length + RSI, not by
the policy repeatedly failing/falling around step 120.

Evidence already gathered from wandb (o4luvqyt): env/termination/fall_mean sits at
~0.0001-0.0002 per step the whole run -- falls are almost never the cause of a training
rollout reset. mimic_control.py's check_resets_and_terminations() resets an env whenever
the current clip finishes (done_clip), regardless of falling (bootstrap_on_episode_end).
So info/episode_length should just track "time until the sampled clip naturally ends,"
which is a function of clip length and the motion_manager's RSI start-time sampling --
not a sign of a stuck training plateau.

This script reproduces motion_manager.py's exact RSI logic (sample_time + init_start_prob
bernoulli override, see envs/motion_manager/motion_manager.py:355-437) against the real
motion library, and reports the predicted mean/median episode length in control steps.
Compare the printed prediction against the observed info/episode_length in wandb.

Runs on CPU only -- no GPU/simulator needed, safe to run alongside ongoing training.

Usage:
    python tools/verify_episode_length.py \\
        --motion-file /workspace/motion_cache/small150_128shape.pt \\
        --init-start-prob 0.2 \\
        --control-fps 30
"""

from __future__ import annotations

import argparse

import torch

from protomotions.components.motion_lib import MotionLib, MotionLibConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-file", type=str, required=True)
    parser.add_argument(
        "--init-start-prob",
        type=float,
        default=0.2,
        help="Must match the experiment's MimicMotionManagerConfig.init_start_prob.",
    )
    parser.add_argument(
        "--control-fps",
        type=float,
        default=30.0,
        help="Control-step rate = sim fps / decimation (isaacgym: 60/2=30, confirmed via "
        "wandb run.config for this run).",
    )
    parser.add_argument("--num-samples", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    env_dt = 1.0 / args.control_fps

    motion_lib = MotionLib(
        config=MotionLibConfig(motion_file=args.motion_file), device="cpu"
    )
    lengths_sec = motion_lib.motion_lengths  # [num_motions], seconds
    num_motions = len(lengths_sec)
    lengths_steps = lengths_sec / env_dt

    print(f"Loaded {num_motions} motions from {args.motion_file}")
    print()
    print("=== Raw clip length distribution (uniform over all motions) ===")
    print(f"  mean:   {lengths_sec.mean().item():.3f} s  ({lengths_steps.mean().item():.1f} steps)")
    print(f"  median: {lengths_sec.median().item():.3f} s  ({lengths_steps.median().item():.1f} steps)")
    print(f"  min:    {lengths_sec.min().item():.3f} s  ({lengths_steps.min().item():.1f} steps)")
    print(f"  max:    {lengths_sec.max().item():.3f} s  ({lengths_steps.max().item():.1f} steps)")
    print()

    # Reproduce motion_manager.py's sample_motions/sample_time exactly:
    #   phase = Uniform(0,1); motion_time = phase * motion_lengths[motion_ids]
    #   if init_start_prob > 0: with prob init_start_prob, force motion_time = 0
    # Predicted episode length (assuming no fall) = time remaining until clip end
    # from that start point = motion_lengths[motion_ids] - motion_time.
    motion_ids = torch.randint(0, num_motions, (args.num_samples,))
    clip_len = lengths_sec[motion_ids]

    phase = torch.rand(args.num_samples)
    motion_time = phase * clip_len

    init_start = torch.bernoulli(torch.full((args.num_samples,), args.init_start_prob))
    motion_time = torch.where(init_start == 1, torch.zeros_like(motion_time), motion_time)

    remaining_sec = clip_len - motion_time
    remaining_steps = remaining_sec / env_dt

    print(f"=== Predicted episode length under real RSI logic (init_start_prob={args.init_start_prob}) ===")
    print("(uniform motion sampling assumed -- ignores motion_weights curriculum reweighting,")
    print(" and assumes falls never cut an episode short, which the ~0.0001-0.0002/step")
    print(" training fall rate on this run supports)")
    print(f"  mean:   {remaining_steps.mean().item():.1f} steps")
    print(f"  median: {remaining_steps.median().item():.1f} steps")
    print()
    print("Compare this to the observed info/episode_length in wandb (o4luvqyt: ~115-123,")
    print("flat since ~epoch 1000). A close match supports 'episode length is clip-length-")
    print("driven, not a failure plateau'; a large mismatch means the hypothesis is wrong")
    print("and needs revisiting.")


if __name__ == "__main__":
    main()
