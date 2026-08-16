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
Slice a single clip_id (all its shape variants, or a capped number of them) out of an
already-local motion `.pt` file into its own small `.pt` file.

Why: `examples/env_kinematic_playback.py` (the project's real kinematic-replay viewer --
plays reference motions on actual humanoid actors in a live IsaacGym/MuJoCo/Newton/Genesis
viewer, no policy or checkpoint needed) assigns motions to envs via the normal
motion_manager sampling. With `--simulator mujoco --num-envs 1` (the only combination that
runs without a CUDA GPU on this machine), that sampling would pick an arbitrary motion out
of whatever file is passed in -- not necessarily the specific clip you want to inspect.
Pointing it at a file that contains only that one clip makes the assignment deterministic.

Purely local tensor slicing -- no rclone/R2, no GPU, just reads and re-saves the existing
`.pt` dict format (gts/grs/gvs/gavs/dvs/dps/contacts/lrs + per-motion metadata).

Usage:
    python tools/extract_single_clip_motion.py \\
        --motion-file data_cache/small150_128shape.pt \\
        --clip-id M002028 \\
        --output /tmp/M002028_only.pt \\
        --max-shapes 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

# Tensor fields indexed per-FRAME (concatenated across all motions; sliced via length_starts).
FRAME_FIELDS = ["gts", "grs", "gvs", "gavs", "dvs", "dps", "contacts", "lrs"]
# Tensor/tuple fields indexed per-MOTION (one entry per motion_id).
MOTION_FIELDS = [
    "motion_num_frames",
    "motion_lengths",
    "motion_dt",
    "motion_weights",
    "motion_betas",
    "motion_gender_ids",
    "motion_files",
    "motion_genders",
    "motion_beta_keys",
    "motion_asset_ids",
    "motion_clip_ids",
    "motion_npz_files",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--motion-file", required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--max-shapes", type=int, default=None, help="Cap the number of shape variants kept (default: all)"
    )
    args = parser.parse_args()

    data = torch.load(args.motion_file, map_location="cpu", weights_only=False)
    clip_ids = data["motion_clip_ids"]
    motion_ids = [i for i, c in enumerate(clip_ids) if c == args.clip_id]
    if not motion_ids:
        raise SystemExit(f"clip_id {args.clip_id!r} not found in {args.motion_file}")
    if args.max_shapes is not None:
        motion_ids = motion_ids[: args.max_shapes]
    print(f"Found {len(motion_ids)} shape variant(s) of {args.clip_id}")

    length_starts = data["length_starts"]
    num_frames = data["motion_num_frames"]

    out = {}
    new_length_starts = []
    running = 0
    for field in FRAME_FIELDS:
        chunks = []
        for mid in motion_ids:
            s = length_starts[mid].item()
            n = num_frames[mid].item()
            chunks.append(data[field][s : s + n])
        out[field] = torch.cat(chunks, dim=0)

    for mid in motion_ids:
        new_length_starts.append(running)
        running += num_frames[mid].item()
    out["length_starts"] = torch.tensor(new_length_starts, dtype=torch.long)

    for field in MOTION_FIELDS:
        if field == "length_starts":
            continue
        src = data[field]
        if isinstance(src, torch.Tensor):
            out[field] = src[torch.tensor(motion_ids)]
        else:  # tuple of python objects
            out[field] = tuple(src[mid] for mid in motion_ids)

    out["motion_weights"] = torch.ones(len(motion_ids), dtype=torch.float32)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, output_path)
    print(f"-> Saved {output_path} ({len(motion_ids)} motion(s), {out['gts'].shape[0]} total frames)")


if __name__ == "__main__":
    main()
