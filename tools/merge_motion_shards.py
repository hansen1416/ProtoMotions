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
Merge N motion chunk files into M shards for multi-GPU training.

Usage:
    python tools/merge_motion_shards.py \
        --src /home/hlz/datasets/humos_proto/offset \
        --dst /home/hlz/datasets/humos_proto/merged4 \
        --num-shards 4

The output files are named humos_0.pt ... humos_3.pt.
Use --motion-file <dst>/humos_slurmrank.pt when training.

Peak RAM per shard = (chunks_per_shard * chunk_size) + one_chunk_size.
  4 shards from 16 × 3 GB chunks → ~15 GB peak  (fits on 32 GB)
  2 shards from 16 × 3 GB chunks → ~27 GB peak  (tight on 32 GB)
"""

import argparse
import gc
from pathlib import Path

import torch


# Keys whose first dimension is frame count (large tensors, preallocated)
FRAME_KEYS = ["gts", "grs", "gvs", "gavs", "dvs", "dps", "contacts", "lrs"]

# Keys whose first dimension is motion count (small tensors, just cat)
MOTION_TENSOR_KEYS = [
    "motion_lengths",
    "motion_dt",
    "motion_num_frames",
    "motion_weights",
    "motion_betas",
    "motion_gender_ids",
]

# length_starts is motion-indexed but needs a frame offset added per chunk
# Tuple/list keys are motion-indexed strings, just extended
MOTION_TUPLE_KEYS = [
    "motion_files",
    "motion_genders",
    "motion_beta_keys",
    "motion_asset_ids",
    "motion_clip_ids",
    "motion_npz_files",
]


def fill_chunk(out_frame, out_motion, out_length_starts, out_tuples, chunk, frame_off, motion_off):
    nf = chunk["gts"].shape[0]
    nm = chunk["length_starts"].shape[0]

    for k in FRAME_KEYS:
        if k in chunk:
            out_frame[k][frame_off : frame_off + nf] = chunk[k]

    for k in MOTION_TENSOR_KEYS:
        if k in chunk:
            out_motion[k][motion_off : motion_off + nm] = chunk[k]

    out_length_starts[motion_off : motion_off + nm] = chunk["length_starts"] + frame_off

    for k in MOTION_TUPLE_KEYS:
        if k in chunk:
            out_tuples[k].extend(chunk[k])


def merge_shard(group_files, dst_path):
    print(f"\nMerging {len(group_files)} chunks → {dst_path.name}")
    for f in group_files:
        print(f"  {f.name}")

    # Load first chunk to learn shapes, then preallocate
    print("  Loading first chunk to determine shapes...")
    first = torch.load(group_files[0], map_location="cpu", weights_only=False)
    frames_per_chunk = first["gts"].shape[0]
    motions_per_chunk = first["length_starts"].shape[0]
    n = len(group_files)
    total_frames = frames_per_chunk * n
    total_motions = motions_per_chunk * n
    print(f"  total_frames={total_frames:,}  total_motions={total_motions:,}")

    # Preallocate frame tensors
    out_frame = {}
    for k in FRAME_KEYS:
        if k in first:
            shape = (total_frames,) + first[k].shape[1:]
            out_frame[k] = torch.empty(shape, dtype=first[k].dtype)

    # Preallocate motion tensors
    out_motion = {}
    for k in MOTION_TENSOR_KEYS:
        if k in first:
            shape = (total_motions,) + first[k].shape[1:]
            out_motion[k] = torch.empty(shape, dtype=first[k].dtype)

    out_length_starts = torch.empty(total_motions, dtype=torch.int64)
    out_tuples = {k: [] for k in MOTION_TUPLE_KEYS}

    # Fill from first chunk, then free it
    fill_chunk(out_frame, out_motion, out_length_starts, out_tuples, first, 0, 0)
    del first
    gc.collect()

    # Fill remaining chunks one at a time
    for i, fpath in enumerate(group_files[1:], start=1):
        print(f"  Loading chunk {i+1}/{n}: {fpath.name}")
        chunk = torch.load(fpath, map_location="cpu", weights_only=False)
        fill_chunk(
            out_frame,
            out_motion,
            out_length_starts,
            out_tuples,
            chunk,
            frame_off=i * frames_per_chunk,
            motion_off=i * motions_per_chunk,
        )
        del chunk
        gc.collect()

    # Assemble output dict
    out = {}
    out.update(out_frame)
    out.update(out_motion)
    out["length_starts"] = out_length_starts
    for k, v in out_tuples.items():
        if v:
            out[k] = tuple(v)

    print(f"  Saving {dst_path} ...")
    torch.save(out, dst_path)
    size_gb = dst_path.stat().st_size / 1e9
    print(f"  Saved {dst_path.name}  ({size_gb:.1f} GB)")

    del out
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Merge motion chunk files into shards.")
    parser.add_argument("--src", required=True, help="Directory containing source chunk files")
    parser.add_argument("--dst", required=True, help="Directory to write merged shard files")
    parser.add_argument("--num-shards", type=int, default=4, help="Number of output shards (default: 4)")
    parser.add_argument("--pattern", default="humos_131072_*_offset.pt", help="Glob pattern for source files")
    parser.add_argument("--out-prefix", default="humos", help="Output filename prefix (e.g. 'humos' → humos_0.pt)")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    files = sorted(src.glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No files matching '{args.pattern}' in {src}")

    n = len(files)
    if n % args.num_shards != 0:
        raise ValueError(f"{n} files is not evenly divisible by --num-shards {args.num_shards}")

    chunks_per_shard = n // args.num_shards
    print(f"Found {n} files → {args.num_shards} shards of {chunks_per_shard} chunks each")

    for shard_idx in range(args.num_shards):
        group = files[shard_idx * chunks_per_shard : (shard_idx + 1) * chunks_per_shard]
        dst_path = dst / f"{args.out_prefix}_{shard_idx}.pt"
        merge_shard(group, dst_path)

    print(f"\nDone. Training flag: --motion-file {dst}/{args.out_prefix}_slurmrank.pt")


if __name__ == "__main__":
    main()
