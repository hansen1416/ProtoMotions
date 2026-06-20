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
"""Redistribute motions across slurmrank .pt files so that:

  1. Every output file has the same total number of motions.
  2. Every output file has the same number of motions per shape (asset_id).

This makes evaluators that assume equal clip counts per shape work correctly.

Motions are assumed to be globally ordered from easiest to hardest (across
the file sequence, file 0 first, file N-1 last).  When a shape's total
count is not divisible by the number of output files, the surplus is dropped
from the beginning (i.e. the easiest motions are removed).

Per-shape ordering is preserved: the easiest clips for each shape land in
output file 0, the hardest in output file N-1.

Usage:
    python tools/equalize_slurmrank_files.py \\
        --input-dir /home/hlz/datasets/humos_proto_neutral/offset \\
        --output-dir /home/hlz/datasets/humos_proto_neutral/offset_eq \\
        --base-name humanml3d_neutral_20951

Output files are named <prefix>_<new_total>_<rank:04d>.pt, e.g.:
    humanml3d_neutral_20946_0000.pt  (if 5 motions total are dropped)

Memory: at most 2 source files are kept in RAM simultaneously.
"""

import argparse
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import torch

FRAME_KEYS = ("gts", "grs", "gvs", "gavs", "dvs", "dps", "contacts", "lrs")
SCALAR_KEYS = (
    "motion_lengths",
    "motion_dt",
    "motion_num_frames",
    "motion_weights",
    "motion_betas",
    "motion_gender_ids",
)
TUPLE_KEYS = (
    "motion_files",
    "motion_genders",
    "motion_beta_keys",
    "motion_asset_ids",
    "motion_clip_ids",
)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_file(path: str) -> Dict:
    print(f"  Loading {os.path.basename(path)} ...", flush=True)
    return torch.load(path, map_location="cpu", weights_only=False)


def extract_indices(data: Dict, local_indices: List[int]) -> Dict:
    """Extract specific motions (by local index) from a loaded data dict.

    Returns a new dict with the same keys, frame tensors concatenated in the
    requested order, and length_starts recomputed from zero.
    Frames within each source file are contiguous, so each motion is a cheap
    single-slice operation.
    """
    if not local_indices:
        return {}

    n = len(local_indices)
    starts = data["length_starts"]
    nframes = data["motion_num_frames"]

    frame_parts = {k: [] for k in FRAME_KEYS}
    for i in local_indices:
        s = starts[i].item()
        f = nframes[i].item()
        for k in FRAME_KEYS:
            frame_parts[k].append(data[k][s : s + f])

    frame_data = {k: torch.cat(frame_parts[k], dim=0) for k in FRAME_KEYS}

    t_idx = torch.tensor(local_indices, dtype=torch.long)
    scalars = {k: data[k][t_idx] for k in SCALAR_KEYS}
    tuples = {k: tuple(data[k][i] for i in local_indices) for k in TUPLE_KEYS}

    new_nframes = scalars["motion_num_frames"]
    new_starts = torch.zeros(n, dtype=torch.long)
    new_starts[1:] = new_nframes[:-1].cumsum(0)

    return {**frame_data, **scalars, **tuples, "length_starts": new_starts}


def concat_two(a: Dict, b: Dict) -> Dict:
    """Concatenate two extracted-motion dicts."""
    if not a:
        return b
    if not b:
        return a
    merged: Dict = {}
    for k in FRAME_KEYS:
        merged[k] = torch.cat([a[k], b[k]], dim=0)
    for k in SCALAR_KEYS:
        merged[k] = torch.cat([a[k], b[k]], dim=0)
    for k in TUPLE_KEYS:
        merged[k] = a[k] + b[k]
    total_n = merged["motion_num_frames"]
    starts = torch.zeros(total_n.shape[0], dtype=torch.long)
    starts[1:] = total_n[:-1].cumsum(0)
    merged["length_starts"] = starts
    return merged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-name", required=True,
                        help="E.g. humanml3d_neutral_20951")
    parser.add_argument("--num-files", type=int, default=6)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    n_out = args.num_files

    input_paths = [
        os.path.join(args.input_dir, f"{args.base_name}_{i:04d}.pt")
        for i in range(n_out)
    ]
    for p in input_paths:
        assert os.path.exists(p), f"Missing: {p}"

    # ------------------------------------------------------------------
    # 1. Build global per-shape motion index (no frame data loaded yet)
    #    shape_index[asset_id] = [(file_idx, local_idx), ...]  in global order
    # ------------------------------------------------------------------
    print("Building global motion index ...")
    shape_index: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for file_idx, path in enumerate(input_paths):
        d = load_file(path)
        for local_idx, asset_id in enumerate(d["motion_asset_ids"]):
            shape_index[asset_id].append((file_idx, local_idx))
        del d

    shapes = sorted(shape_index.keys())
    print(f"Shapes found: {shapes}")
    for sh in shapes:
        print(f"  {sh}: {len(shape_index[sh])} motions")

    # ------------------------------------------------------------------
    # 2. For each shape, drop surplus from the beginning and compute
    #    per-output-file assignment
    # ------------------------------------------------------------------
    # shape_groups[asset_id][out_i] = [(file_idx, local_idx), ...]
    shape_groups: Dict[str, List[List[Tuple[int, int]]]] = {}
    per_shape_per_file: Dict[str, int] = {}

    for sh in shapes:
        motions = shape_index[sh]
        total_sh = len(motions)
        drop_sh = total_sh % n_out
        keep_sh = total_sh - drop_sh
        per_file_sh = keep_sh // n_out
        print(f"  {sh}: keep {keep_sh} / drop {drop_sh} (easiest) → {per_file_sh} per file")
        kept = motions[drop_sh:]   # drop from front = drop easiest
        groups = [kept[i * per_file_sh : (i + 1) * per_file_sh] for i in range(n_out)]
        shape_groups[sh] = groups
        per_shape_per_file[sh] = per_file_sh

    total_per_file = sum(per_shape_per_file.values())
    total_kept = total_per_file * n_out
    print(f"\nTotal per output file : {total_per_file}")
    print(f"Grand total kept      : {total_kept}")

    # Derive output base name
    prefix = args.base_name.rsplit("_", 1)[0]   # e.g. humanml3d_neutral
    out_base = f"{prefix}_{total_kept}"
    print(f"Output base name      : {out_base}\n")

    # ------------------------------------------------------------------
    # 3. Build each output file — load source files on demand, cache ≤ 2
    # ------------------------------------------------------------------
    cache: Dict[int, Dict] = {}

    def get_src(fi: int) -> Dict:
        if fi not in cache:
            if len(cache) >= 2:
                evict = next(k for k in cache if k != fi)
                print(f"    Evicting source file {evict} from cache")
                del cache[evict]
            cache[fi] = load_file(input_paths[fi])
        return cache[fi]

    for out_i in range(n_out):
        print(f"[{out_i + 1}/{n_out}] Building output file {out_i} ...")

        out_data: Dict = {}

        for sh in shapes:
            group = shape_groups[sh][out_i]  # [(file_idx, local_idx), ...]

            # Collect local indices per source file, preserving group order
            per_src: Dict[int, List[int]] = defaultdict(list)
            order: List[Tuple[int, int]] = []  # (file_idx, position-in-per_src)
            for file_idx, local_idx in group:
                pos = len(per_src[file_idx])
                per_src[file_idx].append(local_idx)
                order.append((file_idx, pos))

            # Extract from each source file
            src_extracted: Dict[int, Dict] = {}
            for fi in sorted(per_src.keys()):
                print(f"    Shape '{sh}', src file {fi}: {len(per_src[fi])} motions")
                src_extracted[fi] = extract_indices(get_src(fi), per_src[fi])

            # Re-assemble in group order using per-motion extraction from already-extracted blocks
            # (each block is small — O(1-2) source files per shape per output)
            parts_in_order = []
            if len(per_src) == 1:
                # Fast path: single source → already in order
                (fi,) = per_src.keys()
                parts_in_order = [src_extracted[fi]]
            else:
                # Two source files: split back out in group order
                # Build runs of consecutive (file_idx) to minimise per-motion overhead
                from itertools import groupby
                for fi, run in groupby(order, key=lambda x: x[0]):
                    positions = [pos for _, pos in run]
                    ex = src_extracted[fi]
                    if len(positions) == ex["motion_lengths"].shape[0]:
                        # entire extracted block in order
                        parts_in_order.append(ex)
                    else:
                        # partial run — extract those positions (rare edge case)
                        parts_in_order.append(extract_indices(ex, positions))

            shape_data = parts_in_order[0]
            for part in parts_in_order[1:]:
                shape_data = concat_two(shape_data, part)

            out_data = concat_two(out_data, shape_data)
            print(f"    Shape '{sh}' done: {shape_data['motion_lengths'].shape[0]} motions")

        out_path = os.path.join(args.output_dir, f"{out_base}_{out_i:04d}.pt")
        print(f"  Saving → {out_path} ...", end=" ", flush=True)
        torch.save(out_data, out_path)
        n_frames = out_data["gts"].shape[0]
        n_motions = out_data["motion_lengths"].shape[0]
        print(f"done  (motions={n_motions}, frames={n_frames})")

    print(f"\nAll {n_out} output files written to {args.output_dir}")
    slurmrank_path = os.path.join(args.output_dir, out_base + "_slurmrank.pt")
    print(f"Use --motion-file {slurmrank_path}")


if __name__ == "__main__":
    main()
