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
"""Merge the many `hhi_stage1` batch shards into N (default 6) balanced motion files.

Source data: r2:proto-data/hhi_stage1/batch_{BBBB}_{SSSS}_offset.pt (328 shards,
41,902 motions = 20,951 clips x 2 shapes [male_71fbbe41, female_71fbbe41]).
41,902 is not evenly divisible by 6 (remainder 4; each shape has remainder 5), so
this script drops the surplus per shape from the front (the easiest clips, since
shards are ordered easiest-to-hardest — same convention as
`tools/equalize_slurmrank_files.py`) before splitting evenly.

Output: N files with an equal total motion count AND an equal per-shape motion
count in each, so evaluators that assume balanced shapes still work.

Usage:
    python tools/count_stage1_motions.py --input-dir /home/hlz/datasets/hhi_stage1_raw
    python tools/merge_stage1_shards.py \\
        --input-dir /home/hlz/datasets/hhi_stage1_raw \\
        --output-dir /home/hlz/datasets/hhi_stage1_merged6 \\
        --num-files 6

Memory: at most 2 source shards held in RAM at once.
"""

import argparse
from collections import defaultdict
from itertools import groupby
from pathlib import Path
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
    "motion_npz_files",
)


def load_file(path: Path) -> Dict:
    print(f"  Loading {path.name} ...", flush=True)
    return torch.load(path, map_location="cpu", weights_only=False)


def extract_indices(data: Dict, local_indices: List[int]) -> Dict:
    """Extract specific motions (by local index) from a loaded shard, in order."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--pattern", default="batch_*_offset.pt")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-files", type=int, default=6)
    parser.add_argument("--out-prefix", default="hhi_stage1")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_out = args.num_files

    input_paths = sorted(Path(args.input_dir).glob(args.pattern))
    if not input_paths:
        raise FileNotFoundError(f"No files matching '{args.pattern}' in {args.input_dir}")
    print(f"Found {len(input_paths)} input shards")

    # ------------------------------------------------------------------
    # 1. Build global per-shape motion index (no frame data loaded yet)
    #    shape_index[asset_id] = [(file_idx, local_idx), ...] in global order
    #    (files are sorted, so this preserves the easiest->hardest ordering)
    # ------------------------------------------------------------------
    print("Building global motion index ...")
    shape_index: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for file_idx, path in enumerate(input_paths):
        d = torch.load(path, map_location="cpu", weights_only=False)
        for local_idx, asset_id in enumerate(d["motion_asset_ids"]):
            shape_index[asset_id].append((file_idx, local_idx))
        del d

    shapes = sorted(shape_index.keys())
    print(f"Shapes found: {shapes}")
    total_in = 0
    for sh in shapes:
        print(f"  {sh}: {len(shape_index[sh])} motions")
        total_in += len(shape_index[sh])
    print(f"Total motions in    : {total_in}")

    # ------------------------------------------------------------------
    # 2. Per shape, drop surplus from the front (easiest) so the kept
    #    count divides evenly by n_out, then split into n_out groups.
    # ------------------------------------------------------------------
    shape_groups: Dict[str, List[List[Tuple[int, int]]]] = {}
    per_shape_per_file: Dict[str, int] = {}

    for sh in shapes:
        motions = shape_index[sh]
        total_sh = len(motions)
        drop_sh = total_sh % n_out
        keep_sh = total_sh - drop_sh
        per_file_sh = keep_sh // n_out
        print(f"  {sh}: keep {keep_sh} / drop {drop_sh} (easiest) -> {per_file_sh} per file")
        kept = motions[drop_sh:]
        groups = [kept[i * per_file_sh : (i + 1) * per_file_sh] for i in range(n_out)]
        shape_groups[sh] = groups
        per_shape_per_file[sh] = per_file_sh

    total_per_file = sum(per_shape_per_file.values())
    total_kept = total_per_file * n_out
    print(f"\nTotal per output file : {total_per_file}")
    print(f"Grand total kept      : {total_kept}  (dropped {total_in - total_kept})")

    out_base = f"{args.out_prefix}_{total_kept}"
    print(f"Output base name      : {out_base}\n")

    # ------------------------------------------------------------------
    # 3. Build each output file, loading source shards on demand.
    #    File index only ever increases within a shape's kept range, and
    #    output files are built in order, so a 2-shard cache is enough.
    # ------------------------------------------------------------------
    cache: Dict[int, Dict] = {}

    def get_src(fi: int) -> Dict:
        if fi not in cache:
            if len(cache) >= 2:
                evict = next(k for k in cache if k != fi)
                del cache[evict]
            cache[fi] = load_file(input_paths[fi])
        return cache[fi]

    for out_i in range(n_out):
        print(f"[{out_i + 1}/{n_out}] Building output file {out_i} ...")
        out_data: Dict = {}

        for sh in shapes:
            group = shape_groups[sh][out_i]

            per_src: Dict[int, List[int]] = defaultdict(list)
            order: List[Tuple[int, int]] = []
            for file_idx, local_idx in group:
                pos = len(per_src[file_idx])
                per_src[file_idx].append(local_idx)
                order.append((file_idx, pos))

            src_extracted: Dict[int, Dict] = {}
            for fi in sorted(per_src.keys()):
                src_extracted[fi] = extract_indices(get_src(fi), per_src[fi])

            if len(per_src) == 1:
                (fi,) = per_src.keys()
                parts_in_order = [src_extracted[fi]]
            else:
                parts_in_order = []
                for fi, run in groupby(order, key=lambda x: x[0]):
                    positions = [pos for _, pos in run]
                    ex = src_extracted[fi]
                    if len(positions) == ex["motion_lengths"].shape[0]:
                        parts_in_order.append(ex)
                    else:
                        parts_in_order.append(extract_indices(ex, positions))

            shape_data = parts_in_order[0]
            for part in parts_in_order[1:]:
                shape_data = concat_two(shape_data, part)

            out_data = concat_two(out_data, shape_data)
            print(f"    Shape '{sh}' done: {shape_data['motion_lengths'].shape[0]} motions")

        out_path = out_dir / f"{out_base}_{out_i:04d}.pt"
        print(f"  Saving -> {out_path} ...", end=" ", flush=True)
        torch.save(out_data, out_path)
        n_frames = out_data["gts"].shape[0]
        n_motions = out_data["motion_lengths"].shape[0]
        print(f"done  (motions={n_motions}, frames={n_frames})")

    print(f"\nAll {n_out} output files written to {out_dir}")
    print(f"Use --motion-file {out_dir / (out_base + '_slurmrank.pt')}")


if __name__ == "__main__":
    main()
