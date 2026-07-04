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
"""Count total motions across the hhi_stage1 shard files and check divisibility.

Usage:
    python tools/count_stage1_motions.py --input-dir /home/hlz/datasets/hhi_stage1_raw \
        --pattern "batch_*_offset.pt" --num-files 6
"""

import argparse
from collections import Counter
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--pattern", default="batch_*_offset.pt")
    parser.add_argument("--num-files", type=int, default=6, help="Target number of output files")
    args = parser.parse_args()

    files = sorted(Path(args.input_dir).glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No files matching '{args.pattern}' in {args.input_dir}")

    total_motions = 0
    total_frames = 0
    per_shape = Counter()
    per_file_counts = []

    for i, f in enumerate(files):
        d = torch.load(f, map_location="cpu", weights_only=False)
        n = d["motion_lengths"].shape[0]
        total_motions += n
        total_frames += d["gts"].shape[0]
        per_file_counts.append(n)
        if "motion_asset_ids" in d:
            per_shape.update(d["motion_asset_ids"])
        del d
        if (i + 1) % 20 == 0 or i == len(files) - 1:
            print(f"  [{i + 1}/{len(files)}] {f.name}: {n} motions (running total {total_motions})")

    print(f"\nFiles scanned      : {len(files)}")
    print(f"Total motions      : {total_motions}")
    print(f"Total frames       : {total_frames}")
    print(f"Per-shape counts   : {dict(per_shape)}")
    print(f"Min/max per file   : {min(per_file_counts)} / {max(per_file_counts)}")

    n_out = args.num_files
    print(f"\nDivisible by {n_out}? {'YES' if total_motions % n_out == 0 else 'NO'}")
    print(f"  {total_motions} / {n_out} = {total_motions // n_out} remainder {total_motions % n_out}")

    for shape, cnt in per_shape.items():
        print(f"  shape '{shape}': {cnt} / {n_out} = {cnt // n_out} remainder {cnt % n_out}")


if __name__ == "__main__":
    main()
