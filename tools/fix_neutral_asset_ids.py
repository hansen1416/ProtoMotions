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
"""Overwrite motion_asset_ids, motion_genders, and motion_gender_ids in
slurmrank .pt files so all motions share a single "neutral_neutral" asset_id.

Usage:
    python tools/fix_neutral_asset_ids.py --dir /path/to/offset_eq --base-name humanml3d_neutral_20946
"""

import argparse
import os
import torch


def fix_file(path: str) -> None:
    print(f"Processing {os.path.basename(path)} ...", end=" ", flush=True)
    d = torch.load(path, map_location="cpu", weights_only=False)
    n = len(d["motion_asset_ids"])

    d["motion_asset_ids"] = tuple("neutral_neutral" for _ in range(n))
    d["motion_genders"] = tuple("neutral" for _ in range(n))
    d["motion_gender_ids"] = torch.zeros(n, dtype=torch.long)

    torch.save(d, path)
    print(f"done  ({n} motions)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Directory containing the .pt files")
    parser.add_argument("--base-name", required=True, help="E.g. humanml3d_neutral_20946")
    parser.add_argument("--num-files", type=int, default=6)
    args = parser.parse_args()

    for i in range(args.num_files):
        path = os.path.join(args.dir, f"{args.base_name}_{i:04d}.pt")
        assert os.path.exists(path), f"Missing: {path}"
        fix_file(path)

    print("All files updated.")


if __name__ == "__main__":
    main()
