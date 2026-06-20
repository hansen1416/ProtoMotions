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
"""Recover motion_asset_ids, motion_genders, and motion_gender_ids from the
motion_files field (which encodes gender in the filename).

Fixes files that were incorrectly overwritten by fix_neutral_asset_ids.py.

Usage:
    python tools/recover_neutral_asset_ids.py \
        --dir /workspace/20946_neutral_offset \
        --base-name humanml3d_neutral_20946
"""

import argparse
import os
from pathlib import Path

import torch

GENDER_TO_ID = {"female": -1, "neutral": 0, "male": 1}
KNOWN_GENDERS = {"female", "male", "neutral"}


def gender_from_motion_file(path: str) -> str:
    """Extract gender from filename like 'M004501_v00_female_neutral.motion'."""
    stem = Path(path).stem  # e.g. M004501_v00_female_neutral
    for gender in ("female", "male", "neutral"):
        if f"_{gender}_" in stem or stem.endswith(f"_{gender}"):
            return gender
    raise ValueError(f"Cannot infer gender from motion file path: {path}")


def fix_file(path: str) -> None:
    print(f"Processing {os.path.basename(path)} ...", end=" ", flush=True)
    d = torch.load(path, map_location="cpu", weights_only=False)

    motion_files = d["motion_files"]
    motion_beta_keys = d["motion_beta_keys"]

    genders, gender_ids, asset_ids = [], [], []
    for mf, beta_key in zip(motion_files, motion_beta_keys):
        gender = gender_from_motion_file(mf)
        genders.append(gender)
        gender_ids.append(GENDER_TO_ID[gender])
        asset_ids.append(f"{gender}_{beta_key}")

    d["motion_genders"] = tuple(genders)
    d["motion_gender_ids"] = torch.tensor(gender_ids, dtype=torch.long)
    d["motion_asset_ids"] = tuple(asset_ids)

    torch.save(d, path)

    from collections import Counter
    print(f"done  ({Counter(asset_ids)})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--base-name", required=True)
    parser.add_argument("--num-files", type=int, default=6)
    args = parser.parse_args()

    for i in range(args.num_files):
        path = os.path.join(args.dir, f"{args.base_name}_{i:04d}.pt")
        assert os.path.exists(path), f"Missing: {path}"
        fix_file(path)

    print("All files recovered.")


if __name__ == "__main__":
    main()
