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
Build the combined canonical/AMASS + HUMOS corpus for the source-switched reward
experiment (note/README.note.md Section 67).

Tags every motion in --canonical-file with motion_source_id=0 (MOTION_SOURCE_CANONICAL)
and every motion in --humos-file with motion_source_id=1 (MOTION_SOURCE_HUMOS), then
concatenates the two packaged .pt files into one combined MotionLib file. The two source
files are index-aligned (same clip order, same beta_key/gender ordering per motion_id --
verified in note/README.note.md Section 66/67), so this is a straightforward per-field
`torch.cat`, no remapping needed. Reuses `GlobalClipPool._concat_clip_dicts`'s field-list-
driven concat logic so the field handling stays identical to every other packaged-motion
merge in this pipeline.

    python tools/build_mixed_source_corpus.py \\
        --canonical-file /workspace/motion_cache/150_128shape_canonical/150_128shape_canonical_offset.pt \\
        --humos-file /workspace/motion_cache/small150_128shape.pt \\
        --output /workspace/motion_cache/150_128shape_mixed_source.pt
"""

from __future__ import annotations

import argparse

import torch

from protomotions.components.global_clip_pool import GlobalClipPool

MOTION_SOURCE_CANONICAL = 0
MOTION_SOURCE_HUMOS = 1


def load_and_tag(file_path: str, source_id: int) -> dict:
    print(f"Loading {file_path} ...")
    data = torch.load(file_path, map_location="cpu", weights_only=False)
    n = len(data["motion_num_frames"])
    data["motion_source_id"] = torch.full((n,), source_id, dtype=torch.long)
    print(f"  {n} motions, tagged motion_source_id={source_id}")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-file", required=True, help="Path to canonical/AMASS packaged .pt")
    parser.add_argument("--humos-file", required=True, help="Path to HUMOS packaged .pt")
    parser.add_argument("--output", required=True, help="Output path for combined .pt")
    args = parser.parse_args()

    canonical_dict = load_and_tag(args.canonical_file, MOTION_SOURCE_CANONICAL)
    humos_dict = load_and_tag(args.humos_file, MOTION_SOURCE_HUMOS)

    print("Concatenating ...")
    combined = GlobalClipPool._concat_clip_dicts([canonical_dict, humos_dict])

    n_canonical = len(canonical_dict["motion_num_frames"])
    n_humos = len(humos_dict["motion_num_frames"])
    n_total = len(combined["motion_num_frames"])
    assert n_total == n_canonical + n_humos, (
        f"Expected {n_canonical + n_humos} motions after concat, got {n_total}"
    )
    source_counts = torch.bincount(combined["motion_source_id"])
    print(f"Combined: {n_total} motions total ({n_canonical} canonical + {n_humos} HUMOS)")
    print(f"motion_source_id counts: {source_counts.tolist()}")

    print(f"Saving to {args.output} ...")
    torch.save(combined, args.output)
    print("Done.")


if __name__ == "__main__":
    main()
