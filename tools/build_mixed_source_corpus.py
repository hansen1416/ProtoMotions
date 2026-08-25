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
Build the combined canonical/AMASS + HUMOS corpus for source-conditioned rewards.

Tags every motion in --canonical-file with motion_source_id=0 (MOTION_SOURCE_CANONICAL)
and every motion in --humos-file with motion_source_id=1 (MOTION_SOURCE_HUMOS), then
concatenates the two packaged .pt files into one combined MotionLib file. Source-aware task
ids (``amass::<clip>`` and ``humos::<clip>``) isolate curriculum/evaluation grouping while
``motion_base_clip_ids`` preserves the cross-source clip identity for analysis. The two source
files must be index-aligned (same clip order and shape metadata per motion id); this tool
validates that contract before concatenating them.

    python tools/build_mixed_source_corpus.py \\
        --canonical-file /workspace/motion_cache/150_128shape_canonical/150_128shape_canonical_offset.pt \\
        --humos-file /workspace/motion_cache/small150_128shape.pt \\
        --output /workspace/motion_cache/150_128shape_mixed_source.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from protomotions.components.global_clip_pool import GlobalClipPool

MOTION_SOURCE_CANONICAL = 0
MOTION_SOURCE_HUMOS = 1
MOTION_SOURCE_NAMES = {
    MOTION_SOURCE_CANONICAL: "amass",
    MOTION_SOURCE_HUMOS: "humos",
}


def _base_clip_ids(data: dict, n: int, file_path: str) -> tuple[str, ...]:
    """Return the cross-source clip identity before adding the source namespace."""
    clip_ids = data.get("motion_base_clip_ids", data.get("motion_clip_ids"))
    if clip_ids is None:
        raise RuntimeError(
            f"{file_path} has no motion_clip_ids; source-aware curriculum grouping "
            "cannot be constructed."
        )
    if len(clip_ids) != n:
        raise RuntimeError(
            f"{file_path} has {len(clip_ids)} clip ids for {n} motions."
        )
    return tuple(str(clip_id) for clip_id in clip_ids)


def _tag_source(data: dict, source_id: int, file_path: str) -> dict:
    """Attach source metadata and make curriculum/evaluation task ids source-aware."""
    if source_id not in MOTION_SOURCE_NAMES:
        raise ValueError(f"Unknown motion source id: {source_id}")

    n = len(data["motion_num_frames"])
    source_name = MOTION_SOURCE_NAMES[source_id]
    base_clip_ids = _base_clip_ids(data, n, file_path)

    data["motion_source_id"] = torch.full((n,), source_id, dtype=torch.long)
    data["motion_base_clip_ids"] = base_clip_ids
    data["motion_clip_ids"] = tuple(
        f"{source_name}::{clip_id}" for clip_id in base_clip_ids
    )
    return data


def _validate_aligned_sources(canonical: dict, humos: dict) -> None:
    """Fail before concatenation if clip/shape rows do not represent the same grid."""
    n_canonical = len(canonical["motion_num_frames"])
    n_humos = len(humos["motion_num_frames"])
    if n_canonical != n_humos:
        raise RuntimeError(
            "Canonical and HUMOS corpora must contain the same clip-by-shape grid; "
            f"got {n_canonical} and {n_humos} motions."
        )

    tuple_fields = (
        "motion_base_clip_ids",
        "motion_asset_ids",
        "motion_beta_keys",
        "motion_genders",
    )
    for field in tuple_fields:
        canonical_value = canonical.get(field)
        humos_value = humos.get(field)
        if canonical_value is None and humos_value is None:
            continue
        if canonical_value is None or humos_value is None:
            raise RuntimeError(f"{field!r} is present in only one source corpus.")
        if len(canonical_value) != len(humos_value):
            raise RuntimeError(
                f"Source corpora have different lengths for {field!r}: "
                f"{len(canonical_value)} != {len(humos_value)}."
            )
        if tuple(canonical_value) != tuple(humos_value):
            mismatch = next(
                i
                for i, (left, right) in enumerate(zip(canonical_value, humos_value))
                if left != right
            )
            raise RuntimeError(
                f"Source corpora are not index-aligned at motion {mismatch} for "
                f"{field!r}: {canonical_value[mismatch]!r} != "
                f"{humos_value[mismatch]!r}."
            )

    tensor_fields = ("motion_gender_ids", "motion_betas")
    for field in tensor_fields:
        canonical_value = canonical.get(field)
        humos_value = humos.get(field)
        if canonical_value is None and humos_value is None:
            continue
        if canonical_value is None or humos_value is None:
            raise RuntimeError(f"{field!r} is present in only one source corpus.")
        if canonical_value.shape != humos_value.shape or not torch.allclose(
            canonical_value, humos_value
        ):
            raise RuntimeError(
                f"Source corpora are not index-aligned for tensor field {field!r}."
            )


def load_and_tag(file_path: str, source_id: int) -> dict:
    print(f"Loading {file_path} ...")
    data = torch.load(file_path, map_location="cpu", weights_only=False)
    _tag_source(data, source_id, file_path)
    n = len(data["motion_num_frames"])
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
    _validate_aligned_sources(canonical_dict, humos_dict)

    print("Concatenating ...")
    combined = GlobalClipPool._concat_clip_dicts([canonical_dict, humos_dict])

    n_canonical = len(canonical_dict["motion_num_frames"])
    n_humos = len(humos_dict["motion_num_frames"])
    n_total = len(combined["motion_num_frames"])
    assert n_total == n_canonical + n_humos, (
        f"Expected {n_canonical + n_humos} motions after concat, got {n_total}"
    )
    source_counts = torch.bincount(combined["motion_source_id"])
    task_count = len(set(combined["motion_clip_ids"]))
    base_clip_count = len(set(combined["motion_base_clip_ids"]))
    print(f"Combined: {n_total} motions total ({n_canonical} canonical + {n_humos} HUMOS)")
    print(f"motion_source_id counts: {source_counts.tolist()}")
    print(
        f"Task identities: {task_count} source-specific tasks from "
        f"{base_clip_count} base clips"
    )

    print(f"Saving to {args.output} ...")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(combined, args.output)
    print("Done.")


if __name__ == "__main__":
    main()
